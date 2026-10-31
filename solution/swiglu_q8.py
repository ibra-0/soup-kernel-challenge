"""SwiGLU with int8-compressed saved activations (Triton), for the Qwen2 MLP.

What it does differently from a plain fused SwiGLU:
  * forward computes silu(gate) * up exactly (fp32 math, fp16 output), and in the same pass
    stores `gate` and `up` for backward as int8 with one fp32 scale per token row
    (absmax / 127). The fp16 `gate` and `up` are then free to be released: 2 x 2 bytes per
    element are replaced by 2 x 1 byte (+ 8 bytes per row).
  * backward dequantises those int8 values and applies the SwiGLU derivative.
Only the saved activations are lossy (about 0.4% of a row's largest value per element);
the forward result is unchanged. This is the "activation compression" direction of the task
statement. Idea source: absmax int8 quantisation as in LLM.int8() / bitsandbytes; the kernels
and the autograd wrapper are written from scratch for this task.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _swiglu_q8_forward(G, U, O, QG, QU, SG, SU, H, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = row * H
    lanes = tl.arange(0, BLOCK)

    # pass 1: absmax of the row for gate and for up
    peak_g = tl.zeros([BLOCK], dtype=tl.float32)
    peak_u = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, H, BLOCK):
        cols = start + lanes
        ok = cols < H
        g = tl.load(G + base + cols, mask=ok, other=0.0).to(tl.float32)
        u = tl.load(U + base + cols, mask=ok, other=0.0).to(tl.float32)
        peak_g = tl.maximum(peak_g, tl.abs(g))
        peak_u = tl.maximum(peak_u, tl.abs(u))
    scale_g = tl.maximum(tl.max(peak_g, axis=0), 1e-6) / 127.0
    scale_u = tl.maximum(tl.max(peak_u, axis=0), 1e-6) / 127.0
    inv_g = 1.0 / scale_g
    inv_u = 1.0 / scale_u
    tl.store(SG + row, scale_g)
    tl.store(SU + row, scale_u)

    # pass 2: exact output + int8 copies (round half away from zero)
    for start in range(0, H, BLOCK):
        cols = start + lanes
        ok = cols < H
        g = tl.load(G + base + cols, mask=ok, other=0.0).to(tl.float32)
        u = tl.load(U + base + cols, mask=ok, other=0.0).to(tl.float32)
        out = g * tl.sigmoid(g) * u
        tl.store(O + base + cols, out.to(O.dtype.element_ty), mask=ok)
        qg = (g * inv_g + tl.where(g >= 0, 0.5, -0.5)).to(tl.int8)
        qu = (u * inv_u + tl.where(u >= 0, 0.5, -0.5)).to(tl.int8)
        tl.store(QG + base + cols, qg, mask=ok)
        tl.store(QU + base + cols, qu, mask=ok)


@triton.jit
def _swiglu_q8_backward(DO, QG, QU, SG, SU, DG, DU, H, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = cols < H
    off = row * H + cols

    g = tl.load(QG + off, mask=ok, other=0).to(tl.float32) * tl.load(SG + row)
    u = tl.load(QU + off, mask=ok, other=0).to(tl.float32) * tl.load(SU + row)
    d = tl.load(DO + off, mask=ok, other=0.0).to(tl.float32)

    s = tl.sigmoid(g)
    tl.store(DG + off, (d * u * s * (1.0 + g * (1.0 - s))).to(DG.dtype.element_ty), mask=ok)
    tl.store(DU + off, (d * g * s).to(DU.dtype.element_ty), mask=ok)


class _SwiGLUQ8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        shape = gate.shape
        width = shape[-1]
        g = gate.contiguous().view(-1, width)
        u = up.to(gate.dtype).contiguous().view(-1, width)
        rows = g.shape[0]
        out = torch.empty_like(g)
        qg = torch.empty((rows, width), device=g.device, dtype=torch.int8)
        qu = torch.empty((rows, width), device=g.device, dtype=torch.int8)
        sg = torch.empty(rows, device=g.device, dtype=torch.float32)
        su = torch.empty(rows, device=g.device, dtype=torch.float32)
        _swiglu_q8_forward[(rows,)](g, u, out, qg, qu, sg, su, width, BLOCK=_BLOCK, num_warps=8)
        ctx.save_for_backward(qg, qu, sg, su)
        ctx.meta = (shape, g.dtype)
        return out.view(shape)

    @staticmethod
    def backward(ctx, grad_out):
        qg, qu, sg, su = ctx.saved_tensors
        shape, dtype = ctx.meta
        rows, width = qg.shape
        d = grad_out.to(dtype).contiguous().view(rows, width)
        dg = torch.empty_like(d)
        du = torch.empty_like(d)
        grid = (rows, triton.cdiv(width, _BLOCK))
        _swiglu_q8_backward[grid](d, qg, qu, sg, su, dg, du, width, BLOCK=_BLOCK)
        return dg.view(shape), du.view(shape)


def swiglu_q8(gate, up):
    return _SwiGLUQ8.apply(gate, up)


def patch_qwen2_mlp():
    from transformers.models.qwen2 import modeling_qwen2

    mlp = modeling_qwen2.Qwen2MLP
    if getattr(mlp, "_swiglu_q8_patched", False):
        return
    original = mlp.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or not x.is_cuda:
            return original(self, x)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        compress = (
            torch.is_grad_enabled()
            and (gate.requires_grad or up.requires_grad)
            and gate.dtype == torch.float16
            and up.dtype == torch.float16
        )
        if not compress:
            return self.down_proj(torch.nn.functional.silu(gate) * up)
        return self.down_proj(swiglu_q8(gate, up))

    mlp.forward = forward
    mlp._swiglu_q8_patched = True
