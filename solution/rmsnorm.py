"""Fused RMSNorm forward and Triton backward kernels for Qwen2."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_forward_fused(
    X, W, Y, R,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < H

    x = tl.load(X + row * H + col, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / H
    inv_rms = tl.rsqrt(mean_square + EPS)
    w = tl.load(W + col, mask=mask, other=0.0).to(tl.float32)

    normalized = (x * inv_rms).to(X.dtype.element_ty).to(tl.float32)
    y = normalized * w

    tl.store(Y + row * H + col, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(R + row, inv_rms)


@triton.jit
def _rmsnorm_backward_dx(
    X, W, DY, R, DX,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < H

    x = tl.load(X + row * H + col, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + col, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * H + col, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.load(R + row)

    xhat = x * inv_rms
    grad = dy * w
    projection = tl.sum(grad * xhat, axis=0) / H
    dx = (grad - xhat * projection) * inv_rms
    tl.store(DX + row * H + col, dx, mask=mask)


@triton.jit
def _rmsnorm_backward_dw(
    X, DY, R, DW,
    N: tl.constexpr,
    H: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BR)
    cols = pid * BC + tl.arange(0, BC)
    acc = tl.zeros((BC,), dtype=tl.float32)

    for start in range(0, N, BR):
        rr = start + rows
        row_mask = rr < N
        col_mask = cols < H
        x = tl.load(
            X + rr[:, None] * H + cols[None, :],
            row_mask[:, None] & col_mask[None, :],
            0.0,
        ).to(tl.float32)
        dy = tl.load(
            DY + rr[:, None] * H + cols[None, :],
            row_mask[:, None] & col_mask[None, :],
            0.0,
        ).to(tl.float32)
        inv_rms = tl.load(R + rr, row_mask, 0.0)
        normalized = (x * inv_rms[:, None]).to(X.dtype.element_ty).to(tl.float32)
        acc += tl.sum(dy * normalized, axis=0)

    tl.store(DW + cols, acc, cols < H)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, out_fp16=False):
        n, h = x.shape
        block = triton.next_power_of_2(h)
        inv_rms = torch.empty((n,), device=x.device, dtype=torch.float32)
        y = torch.empty(x.shape, device=x.device,
                        dtype=torch.float16 if out_fp16 else weight.dtype)

        _rmsnorm_forward_fused[(n,)](
            x, weight, y, inv_rms,
            h, eps, block,
        )

        ctx.save_for_backward(x, weight, inv_rms)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, inv_rms = ctx.saved_tensors
        n, h = x.shape
        block = triton.next_power_of_2(h)
        grad_output = grad_output.contiguous()

        dx = None
        dw = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty_like(x)
            _rmsnorm_backward_dx[(n,)](
                x, weight, grad_output, inv_rms, dx,
                h, block,
            )

        if ctx.needs_input_grad[1]:
            dw = torch.empty_like(weight)
            _rmsnorm_backward_dw[(triton.cdiv(h, 128),)](
                x, grad_output, inv_rms, dw,
                n, h, 16, 128,
            )

        return dx, dw, None, None


def rms_norm(x, weight, eps, out_fp16=False):
    original_shape = x.shape
    hidden_size = original_shape[-1]
    x = x.contiguous().view(-1, hidden_size)
    result = _RMSNorm.apply(x, weight, eps, out_fp16)
    if out_fp16:
        return result.view(original_shape)
    return result.to(weight.dtype).view(original_shape)


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    if getattr(cls, "_custom_triton_patched", False):
        return

    original = cls.forward

    def forward(self, hidden_states):
        if not hidden_states.is_cuda:
            return original(self, hidden_states)
        out_fp16 = getattr(self, "_out_fp16", False) and torch.is_autocast_enabled("cuda")
        return rms_norm(hidden_states, self.weight, self.variance_epsilon, out_fp16)

    cls.forward = forward
    cls._custom_triton_patched = True

    # mark the two norms inside every decoder layer (the final model.norm keeps fp32 output)
    layer_cls = modeling_qwen2.Qwen2DecoderLayer
    if not getattr(layer_cls, "_custom_norm_flag_patched", False):
        original_init = layer_cls.__init__

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.input_layernorm._out_fp16 = True
            self.post_attention_layernorm._out_fp16 = True

        layer_cls.__init__ = __init__
        layer_cls._custom_norm_flag_patched = True
