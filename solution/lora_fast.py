"""LoRA input path: one Triton kernel instead of cast->fp32, fp32 dropout, cast->fp16.

peft's LoRA forward does  result + lora_B(lora_A(dropout(x.to(fp32)))) * scaling.
On this task that is, per q_proj / v_proj and per direction: an fp32 copy of x, an fp32
dropout (+ a stored mask), and an fp16 copy for the autocast matmul. Here the dropout (same
p, same 1/(1-p) rescale) is one kernel that reads x and writes fp16; the mask is never stored:
it is regenerated from a seed in backward (Philox, tl.rand).

Only the random stream differs from torch.nn.Dropout; LoRA rank, alpha, dropout p, the
trainable parameters and the saved adapter are unchanged.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def _dropout_scale_kernel(X, Y, n, seed, p, mult, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    keep = tl.rand(seed, offs) >= p
    y = tl.where(keep, x * mult, 0.0)
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


class _DropScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, mult, out_dtype):
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if p > 0 else 0
        xc = x.contiguous()
        y = torch.empty(xc.shape, device=xc.device, dtype=out_dtype)
        n = xc.numel()
        _dropout_scale_kernel[(triton.cdiv(n, BLOCK),)](xc, y, n, seed, p, mult, BLOCK=BLOCK)
        ctx.cfg = (seed, p, mult, x.dtype)
        return y

    @staticmethod
    def backward(ctx, dy):
        seed, p, mult, x_dtype = ctx.cfg
        dyc = dy.contiguous()
        dx = torch.empty(dyc.shape, device=dyc.device, dtype=x_dtype)
        n = dyc.numel()
        _dropout_scale_kernel[(triton.cdiv(n, BLOCK),)](dyc, dx, n, seed, p, mult, BLOCK=BLOCK)
        return dx, None, None, None


class FusedLoraDropout(torch.nn.Module):
    """Drop-in for nn.Dropout(p) inside a peft LoRA layer (no parameters)."""

    def __init__(self, p):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not x.is_cuda or x.numel() >= 2**31:
            return torch.nn.functional.dropout(x, self.p, self.training)
        p = self.p if self.training else 0.0
        mult = 1.0 / (1.0 - p) if p > 0 else 1.0
        out_dtype = torch.float16 if torch.is_autocast_enabled("cuda") else x.dtype
        return _DropScale.apply(x, p, mult, out_dtype)


def patch_lora_dropout():
    from peft.tuners.lora import layer as lora_layer

    cls = lora_layer.LoraLayer
    if getattr(cls, "_custom_lora_patched", False):
        return

    original_update = cls.update_layer

    def update_layer(self, *args, **kwargs):
        # peft passes some of these by keyword, so do not rely on the signature
        out = original_update(self, *args, **kwargs)
        adapter_name = args[0] if args else kwargs.get("adapter_name")
        drop = self.lora_dropout[adapter_name]
        if type(drop) is torch.nn.Dropout and not kwargs.get("use_dora", False):
            self.lora_dropout[adapter_name] = FusedLoraDropout(drop.p)
        return out

    cls.update_layer = update_layer

    # peft casts x to the (fp32) adapter dtype before dropout; under fp16 autocast the matmul
    # casts it back anyway, so skip the round trip when x is already fp16.
    if hasattr(cls, "_cast_input_dtype"):
        original_cast = cls._cast_input_dtype

        def _cast_input_dtype(self, x, dtype):
            if x is not None and x.is_cuda and x.dtype == torch.float16 \
                    and torch.is_autocast_enabled("cuda"):
                return x
            return original_cast(self, x, dtype)

        cls._cast_input_dtype = _cast_input_dtype

    cls._custom_lora_patched = True
