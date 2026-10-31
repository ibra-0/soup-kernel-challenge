import torch
import triton
import triton.language as tl
 
 
@triton.jit
def _rope_kernel(
    X, COS, SIN, Y,
    sxb, sxh, sxs,
    syb, syh, sys_,
    scb, scs,
    S,
    HALF: tl.constexpr,
    BACKWARD: tl.constexpr,
):
    t = tl.program_id(0)          # token index in [0, B*S)
    h = tl.program_id(1)          # head
    b = t // S
    s = t % S
    d = tl.arange(0, HALF)
 
    xp = X + b * sxb + h * sxh + s * sxs
    yp = Y + b * syb + h * syh + s * sys_
    cp = COS + b * scb + s * scs
    sp = SIN + b * scb + s * scs
 
    x1 = tl.load(xp + d).to(tl.float32)
    x2 = tl.load(xp + HALF + d).to(tl.float32)
    c1 = tl.load(cp + d).to(tl.float32)
    c2 = tl.load(cp + HALF + d).to(tl.float32)
    s1 = tl.load(sp + d).to(tl.float32)
    s2 = tl.load(sp + HALF + d).to(tl.float32)
 
    if BACKWARD:
        # transpose of the rotation
        y1 = x1 * c1 + x2 * s2
        y2 = x2 * c2 - x1 * s1
    else:
        # q*cos + cat(-q2, q1)*sin
        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2
 
    tl.store(yp + d, y1.to(Y.dtype.element_ty))
    tl.store(yp + HALF + d, y2.to(Y.dtype.element_ty))
 
 
def _launch(x, cos, sin, backward, out_dtype):
    # x: [B, H, S, D] (any strides, last stride 1); cos / sin: [B or 1, S, D]
    if x.stride(-1) != 1:
        x = x.contiguous()
    B, H, S, D = x.shape
    y = torch.empty_like(x, dtype=out_dtype)  # keeps x's memory layout
    if cos.stride(-1) != 1:
        cos = cos.contiguous()
    if sin.stride(-1) != 1:
        sin = sin.contiguous()
    scb = 0 if cos.shape[0] == 1 else cos.stride(0)
    _rope_kernel[(B * S, H)](
        x, cos, sin, y,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        scb, cos.stride(1),
        S,
        HALF=D // 2,
        BACKWARD=backward,
        num_warps=1,
    )
    return y
 
 
class _Rope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        ctx.x_dtype = x.dtype
        return _launch(x, cos, sin, False, x.dtype)
 
    @staticmethod
    def backward(ctx, dy):
        cos, sin = ctx.saved_tensors
        return _launch(dy, cos, sin, True, ctx.x_dtype), None, None
 
 
def rope(x, cos, sin):
    return _Rope.apply(x, cos, sin)
 
 
def _supported(q, cos, sin, args, kwargs):
    if args or set(kwargs) - {"unsqueeze_dim"}:
        return False
    if kwargs.get("unsqueeze_dim", 1) != 1:
        return False
    if not (q.is_cuda and q.ndim == 4 and cos.ndim == 3 and cos.shape == sin.shape):
        return False
    d = q.shape[-1]
    half = d // 2
    return d % 2 == 0 and (half & (half - 1)) == 0 and cos.shape[-1] == d
 
 
def patch_qwen2_rope():
    from transformers.models.qwen2 import modeling_qwen2 as module
 
    if getattr(module, "_custom_rope_patched", False):
        return
 
    original = module.apply_rotary_pos_emb
 
    def apply_rotary_pos_emb(q, k, cos, sin, *args, **kwargs):
        if not _supported(q, cos, sin, args, kwargs) or not k.is_cuda:
            return original(q, k, cos, sin, *args, **kwargs)
        return rope(q, cos, sin), rope(k, cos, sin)
 
    module.apply_rotary_pos_emb = apply_rotary_pos_emb
    module._custom_rope_patched = True
 
