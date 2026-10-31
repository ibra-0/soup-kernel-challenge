import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _ref_rope(x, cos, sin):
    x = x.float()
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos.unsqueeze(1) + rot * sin.unsqueeze(1)


@pytest.mark.parametrize("heads", [12, 2])
@pytest.mark.parametrize("batch,seq", [(2, 77), (4, 512)])
def test_rope_matches_reference(heads, batch, seq):
    from solution.rope import rope

    torch.manual_seed(0)
    d = 128
    base = torch.randn(batch, seq, heads, d, device="cuda", dtype=torch.float16)
    x = base.transpose(1, 2).requires_grad_(True)  # same layout as q after view+transpose
    cos = torch.randn(batch, seq, d, device="cuda", dtype=torch.float32)
    sin = torch.randn(batch, seq, d, device="cuda", dtype=torch.float32)
    grad = torch.randn(batch, heads, seq, d, device="cuda", dtype=torch.float16)

    ref_in = x.detach().float().requires_grad_(True)
    ref = _ref_rope(ref_in, cos, sin)
    ref.backward(grad.float())

    out = rope(x, cos, sin)
    out.backward(grad)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(x.grad.float(), ref_in.grad, atol=2e-3, rtol=2e-3)


def test_lora_dropout_statistics_and_backward():
    from solution.lora_fast import FusedLoraDropout

    torch.manual_seed(0)
    drop = FusedLoraDropout(0.05).cuda().train()
    x = (torch.randn(4, 512, 1536, device="cuda") + 3).half().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.float16):
        y = drop(x)
    assert y.dtype == torch.float16
    kept = y != 0
    assert abs(kept.float().mean().item() - 0.95) < 0.005
    torch.testing.assert_close(y[kept].float(), (x[kept].float() / 0.95).half().float(), atol=2e-3, rtol=2e-3)
    y.backward(torch.ones_like(y))
    # backward uses exactly the forward mask
    assert torch.equal(x.grad != 0, kept)

    drop.eval()
    with torch.autocast("cuda", dtype=torch.float16):
        assert torch.equal(drop(x), x)
