import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_q8_matches_pytorch(shape):
    from solution.swiglu_q8 import swiglu_q8

    torch.manual_seed(0)
    gate = (torch.randn(shape, device="cuda") * 2).half().requires_grad_(True)
    up = (torch.randn(shape, device="cuda") * 2).half().requires_grad_(True)
    grad = torch.randn(shape, device="cuda").half()

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu_q8(gate, up)
    out.backward(grad)

    # forward is exact (only the saved copies are int8)
    torch.testing.assert_close(out.float(), ref, rtol=2e-3, atol=2e-3)
    # backward uses int8 copies: error stays a small fraction of the largest gradient
    for got, want in ((gate.grad.float(), ref_dgate), (up.grad.float(), ref_dup)):
        rel = (got - want).abs().max() / want.abs().max()
        assert rel < 0.03, rel
    assert gate.grad.dtype == torch.float16 and up.grad.dtype == torch.float16
