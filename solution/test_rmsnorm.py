import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 235, 1536), (3, 77, 1001)])
def test_rmsnorm_forward_and_backward(shape, dtype):
    from solution.rmsnorm import rms_norm

    torch.manual_seed(0)
    eps = 1e-6
    x0 = torch.randn(shape, device="cuda", dtype=dtype)
    w0 = torch.randn(shape[-1], device="cuda", dtype=torch.float32)
    grad = torch.randn(shape, device="cuda", dtype=torch.float32)

    x_ref = x0.detach().clone().requires_grad_(True)
    w_ref = w0.detach().clone().requires_grad_(True)
    x32 = x_ref.float()
    inv = torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)
    ref = (x32 * inv).to(dtype) * w_ref
    ref.backward(grad)

    x_test = x0.detach().clone().requires_grad_(True)
    w_test = w0.detach().clone().requires_grad_(True)
    out = rms_norm(x_test, w_test, eps)
    out.backward(grad)

    tol = 2e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)
    torch.testing.assert_close(x_test.grad, x_ref.grad, rtol=tol, atol=tol)
    torch.testing.assert_close(w_test.grad, w_ref.grad, rtol=tol, atol=tol)
