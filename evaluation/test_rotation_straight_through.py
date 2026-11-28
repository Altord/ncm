"""Check the experimental aggregate rotation estimator's forward and gradient."""
import torch
from models.quant import rotation_straight_through


def main():
    encoded = torch.tensor([[1., 0.], [0., 0.], [1., 0.]], requires_grad=True)
    quantized = torch.tensor([[0., 2.], [1., 2.], [-1., 0.]])
    output = rotation_straight_through(encoded, quantized)
    torch.testing.assert_close(output, quantized)
    output.backward(torch.tensor([[0., 1.], [3., 4.], [5., 6.]]))
    torch.testing.assert_close(encoded.grad[0], torch.tensor([2., 0.]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(encoded.grad[1:], torch.tensor([[3., 4.], [5., 6.]]))
    assert torch.isfinite(encoded.grad).all()
    print('Rotation estimator forward, gradient, and degenerate checks passed.')


if __name__ == '__main__':
    main()
