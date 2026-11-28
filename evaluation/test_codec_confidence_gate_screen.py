"""Verify gate choices use margins and preserve equal-logit predictions."""
import torch
from evaluation.codec_confidence_gate_screen import candidates


def main():
    a = torch.tensor([[[3., 0.], [1., 0.], [2., 0.]]])
    b = torch.tensor([[[0., 2.], [0., 3.], [0., 2.]]])
    r = candidates(a, b)
    assert r['equal'].tolist() == [[0, 1, 0]]
    assert r['margin_one'].tolist() == [[0, 1, 0]]
    assert r['margin_two'].tolist() == [[1, 1, 1]]
    print('Confidence gate margin checks passed.')


if __name__ == '__main__': main()
