"""Check ensemble logits and mismatch guard without using target labels."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from evaluation.ensemble_codec_methods import combine

a = torch.tensor([[[4., 2., 0.], [0., 1., 0.]]])
b = torch.tensor([[[0., 3., 0.], [0., 0., 4.]]])
assert combine(a, b).tolist() == [[1, 2]]
assert combine(a, b, torch.tensor([[[10., 0., 0.], [0., 8., 0.]]])).tolist() == [[0, 1]]
try:
    combine(a, b[:, :1])
    raise AssertionError('Mismatched logits accepted')
except ValueError:
    pass
print('Two-codec ensemble checks passed.')
