"""Check teacher averaging, top-k ordering and distillation behavior."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from training.prepare_ensemble_teacher_cache import topk_ensemble
from training.reconstruction_loss import topk_teacher_distillation_loss

a = torch.tensor([[[4., 2., 0.], [0., 3., 0.]]])
b = torch.tensor([[[0., 3., 0.], [0., 0., 5.]]])
ids, probs = topk_ensemble(a, b, 2)
assert ids.tolist() == [[[1, 0], [2, 1]]]
expected = torch.softmax(torch.tensor([2.5, 2.0]), dim=0)
assert torch.allclose(probs[0, 0], expected)
student = torch.randn(1, 2, 3, requires_grad=True)
target = torch.tensor([[1, 0]])
valid = torch.tensor([[True, True]])
loss, count = topk_teacher_distillation_loss(student, target, valid, [(ids, probs)])
assert count.item() == 1
loss.backward()
assert student.grad[0, 0].abs().sum() > 0 and student.grad[0, 1].abs().sum() == 0
try:
    topk_ensemble(a, b[:, :1], 2)
    raise AssertionError('Mismatched shapes accepted')
except ValueError:
    pass
print('Ensemble teacher cache checks passed.')
