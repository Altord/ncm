"""Check frozen-teacher error weighting excludes padding and validates multiplier."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from training.train_tokenizer import teacher_error_factors, reconstruction_ce

target = torch.tensor([[1, 2, 3]])
predicted = torch.tensor([[1, 7, 8]])
valid = torch.tensor([[True, True, False]])
assert teacher_error_factors(predicted, target, valid, 2).tolist() == [[1, 2, 1]]
logits = torch.zeros((1, 3, 10), requires_grad=True)
raw, weighted = reconstruction_ce(logits, target, valid, teacher_error_factors(predicted, target, valid, 2))
assert torch.allclose(raw, weighted)
weighted.backward()
assert logits.grad[0, 1].abs().sum() > logits.grad[0, 0].abs().sum()
assert logits.grad[0, 2].abs().sum() == 0
for invalid in (0.5, float('nan'), float('inf')):
    try:
        teacher_error_factors(predicted, target, valid, invalid)
        raise AssertionError('Invalid multiplier accepted')
    except ValueError:
        pass
print('Codec teacher-error weighting checks passed.')
