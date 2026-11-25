"""Verify normalized final-level weighting and its gradients."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from models.quant import weighted_level_loss

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)
assert weighted_level_loss([a, b]).item() == 3.0
loss = weighted_level_loss([a, b], 3.0)
assert loss.item() == 3.5
loss.backward()
assert a.grad.item() == 0.25 and b.grad.item() == 0.75
for bad in (0, -1, float('nan'), float('inf')):
    try:
        weighted_level_loss([a, b], bad)
        raise AssertionError('Invalid weight accepted')
    except ValueError:
        pass
print('Final-level VQ weighting checks passed.')
