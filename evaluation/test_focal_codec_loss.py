"""Check optional token focal loss, baseline compatibility and padding."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from training.train_tokenizer import reconstruction_ce

logits = torch.tensor([[[4., 0.], [0., 4.], [2., -2.]]], requires_grad=True)
targets = torch.tensor([[0, 0, 1]])
mask = torch.tensor([[True, True, False]])
raw, normal = reconstruction_ce(logits, targets, mask)
raw2, focal = reconstruction_ce(logits, targets, mask, focal_gamma=1.0)
assert torch.allclose(raw, raw2) and torch.allclose(raw, normal)
assert 0 < focal < normal
focal.backward()
assert logits.grad[0, 1].abs().sum() > logits.grad[0, 0].abs().sum()
assert logits.grad[0, 2].abs().sum() == 0
print('Token focal loss CPU check passed.')
