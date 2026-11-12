"""Check hard-example objective preserves ordinary loss and ignores padding."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from training.train_tokenizer import reconstruction_ce

logits = torch.tensor([[[4., 0.], [0., 4.]], [[0., 4.], [0., 4.]],
                       [[10., -10.], [10., -10.]]], requires_grad=True)
targets = torch.tensor([[0, 1], [0, 1], [1, 1]])
mask = torch.tensor([[True, True], [True, True], [False, False]])
weights = torch.tensor([[1., 2.], [1., 2.], [100., 100.]])
raw, normal = reconstruction_ce(logits, targets, mask, weights)
raw2, hard = reconstruction_ce(logits, targets, mask, weights, hard_fraction=0.5, hard_mix=1.0)
assert torch.allclose(raw, raw2) and hard > normal
hard.backward()
assert logits.grad[1].abs().sum() > 0 and logits.grad[0].abs().sum() == 0
assert logits.grad[2].abs().sum() == 0
print('Hard-example loss CPU check passed.')
