"""Check repeated-piece weights stay inside each example and ignore padding."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from training.reconstruction_loss import repeated_identifier_piece_factors
from training.train_tokenizer import reconstruction_ce

targets = torch.tensor([[4, 4, 5, 5, 4, 9], [4, 6, 6, 4, 9, 9]])
valid = torch.tensor([[1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
labels = torch.tensor([[1.5, 1.5, 3., 1.5, 1.5, 1.], [1.5, 1.5, 1.5, 3., 1.5, 1.5]])
factors = repeated_identifier_piece_factors(targets, valid, labels, 1.5, 1.5)
expected = torch.tensor([[1.5, 1.5, 1., 1., 1., 1.], [1., 1.5, 1.5, 1., 1., 1.]])
assert torch.equal(factors, expected)
logits = torch.randn(2, 6, 10, requires_grad=True)
_, loss = reconstruction_ce(logits, targets, valid, labels * factors)
loss.backward()
assert torch.isfinite(loss) and logits.grad[~valid].abs().sum() == 0
print('Repeated identifier-piece weighting CPU check passed.')
