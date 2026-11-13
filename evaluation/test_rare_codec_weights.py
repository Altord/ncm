"""Check train-only frequency weights, mask handling and existing loss interaction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from training.reconstruction_loss import training_token_frequency_weights, select_reconstruction_weights
from training.train_tokenizer import reconstruction_ce


class Cache:
    _arrays = {'tokens': np.array([[1, 1, 2, 7], [1, 3, 7, 7]], dtype=np.int32),
               'mask': np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.uint8)}
    def __len__(self):
        return 2


weights, counts = training_token_frequency_weights(Cache(), 8, threshold=4, max_weight=2.0)
assert counts.tolist() == [0, 3, 1, 1, 0, 0, 0, 0]
assert torch.allclose(weights[[1, 2, 3, 7]], torch.tensor([(4/3)**0.5, 2., 2., 1.]))
targets = torch.tensor([[1, 2, 7]])
occurrences = torch.tensor([[3., 1.5, 1.]])
combined = select_reconstruction_weights(targets, None, occurrences) * weights[targets]
logits = torch.randn(1, 3, 8, requires_grad=True)
raw, weighted = reconstruction_ce(logits, targets, torch.tensor([[1, 1, 0]], dtype=torch.bool), combined)
weighted.backward()
assert torch.isfinite(raw) and torch.isfinite(weighted) and logits.grad[0, 2].abs().sum() == 0
print('Rare token weight CPU check passed.')
