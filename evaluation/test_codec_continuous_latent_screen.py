import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from evaluation.codec_continuous_latent_screen import evaluate_predictions

target = torch.tensor([[1, 2, 0], [3, 4, 5]])
mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
prediction = torch.tensor([[1, 9, 9], [3, 4, 5]])
assert evaluate_predictions(prediction, target, mask) == (4, 5)
try:
    evaluate_predictions(prediction[:, :2], target, mask)
    raise AssertionError('Shape mismatch accepted')
except ValueError:
    pass
print('Continuous-latent metric checks passed.')
