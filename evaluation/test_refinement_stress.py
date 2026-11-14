"""Verify gating decisions and shape on a small CPU example."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from evaluation.evaluate_refinement_stress import gate_predictions

logits = torch.tensor([[[2.0, 1.0, 0.0], [4.0, 0.0, 0.0]]])
refined = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 3.0, 0.0]]])
assert gate_predictions(logits, refined, 2.0).tolist() == [[2, 0]]
assert gate_predictions(logits, refined, -1.0).tolist() == [[0, 0]]
assert gate_predictions(logits, refined, float('inf')).tolist() == [[2, 1]]
print('Refinement stress gate checks passed.')
