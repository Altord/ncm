"""Verify conditional scoring uses candidate tokens, excluding prompt loss."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import torch
from evaluation.rank_codec_candidates_lm import candidate_score

class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1, 2] if text.endswith('\n') else [3, 4]

class Model:
    def __call__(self, input_ids):
        logits = torch.zeros(1, input_ids.shape[1], 5)
        logits[0, 1, 3] = 5
        logits[0, 2, 4] = 5
        return SimpleNamespace(logits=logits)

score = candidate_score(Model(), Tokenizer(), 'prompt', 'candidate', torch.device('cpu'))
assert -0.1 < score < 0
print('Candidate ranker conditional scoring check passed.')
