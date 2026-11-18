"""Check true-token ranks, including documented optimistic ties."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from evaluation.measure_method_token_ranks import optimistic_rank

scores = torch.tensor([3.0, 2.0, 2.0, -1.0])
assert [optimistic_rank(scores, i) for i in range(4)] == [1, 2, 2, 4]
print('Method token-rank checks passed.')
