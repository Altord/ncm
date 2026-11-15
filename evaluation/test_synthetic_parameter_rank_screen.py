"""Check synthetic mutations affect a parameter reference rather than a literal."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.synthetic_parameter_rank_screen import build_case

target = 'static int Subtract(int first, int second) => first - second;'
candidates = build_case({'target': target})
assert candidates and target in candidates and len(candidates) == 2
assert 'static int Subtract(int first, int second) => second - second;' in candidates
assert build_case({'target': 'static int M(int first, float second) => first;'}) is None
print('Synthetic parameter ranking checks passed.')
