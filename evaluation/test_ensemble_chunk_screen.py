"""Check fixed-token chunk coverage and invalid sizes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.ensemble_chunk_screen import partition

assert partition(list(range(5)), 2) == [[0, 1], [2, 3], [4]]
assert partition([], 2) == [[]]
try:
    partition([1], 0)
    raise AssertionError('Invalid chunk size accepted')
except ValueError:
    pass
print('Fixed-token chunk checks passed.')
