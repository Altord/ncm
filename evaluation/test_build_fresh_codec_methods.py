"""Check deterministic lexical-overlap screening."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.build_fresh_codec_methods import build_index, compact, near_overlap

known = ' '.join(f'identifier{i}' for i in range(30))
index = build_index([known])
assert near_overlap(known, index)
assert not near_overlap(' '.join(f'unrelated{i}' for i in range(30)), index)
assert not near_overlap('short method', index)
assert compact('a \n b') == 'ab'
print('External method overlap checks passed.')
