"""Check source grouping, calibration ties and no-label inference selection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.codec_latent_refiner_gate import choose_threshold, select, split_source, summarize

rows = [
 {'changed': True, 'confidence_delta': 0.4, 'baseline_exact': False, 'adapter_exact': True},
 {'changed': True, 'confidence_delta': 0.1, 'baseline_exact': True, 'adapter_exact': False},
 {'changed': False, 'confidence_delta': 2.0, 'baseline_exact': True, 'adapter_exact': True},
]
threshold = choose_threshold(rows)
assert threshold == 0.4
assert [select(r, threshold) for r in rows] == [True, False, False]
summary = summarize(rows, threshold)
assert summary['baseline_exact'] == 2 and summary['selected_exact'] == 3
assert summary['gained_exact'] == 1 and summary['lost_exact'] == 0
assert choose_threshold([rows[1]]) is None
assert split_source('project/file.cs') == split_source('project/file.cs')
try:
    split_source('')
    raise AssertionError('Missing source accepted')
except ValueError:
    pass
print('Latent adapter gate checks passed.')
