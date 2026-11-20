"""Verify supplied signature copying and baseline-score integrity."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.apply_ensemble_signature import evaluate

target = 'static int M(int value) => value;'
source = {'one': {'id':'one', 'prompt':'Complete this C# method: static int M(int value)', 'target':target}}
ensemble = {'one': {'id':'one', 'selected':'static int Wrong(int value) => value;', 'ensemble_exact':False}}
rows = evaluate(source, ensemble)
assert rows[0]['selected'] == target and rows[0]['after_exact'] and not rows[0]['before_exact']
ensemble['one']['ensemble_exact'] = True
try:
    evaluate(source, ensemble)
    raise AssertionError('Inconsistent saved score accepted')
except ValueError:
    pass
print('Ensemble signature-copy checks passed.')
