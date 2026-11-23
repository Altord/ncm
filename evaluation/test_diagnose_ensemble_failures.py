"""Check source alignment, grouping and region accounting."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.diagnose_ensemble_failures import analyze

source = {'a': {'target': 'int M() => 2;', 'source': 'snake/A.cs'},
          'b': {'target': 'int N() => 1;', 'source': 'flock/B.cs'}}
ensemble = {'a': {'selected': 'int M() => 3;', 'ensemble_exact': False},
            'b': {'selected': 'int N() => 1;', 'ensemble_exact': True}}
result = analyze(source, ensemble, {'a': 10, 'b': 90})
assert result['all']['exact'] == 1
assert result['project:snake']['exact'] == 0
assert result['project:snake']['methods_with_change_by_region']['literal'] == 1
assert result['length:65-128']['exact'] == 1
try:
    analyze(source, ensemble, {'a': 10})
    raise AssertionError('Mismatched IDs accepted')
except ValueError:
    pass
print('Ensemble failure diagnostic checks passed.')
