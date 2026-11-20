"""Check compile reuse, new calls, and rejection of conflicting results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation.compile_codec_ensemble as screen

source = {'x': {'target': 'target'}}
ensemble = {'x': {'selected': 'new'}}
def row(text, status):
    return {'target': 'target', 'reconstructed': text, 'original_compile': {'status': 'passed'},
            'reconstruction_compile': {'status': status}}
a, b = {'x': row('a', 'failed')}, {'x': row('b', 'passed')}
calls = []
screen.compile_candidate = lambda text, framework: calls.append(text) or {'status': 'passed'}
rows, count = screen.compile_rows(source, ensemble, a, b)
assert count == 1 and calls == ['new'] and rows[0]['compile_status'] == 'passed'
ensemble['x']['selected'] = 'b'
rows, count = screen.compile_rows(source, ensemble, a, b)
assert count == 0 and rows[0]['compile_status'] == 'passed'
b['x'] = row('a', 'passed')
try:
    screen.compile_rows(source, ensemble, a, b)
    raise AssertionError('Conflicting compiler statuses accepted')
except ValueError:
    pass
print('Ensemble compile checks passed.')
