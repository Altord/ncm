"""Exercise signature compile reuse and compile-first fallback."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation.compile_ensemble_signature as screen
import evaluation.evaluate_pointer_stress as stress

target = 'static int M(int value) => value;'
before = 'static int Wrong(int value) => value;'
source = {'one': {'id':'one', 'target':target}}
ensemble = {'one': {'id':'one', 'selected':before, 'compile_status':'passed', 'exact_text':False}}
reference = {'one': {'id':'one', 'target':target, 'reconstructed':before}}
hybrid = {'one': {'id':'one', 'baseline':{'text':before},
                  'select_strict':{'text':'bad', 'compile_status':'failed'}}}
calls = []
stress.compile_candidate = lambda text, framework: calls.append(text) or {'status':'passed'}
rows, count = screen.evaluate(source, ensemble, hybrid, reference)
assert count == 1 and calls == [target]
assert rows[0]['selected_exact'] and rows[0]['used_ensemble_fallback'] and rows[0]['combined_exact']
hybrid['one']['select_strict'] = {'text':'old', 'compile_status':'passed'}
rows, count = screen.evaluate(source, ensemble, hybrid, reference)
assert rows[0]['combined'] == 'old' and not rows[0]['used_ensemble_fallback']
print('Ensemble signature compile checks passed.')
