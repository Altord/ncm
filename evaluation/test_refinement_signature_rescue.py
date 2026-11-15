"""Check compiler-only rescue and unchanged successful prior candidates."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation.refinement_signature_rescue as rescue
import evaluation.evaluate_pointer_stress as stress

target = 'static int M(int x) => x;'
bad = 'static int Other(int x) => x;'
source = {'a': {'target': target}, 'b': {'target': target}}
hybrid = {'a': {'baseline': {'text': bad}, 'select_strict': {'text': bad, 'compile_status': 'failed'}},
          'b': {'baseline': {'text': bad}, 'select_strict': {'text': bad, 'compile_status': 'passed'}}}
head = {key: {'baseline': bad, 'selected': bad, 'selected_compile': status}
        for key, status in [('a', 'failed'), ('b', 'passed')]}
calls = []
stress.compile_candidate = lambda text, framework: calls.append(text) or {'status': 'passed'}
rows, count = rescue.evaluate(source, hybrid, (head, head))
assert count == 1 and calls == [target]
assert rows[0]['selected_exact'] and rows[0]['selected']['compile_status'] == 'passed'
assert rows[1]['selected']['text'] == bad and rows[1]['selected']['selection'] == 'earlier_hybrid'
print('Refinement signature rescue checks passed.')
