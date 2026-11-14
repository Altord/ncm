"""Check that supplied signatures combine with pointer text and reuse compiles."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.data import sha256_file
import evaluation.evaluate_pointer_stress as stress
import evaluation.evaluate_pointer_signature_hybrid as hybrid

target = 'static int M(int x) => x;'
wrong = 'static int Miss(int x) => x;'
prompt = 'Complete this C# method: static int M(int x)'
calls = []
stress.compile_candidate = lambda text, framework: calls.append(text) or {'status': 'passed'}
source = {'x': {'id': 'x', 'target': target}}
candidate = {'text': wrong, 'compile_status': 'passed', 'exact_text': False}
prior = {'x': {'id': 'x', 'prompt': prompt, 'baseline': candidate,
               'signature': candidate, 'contextual': candidate}}
totals, rows, count = hybrid.evaluate(source, prior)
assert count == 1 and calls == [target]  # Four variants share one compilation.
assert totals['baseline']['exact_text'] == 0
assert all(totals[name]['exact_text'] == 1 for name in hybrid.VARIANTS)
assert rows[0]['signature_strict']['text'] == target

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / 'input.jsonl').write_text(json.dumps(source['x']) + '\n')
    (root / 'pointer.jsonl').write_text(json.dumps(prior['x']) + '\n')
    summary = {name: {'exact_text': 0, 'compile_passes': 1, 'name_exact': 0}
               for name in hybrid.ORIGINAL_NAMES}
    report = {'input_sha256': sha256_file(root / 'input.jsonl'), 'examples': 1, 'results': summary}
    (root / 'pointer.report.json').write_text(json.dumps(report))
    sys.argv = ['test', '--input', str(root / 'input.jsonl'),
                '--pointer-results', str(root / 'pointer.jsonl'), '--output', str(root / 'out.jsonl')]
    hybrid.main()
    saved = json.loads((root / 'out.report.json').read_text())
    assert saved['results']['signature_strict']['exact_text'] == 1
    assert saved['new_compiler_calls'] == 1
print('Pointer signature hybrid checks passed.')
