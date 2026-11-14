"""Check signature copying and compiler-only candidate selection."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.data import sha256_file
import evaluation.evaluate_pointer_stress as stress
import evaluation.evaluate_repair_signature_hybrid as hybrid

target = 'static int M(int x) => x;'
baseline = 'static int Other(int x) => x;'
repair = 'static int Miss(int x) => x;'
prompt = 'Complete this C# method: static int M(int x)'
calls = []
stress.compile_candidate = lambda text, framework: calls.append(text) or {'status': 'passed'}
source = {'x': {'id': 'x', 'target': target}}
previous = {'x': {'id': 'x', 'baseline': baseline, 'selected': repair,
                  'compile_status': 'passed', 'selected_text_exact': False}}
saved = {'text': baseline, 'compile_status': 'passed', 'exact_text': False}
combined = {'x': {'id': 'x', 'baseline': saved, 'signature_strict': saved, 'signature_fallback': saved}}
pointer = {'x': {'id': 'x', 'prompt': prompt, 'baseline': {'text': baseline, 'compile_status': 'passed'}}}
totals, rows, count = hybrid.evaluate(source, previous, combined, pointer)
assert count == 1 and calls == [target]
assert totals['repair']['exact_text'] == 0 and totals['repair_strict']['exact_text'] == 1
assert rows[0]['select_strict']['selection'] == 'repair_preferred'
assert rows[0]['select_strict']['text'] == target
assert hybrid.choose_compiling({'compile_status': 'failed'}, {'compile_status': 'passed'})['selection'] == 'pointer_fallback'

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    for name, row in (('input', source['x']), ('repair', previous['x']),
                      ('hybrid', combined['x']), ('pointer', pointer['x'])):
        (root / (name + '.jsonl')).write_text(json.dumps(row) + '\n')
    input_hash = sha256_file(root / 'input.jsonl')
    (root / 'repair.report.json').write_text(json.dumps({'examples': 1, 'selected_compiles': 1, 'selected_exact_text': 0}))
    (root / 'pointer.report.json').write_text(json.dumps({'examples': 1, 'input_sha256': input_hash}))
    summary = {name: {'exact_text': 0, 'compile_passes': 1, 'name_exact': 0}
               for name in ('baseline', 'signature_strict', 'signature_fallback')}
    (root / 'hybrid.report.json').write_text(json.dumps({'examples': 1, 'input_sha256': input_hash,
        'pointer_results_sha256': sha256_file(root / 'pointer.jsonl'), 'results': summary}))
    sys.argv = ['test', '--input', str(root / 'input.jsonl'), '--repair', str(root / 'repair.jsonl'),
                '--hybrid', str(root / 'hybrid.jsonl'), '--pointer', str(root / 'pointer.jsonl'),
                '--output', str(root / 'out.jsonl')]
    hybrid.main()
    report = json.loads((root / 'out.report.json').read_text())
    assert report['results']['select_strict']['exact_text'] == 1
    assert report['new_compiler_calls'] == 1
print('Repair signature hybrid checks passed.')
