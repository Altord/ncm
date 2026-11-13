"""Check that repair candidates use decoded names, not original targets."""
import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.codec_scope_repair import proposals
import evaluation.codec_scope_repair as repair

text = 'static bool TrySpend(int balance, int cost) { return Mon < cost; }'
detail = "error CS0103: The name 'Mon' does not exist in the current context"
items = proposals(text, detail)
assert any('return cost < cost;' in candidate for candidate, _ in items)
assert any('return balance < cost;' in candidate for candidate, _ in items)
assert not proposals('static int M() => 0;', '')
text = 'static bool M(int n) { return n class }'
assert any('return n ;' in candidate for candidate, _ in proposals(text, ''))
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    text = 'static bool M(int balance, int cost) { return Mon < cost; }'
    detail = "error CS0103: The name 'Mon' does not exist in the current context"
    original = text.replace('Mon', 'cost')
    baseline = {'id': 'x', 'target': original, 'reconstructed': text,
                'reconstruction_compile': {'status': 'failed', 'detail': detail}}
    previous = {'id': 'x', 'baseline': text, 'selected': text,
                'selection': 'baseline', 'compile_status': 'failed'}
    for name, row in [('baseline.jsonl', baseline), ('previous.jsonl', previous)]:
        (root / name).write_text(json.dumps(row) + '\n')
    repair.compile_candidate = lambda text, framework: {'status': 'passed', 'detail': ''}
    sys.argv = ['test', '--baseline', str(root / 'baseline.jsonl'),
                '--previous', str(root / 'previous.jsonl'), '--output', str(root / 'out.jsonl'),
                '--require-unique']
    repair.main()
    result = json.loads((root / 'out.jsonl').read_text())
    assert result['compile_status'] == 'failed'
    assert result['compiling_candidates_found'] == 2
    assert result['search_exhausted'] is True
print('Scope-inspired proposal checks passed.')
