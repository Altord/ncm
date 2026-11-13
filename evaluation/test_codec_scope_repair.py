"""Check that repair candidates use decoded names, not original targets."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.codec_scope_repair import proposals

text = 'static bool TrySpend(int balance, int cost) { return Mon < cost; }'
detail = "error CS0103: The name 'Mon' does not exist in the current context"
items = proposals(text, detail)
assert any('return cost < cost;' in candidate for candidate, _ in items)
assert any('return balance < cost;' in candidate for candidate, _ in items)
assert not proposals('static int M() => 0;', '')
text = 'static bool M(int n) { return n class }'
assert any('return n ;' in candidate for candidate, _ in proposals(text, ''))
print('Scope-inspired proposal checks passed.')
