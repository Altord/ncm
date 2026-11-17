"""AST candidate edits stay inside identifier error nodes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.ast_codec_candidates import ast_candidates

prompt = 'Complete this C# method: static int M(int value, int count)'
text = 'static int M(int value, int count) => value + nope nope;'
candidates = ast_candidates(prompt, text)
assert candidates and all('nope' in c for c in candidates)
assert any('value + count nope' in c for c in candidates)
valid = 'static int M(int value, int count) => value + count;'
assert ast_candidates(prompt, valid) == []
with_comment = 'static int M(int value, int count) { // keep nope\n return value + nope nope; }'
assert all('// keep nope' in c for c in ast_candidates(prompt, with_comment))
assert ast_candidates('plain prompt', text) == []
print('AST codec candidate checks passed.')
