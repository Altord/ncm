"""Exercise parameter proposals without target body or a compiler."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.signature_parameter_search import generate_candidates

prompt = 'Complete this C# method: static int MultiplyScore(int score, int multiplier)'
text = 'static int MultiplyScore(int score, int multiplier) => Math.Max(0, score) * Math.Max(0, multi...);'
candidates, complete = generate_candidates(prompt, text, 12)
assert complete and len(candidates) == 2
assert 'static int MultiplyScore(int score, int multiplier) => Math.Max(0, score) * Math.Max(0, multiplier);' in candidates
prompt = 'Complete this C# method: static bool Check(float value, float epsilon)'
text = 'static bool Check(float value, float epsilon) => value > eWeek ? 1 : value < - Nancy ? -1 : 0;'
candidates, complete = generate_candidates(prompt, text, 12)
assert complete and len(candidates) == 4
assert any('value > epsilon ? 1 : value < - epsilon ?' in c for c in candidates)
assert generate_candidates('plain language', text, 12) == ([], False)
print('Signature parameter search checks passed.')
