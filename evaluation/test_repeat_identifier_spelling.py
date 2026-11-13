"""Check repeated whole identifier spellings, independent of tokenizer pieces."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.csharp_structure import token_occurrence_weights

text = 'class C { // cost is not code\n int F(int cost, int other) => cost + other + cost; }'
offsets = [(i, i + 1) for i in range(len(text))]
old = token_occurrence_weights(text, offsets, 3.0, 1.5)
new = token_occurrence_weights(text, offsets, 3.0, 1.5, repeated_identifier_weight=2.25)
assert new[text.index('cost')] == 1.0  # Comment is not an identifier node.
for start in (text.index('cost', text.index('int F')), text.rindex('cost')):
    assert new[start:start + 4] == [2.25] * 4
assert new[text.index('int F') + 4] == 1.5  # The method F is not repeated.
assert old[text.rindex('cost')] == 1.5
print('Repeated identifier spelling checks passed.')
