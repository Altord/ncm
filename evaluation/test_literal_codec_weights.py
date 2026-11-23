"""Check literal weighting leaves identifiers and interpolation expressions alone."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.csharp_structure import token_occurrence_weights

text = 'class A { // note\n string M(int x) => $"café {x}"; int n = 42; }'
offsets = [(i, i + 1) for i in range(len(text))]
baseline = token_occurrence_weights(text, offsets, 3.0, 1.5)
weighted = token_occurrence_weights(text, offsets, 3.0, 1.5, literal_weight=2.0)
for piece in ('café', '42'):
    i = text.index(piece)
    assert baseline[i] == 1.5 and weighted[i] == 2.0, piece
for i in (text.index('x)'), text.index('{x}') + 1):
    assert baseline[i] == weighted[i] == 1.5
assert baseline[text.index('note')] == weighted[text.index('note')] == 1.0
assert baseline[text.index('class')] == weighted[text.index('class')] == 3.0
print('Literal codec weighting checks passed.')
