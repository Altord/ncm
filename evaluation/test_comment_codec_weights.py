"""Check optional comment loss weights preserve non-comment occurrences."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.csharp_structure import token_occurrence_weights

text = 'class A { // café count\n int count = 2; /* note */ }'
offsets = [(i, i+1) for i in range(len(text))]
baseline = token_occurrence_weights(text, offsets, 3.0, 1.5)
weighted = token_occurrence_weights(text, offsets, 3.0, 1.5, comment_weight=2.0)
assert weighted[text.index('café')] == 2.0
assert weighted[text.index('note')] == 2.0
assert baseline[text.index('café')] == 1.0
assert weighted[text.index('count =')] == baseline[text.index('count =')] == 1.5
assert weighted[text.index('2;')] == baseline[text.index('2;')] == 1.5
assert weighted[text.index('{')] == baseline[text.index('{')] == 3.0
print('Comment codec weighting checks passed.')
