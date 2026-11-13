"""Verify AST method-name labels, including UTF-8 offsets and legacy behavior."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.csharp_structure import token_occurrence_weights

text = 'class C { // café\n public int Cooldown(int value) => value; }'
offsets = [(i, i + 1) for i in range(len(text))]
old = token_occurrence_weights(text, offsets, 3.0, 1.5)
new = token_occurrence_weights(text, offsets, 3.0, 1.5, method_name_weight=4.0)
name = text.index('Cooldown')
variable = text.index('value')
assert all(v == 4.0 for v in new[name:name + len('Cooldown')])
assert all(v == 1.5 for v in old[name:name + len('Cooldown')])
assert all(v == 1.5 for v in new[variable:variable + len('value')])
assert all(a == b for i, (a, b) in enumerate(zip(old, new))
           if not name <= i < name + len('Cooldown'))
print('AST method-name weighting checks passed.')
