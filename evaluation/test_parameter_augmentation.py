"""Check conservative parameter augmentation and syntax guards."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.augment_parameter_names import augment

text = 'class C { int Add(int score, int cost) { return score + cost; } }'
new = augment(text)
assert new == 'class C { int Add(int ncmParameter0, int ncmParameter1) { return ncmParameter0 + ncmParameter1; } }'
assert augment('class C { int Bad( {') is None
assert augment('class C { int M(int score) { return other.score + score; } }') is None
assert augment('class C { int M(int score) { System.Func<int> f = () => score; return f(); } }') is None
print('Parameter-name augmentation checks passed.')
