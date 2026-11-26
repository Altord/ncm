"""Ensure identifier canonicalization is reversible for common C# syntax."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.codec_identifier_table_screen import canonicalize, restore

for text in (
    'static int Sum(int value, int count) => value + count;',
    'static string M(int café) => $"café {café}";',
    'static int M(int @class) { /* @class */ return @class; }',
    'static string M(string x) => "__ncmid_0__" + x;',
):
    result = canonicalize(text)
    assert result is not None, text
    transformed, table = result
    assert restore(transformed, table) == text, (text, transformed, table)
    assert table
assert canonicalize('static int M( {') is None
print('Identifier-table reversibility checks passed.')
