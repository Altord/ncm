"""Check UTF-8 region labels and changed-reference counts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.compare_codec_error_regions import regions, summarize

text = 'static string M(int count) { // note\n return "é" + count; }'
labels, bad = regions(text)
assert not bad and len(labels) == len(text)
assert labels[text.index('count')] == 'identifier'
assert labels[text.index('note')] == 'comment'
assert labels[text.index('é')] == 'literal'
row = {'target': text, 'reconstructed': text.replace('count;', 'other;')}
summary = summarize([row])
assert summary['exact'] == 0 and summary['methods_with_change_by_region']['identifier'] == 1
assert summary['changed_reference_characters_by_region']['identifier'] > 0
print('Codec error-region checks passed.')
