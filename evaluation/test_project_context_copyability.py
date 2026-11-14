"""Check source removal, path safety, prompt budgets and optimistic coverage."""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.measure_project_context_copyability import augment_record, token_coverage


class Tokenizer:
    def encode(self, text):
        return [ord(c) for c in text]


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    target = 'int Spend(int cost) { return cost + 1; }'
    source = 'class C {\n int other;\n ' + target + '\n int Sibling() { return other; }\n}'
    (root / 'a.cs').write_text(source, encoding='utf-8')
    row = {'id': 'x', 'source': 'a.cs', 'source_line': 3, 'target': target,
           'prompt': 'Complete this C# method: int Spend(int cost)'}
    augmented, reason = augment_record(row, root, Tokenizer(), max_tokens=512, window=100)
    assert reason == 'context_added' and augmented['prompt'].startswith(row['prompt'])
    assert target not in augmented['prompt'] and 'Sibling' in augmented['prompt']
    assert len(Tokenizer().encode(augmented['prompt'])) <= 512
    escaped = dict(row, source='../outside.cs')
    assert augment_record(escaped, root, Tokenizer(), max_tokens=512)[1] == 'source_outside_root'
    (root / 'a.cs').write_text(source + '\n' + target, encoding='utf-8')
    from evaluation.measure_project_context_copyability import parse_source
    parse_source.cache_clear()
    assert augment_record(row, root, Tokenizer(), max_tokens=512)[1] == 'verbatim_target_elsewhere'

stats = token_coverage([1, 2, 3, 4], [1, 1, 1, 0], [0, 0, 3, 0], [1], [1, 2])
assert stats['valid_tokens'] == 3 and stats['baseline_errors'] == 2
assert stats['prompt_copyable'] == 1 and stats['context_copyable'] == 2
assert stats['newly_copyable'] == 1 and stats['no_longer_copyable'] == 0
print('Project context copyability checks passed.')
