"""End-to-end CPU check of source-disjoint, target-removed training context."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from training.data import sha256_file
import training.prepare_project_context_pairs as prep


class Tokenizer:
    def encode(self, text):
        return [2, 3, 4]


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    sources = root / 'src'
    sources.mkdir()
    target = 'int Spend(int cost) { return cost + 1; }'
    for split in ('train', 'validation'):
        selected_target = target.replace('Spend', 'Consume') if split == 'train' else target
        (sources / f'{split}.cs').write_text('class C {\n ' + selected_target + '\n int sibling;\n}')
        row = {'id': split, 'source': f'{split}.cs', 'source_line': 2,
               'prompt': 'Complete this C# method: int ' + ('Consume' if split == 'train' else 'Spend') + '(int cost)',
               'target': selected_target}
        (root / f'{split}.jsonl').write_text(json.dumps(row) + '\n')
    val = prep.load_rows(root / 'validation.jsonl')
    augmented, counts = prep.prepare_rows(val, sources, Tokenizer(), 256)
    (root / 'augmented.jsonl').write_text(json.dumps(augmented[0]) + '\n')
    (root / 'codec.pt').write_bytes(b'fake codec')
    report = {'input_sha256': sha256_file(root / 'validation.jsonl'),
              'codec_sha256': sha256_file(root / 'codec.pt'),
              'context_token_budget': 256, 'context_status': dict(counts)}
    (root / 'coverage.json').write_text(json.dumps(report))
    OmegaConf.save(OmegaConf.create({'tokenizer': {}}), root / 'config.yaml')
    prep.load_base_tokenizer = lambda cfg: Tokenizer()
    sys.argv = ['test', '--config', str(root / 'config.yaml'), '--train', str(root / 'train.jsonl'),
                '--validation-original', str(root / 'validation.jsonl'),
                '--validation-augmented', str(root / 'augmented.jsonl'),
                '--coverage-report', str(root / 'coverage.json'), '--checkpoint', str(root / 'codec.pt'),
                '--source-root', str(sources), '--output', str(root / 'train-context.jsonl')]
    prep.main()
    result = json.loads((root / 'train-context.jsonl').read_text())
    manifest = json.loads((root / 'train-context.manifest.json').read_text())
    assert result['target'] == target.replace('Spend', 'Consume') and result['target'] not in result['prompt']
    assert 'sibling' in result['prompt'] and manifest['context_status']['context_added'] == 1
    assert manifest['validation_examples'] == 1
print('Project context training preparation checks passed.')
