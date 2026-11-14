"""Exercise project-context coverage end-to-end with a tiny CPU codec."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
import evaluation.measure_project_context_copyability as measurement


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def encode(self, text):
        return [2, 3, 4, 5] if 'Nearby project' in text else [2, 3]


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source_root = root / 'source'
    source_root.mkdir()
    target = 'int Spend(int cost) { return cost + 1; }'
    (source_root / 'a.cs').write_text('class C {\n ' + target + '\n int sibling;\n}', encoding='utf-8')
    row = {'id': 'x', 'source': 'a.cs', 'source_line': 2, 'target': target,
           'prompt': 'Complete this C# method: int Spend(int cost)'}
    (root / 'data.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
    cfg = OmegaConf.create({'tokenizer': {}, 'model': {'context_length': 16},
        'vqvae': {'in_vocab_size': 32, 'context_length': 16, 'embed_dim': 8,
                  'vqvae_vocab_size': [16, 16], 'n_layers': 2, 'compression_factor': 1,
                  'dropout': 0.0, 'vector_quantizer_config': {
                      'clss': 'multiscale_residual_vector_quantizer', 'decay': 0.95,
                      'epsilon': 1e-5, 'commitment_cost': 0.25,
                      'levels': {'use_manual_levels': True, 'manual_levels': [4, 16]},
                      'quant_resi': {'enabled': True, 'ratio': 0.5, 'share_mode': 1},
                      'aux': {'fine_drop': {'prob': 0.0}}}}})
    OmegaConf.save(cfg, root / 'config.yaml')
    codec = VQVAE(**cfg.vqvae)
    torch.save({'model': codec.state_dict()}, root / 'codec.pt')
    measurement.load_base_tokenizer = lambda cfg: Tokenizer()
    sys.argv = ['test', '--config', str(root / 'config.yaml'), '--checkpoint', str(root / 'codec.pt'),
                '--input', str(root / 'data.jsonl'), '--source-root', str(source_root),
                '--output', str(root / 'out'), '--device', 'cpu', '--batch-size', '1']
    measurement.main()
    report = json.loads((root / 'out/report.json').read_text())
    assert report['examples'] == 1 and report['context_status']['context_added'] == 1
    assert report['valid_tokens'] == 3
    assert report['context_copyable'] >= report['prompt_copyable']
    augmented = json.loads((root / 'out/augmented.jsonl').read_text())
    assert target not in augmented['prompt'] and 'sibling' in augmented['prompt']
print('Project context copyability integration checks passed.')
