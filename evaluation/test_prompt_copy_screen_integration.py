"""Run the prompt-copy trainer end-to-end with a tiny CPU codec and fake tokenizer."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
import training.train_prompt_copy_screen as screen


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def encode(self, text):
        return [2, 3, 4]
    def decode(self, ids, skip_special_tokens=True):
        return 'static int M(int x) => x;'


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
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
    for split in ('train', 'val'):
        row = {'id': split, 'source': split + '.cs', 'prompt': 'Complete M',
               'target': 'static int M(int x) => x;'}
        (root / f'{split}.jsonl').write_text(json.dumps(row) + '\n')
    screen.load_base_tokenizer = lambda cfg: Tokenizer()
    sys.argv = ['test', '--codec-config', str(root / 'config.yaml'),
                '--codec-checkpoint', str(root / 'codec.pt'), '--train', str(root / 'train.jsonl'),
                '--validation', str(root / 'val.jsonl'), '--output', str(root / 'out'),
                '--device', 'cpu', '--steps', '2', '--batch-size', '1', '--prefix-length', '8',
                '--copyable-error-boost', '20']
    screen.main()
    result = json.loads((root / 'out/report.json').read_text())
    assert result['train_steps'] == 2 and result['validation_examples'] == 1
    assert result['copyable_error_boost'] == 20
    assert result['before_exact_text'] == result['after_exact_text'] == 1
    assert (root / 'out/copy_head.pt').is_file()
print('Prompt-copy trainer integration checks passed.')
