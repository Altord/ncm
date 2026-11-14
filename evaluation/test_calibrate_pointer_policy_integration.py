"""Exercise pointer-policy calibration with a tiny CPU codec."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.prompt_copy_head import PromptCopyHead
from training.data import sha256_file
import evaluation.calibrate_pointer_policy as experiment


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
    head = PromptCopyHead(8, confidence_aware_gate=True, context_layers=1, context_heads=2, context_max_length=8)
    torch.save({'model': head.state_dict(), 'confidence_aware_gate': True,
                'prompt_context_layers': 1, 'prompt_context_heads': 2, 'context_max_length': 8,
                'codec_sha256': sha256_file(root / 'codec.pt'),
                'train_sha256': sha256_file(root / 'train.jsonl')}, root / 'head.pt')
    experiment.load_base_tokenizer = lambda cfg: Tokenizer()
    sys.argv = ['test', '--codec-config', str(root / 'config.yaml'),
                '--codec-checkpoint', str(root / 'codec.pt'), '--head', str(root / 'head.pt'),
                '--train', str(root / 'train.jsonl'), '--validation', str(root / 'val.jsonl'),
                '--output', str(root / 'out'), '--device', 'cpu', '--sample-size', '1',
                '--batch-size', '1', '--prefix-length', '8']
    experiment.main()
    report = json.loads((root / 'out/report.json').read_text())
    assert report['calibration_examples'] == 1 and report['validation_examples'] == 1
    assert report['metrics']['valid_tokens'] == 4
    assert (root / 'out/results.jsonl').is_file()
print('Pointer-policy integration checks passed.')
