"""Verify calibration screen runs with source-separated tiny inputs."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.context_refinement_head import ContextRefinementHead
from training.data import sha256_file
import evaluation.calibrate_context_refinement as screen

class Tokenizer:
    pad_token_id, eos_token_id = 0, 1
    def encode(self, text): return [2, 3, 4]
    def decode(self, ids, skip_special_tokens=True): return 'static int M() => 1;'

with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    cfg = OmegaConf.create({'tokenizer': {}, 'model': {'context_length': 16},
        'vqvae': {'in_vocab_size': 32, 'context_length': 16, 'embed_dim': 8,
        'vqvae_vocab_size': [16,16], 'n_layers': 2, 'compression_factor': 1, 'dropout': 0.0,
        'vector_quantizer_config': {'clss': 'multiscale_residual_vector_quantizer', 'decay': 0.95,
        'epsilon': 1e-5, 'commitment_cost': 0.25, 'levels': {'use_manual_levels': True, 'manual_levels': [4,16]},
        'quant_resi': {'enabled': True, 'ratio': 0.5, 'share_mode': 1}, 'aux': {'fine_drop': {'prob': 0.0}}}}})
    OmegaConf.save(cfg, root/'config.yaml')
    torch.save({'model': VQVAE(**cfg.vqvae).state_dict()}, root/'codec.pt')
    head = ContextRefinementHead(8, max_prompt_length=8)
    torch.save({'model': head.state_dict(), 'prefix_length': 8, 'codec_sha256': sha256_file(root/'codec.pt')}, root/'head.pt')
    rows = [{'id': str(i), 'source': f'{i}.cs', 'prompt': 'Complete M', 'target': 'static int M() => 1;'} for i in range(10)]
    (root/'validation.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    screen.load_base_tokenizer = lambda _: Tokenizer()
    sys.argv = ['test', '--codec-config', str(root/'config.yaml'), '--codec-checkpoint', str(root/'codec.pt'),
        '--head-checkpoint', str(root/'head.pt'), '--validation', str(root/'validation.jsonl'),
        '--output', str(root/'report.json'), '--device', 'cpu', '--batch-size', '2']
    screen.main()
    r = json.loads((root/'report.json').read_text())
    assert r['calibration'][0]['examples'] and r['evaluation'][0]['examples']
    assert r['calibration'][0]['examples'] + r['evaluation'][0]['examples'] == 10
    assert r['selected_index'] == 0  # Zero-initialized head ties baseline, conservative choice.
    assert r['evaluation'][0]['introduced'] == r['evaluation'][0]['fixed'] == 0
print('Calibration integration checks passed.')
