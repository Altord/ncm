"""Exercise discrete and continuous paths with a tiny frozen CPU codec."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from training.data import sha256_file
import evaluation.codec_continuous_latent_screen as screen

class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def encode(self, text): return [2, 3, 4]
    def decode(self, ids, skip_special_tokens=True): return 'int M() => 1;'

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    cfg = OmegaConf.create({'tokenizer': {}, 'model': {'context_length': 16},
        'vqvae': {'in_vocab_size': 32, 'context_length': 16, 'embed_dim': 8,
          'vqvae_vocab_size': [16, 16], 'n_layers': 2, 'compression_factor': 1, 'dropout': 0.0,
          'vector_quantizer_config': {'clss': 'multiscale_residual_vector_quantizer', 'decay': 0.95,
            'epsilon': 1e-5, 'commitment_cost': 0.25, 'levels': {'use_manual_levels': True, 'manual_levels': [4, 16]},
            'quant_resi': {'enabled': True, 'ratio': 0.5, 'share_mode': 1}, 'aux': {'fine_drop': {'prob': 0.0}}}}})
    source = root/'input.jsonl'; checkpoint = root/'codec.pt'
    target = 'int M() => 1;'
    source.write_text(json.dumps({'id': 'one', 'target': target})+'\n')
    OmegaConf.save(cfg, root/'config.yaml')
    torch.save({'model': VQVAE(**cfg.vqvae).state_dict()}, checkpoint)
    (root/'baseline.jsonl').write_text(json.dumps({'id': 'one', 'target': target, 'reconstructed': target})+'\n')
    (root/'baseline.report.json').write_text(json.dumps({'input_sha256': sha256_file(source), 'checkpoint': str(checkpoint)}))
    screen.load_base_tokenizer = lambda _: Tokenizer()
    sys.argv = ['test', '--config', str(root/'config.yaml'), '--checkpoint', str(checkpoint), '--input', str(source),
                '--baseline', str(root/'baseline.jsonl'), '--output', str(root/'out.jsonl'), '--device', 'cpu']
    with contextlib.redirect_stdout(io.StringIO()): screen.main()
    report = json.loads((root/'out.report.json').read_text())
    assert report['examples'] == 1 and report['valid_tokens'] == 4
    assert report['discrete_exact'] == 1 and report['continuous_exact'] == 1
print('Continuous-latent integration checks passed.')
