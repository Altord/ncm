"""Run the two-codec ensemble on tiny CPU codecs and verify saved baselines."""
import json
import contextlib
import io
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from training.data import sha256_file
import evaluation.ensemble_codec_methods as screen


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def encode(self, text): return [2, 3, 4]
    def decode(self, ids, skip_special_tokens=True): return 'static int M() => 1;'


with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    cfg = OmegaConf.create({'tokenizer': {}, 'model': {'context_length': 16},
        'vqvae': {'in_vocab_size': 32, 'context_length': 16, 'embed_dim': 8,
                  'vqvae_vocab_size': [16, 16], 'n_layers': 2, 'compression_factor': 1,
                  'dropout': 0.0, 'vector_quantizer_config': {
                      'clss': 'multiscale_residual_vector_quantizer', 'decay': 0.95,
                      'epsilon': 1e-5, 'commitment_cost': 0.25,
                      'levels': {'use_manual_levels': True, 'manual_levels': [4, 16]},
                      'quant_resi': {'enabled': True, 'ratio': 0.5, 'share_mode': 1},
                      'aux': {'fine_drop': {'prob': 0.0}}}}})
    source = root/'input.jsonl'
    target = 'static int M() => 1;'
    source.write_text(json.dumps({'id':'one', 'target':target})+'\n')
    for name in ('a', 'b'):
        OmegaConf.save(cfg, root/f'{name}.yaml')
        torch.save({'model': VQVAE(**cfg.vqvae).state_dict()}, root/f'{name}.pt')
        (root/f'{name}.jsonl').write_text(json.dumps({'id':'one', 'target':target, 'reconstructed':target})+'\n')
        (root/f'{name}.report.json').write_text(json.dumps({'input_sha256':sha256_file(source),
                                                           'checkpoint':str(root/f'{name}.pt')}))
    screen.load_base_tokenizer = lambda _: Tokenizer()
    sys.argv = ['test', '--config-a', str(root/'a.yaml'), '--checkpoint-a', str(root/'a.pt'), '--saved-a', str(root/'a.jsonl'),
                '--config-b', str(root/'b.yaml'), '--checkpoint-b', str(root/'b.pt'), '--saved-b', str(root/'b.jsonl'),
                '--input', str(source), '--output', str(root/'out.jsonl'), '--device', 'cpu', '--batch-size', '1']
    with contextlib.redirect_stdout(io.StringIO()):
        screen.main()
    result = json.loads((root/'out.report.json').read_text())
    assert result['examples'] == 1 and result['ensemble_exact'] == 1
print('Two-codec ensemble integration checks passed.')
