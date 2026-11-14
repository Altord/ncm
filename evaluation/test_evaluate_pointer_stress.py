"""Test synthetic-signature extraction and stress evaluation without GPU or .NET."""
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
import evaluation.evaluate_pointer_stress as experiment


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def encode(self, text):
        return [2, 3, 4]
    def decode(self, ids, skip_special_tokens=True):
        return 'static int M(int x) => x;'


target = 'static int M(int x) => x;'
assert experiment.make_signature_prompt(target) == 'Complete this C# method: static int M(int x)'
assert '=> x;' not in experiment.make_signature_prompt(target)
try:
    experiment.make_signature_prompt('bad !!!')
except ValueError:
    pass
else:
    raise AssertionError('Malformed target accepted')
calls = []
experiment.compile_candidate = lambda text, framework: calls.append(text) or {'status': 'failed'}
cache = {'same': 'passed'}
assert experiment.compile_or_reuse('same', cache, 'net8.0') == ('passed', False)
assert experiment.compile_or_reuse('changed', cache, 'net8.0') == ('failed', True)
assert experiment.compile_or_reuse('changed', cache, 'net8.0') == ('failed', False)
assert calls == ['changed']
assert experiment.compile_or_reuse('', cache, 'net8.0') == ('empty', False)

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
    source = [{'id': f'row-{i}', 'source': 'test.cs', 'target': target} for i in range(2)]
    (root / 'input.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in source))
    saved = [dict(r, reconstructed=target, original_compile={'status': 'passed'},
                  reconstruction_compile={'status': 'passed'}) for r in source]
    (root / 'baseline.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in saved))
    report = {'input_sha256': sha256_file(root / 'input.jsonl'),
              'checkpoint': str((root / 'codec.pt').resolve()), 'compile_requested': True, 'examples': 2}
    (root / 'baseline.report.json').write_text(json.dumps(report))
    for name, prefix, layers in [('signature', 64, 0), ('contextual', 256, 1)]:
        head = PromptCopyHead(8, context_layers=layers, context_heads=2, context_max_length=prefix)
        torch.save({'model': head.state_dict(), 'codec_sha256': sha256_file(root / 'codec.pt'),
                    'prompt_context_layers': layers, 'prompt_context_heads': 2,
                    'context_max_length': prefix}, root / (name + '.pt'))
    experiment.load_base_tokenizer = lambda cfg: Tokenizer()
    experiment.compile_candidate = lambda text, framework: {'status': 'passed'}
    sys.argv = ['test', '--config', str(root / 'config.yaml'), '--codec', str(root / 'codec.pt'),
                '--input', str(root / 'input.jsonl'), '--baseline', str(root / 'baseline.jsonl'),
                '--signature-head', str(root / 'signature.pt'), '--context-head', str(root / 'contextual.pt'),
                '--output', str(root / 'output.jsonl'), '--device', 'cpu', '--batch-size', '2']
    experiment.main()
    result = json.loads((root / 'output.report.json').read_text())
    assert result['examples'] == 2 and result['new_compiler_calls'] == 0
    assert all(result['results'][name]['compile_passes'] == 2 for name in ('baseline', 'signature', 'contextual'))
print('Pointer stress evaluation checks passed.')
