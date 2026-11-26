"""Verify index proposals preserve layout and scoring ignores padding."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from evaluation.codec_index_search import is_improvement, propose, score

cfg = OmegaConf.create({'in_vocab_size': 32, 'context_length': 16, 'embed_dim': 8,
 'vqvae_vocab_size': [16, 16], 'n_layers': 2, 'compression_factor': 1, 'dropout': 0.0,
 'vector_quantizer_config': {'clss': 'multiscale_residual_vector_quantizer', 'decay': 0.95,
  'epsilon': 1e-5, 'commitment_cost': 0.25, 'levels': {'use_manual_levels': True, 'manual_levels': [4, 16]},
  'quant_resi': {'enabled': True, 'ratio': 0.5, 'share_mode': 1}, 'aux': {'fine_drop': {'prob': 0.0}}}})
torch.manual_seed(7)
model = VQVAE(**cfg).eval().requires_grad_(False)
assert is_improvement(1, 10.0, 2, 0.1)
assert is_improvement(2, 0.1, 2, 0.2)
assert not is_improvement(3, 0.0, 2, 1.0)
assert not is_improvement(2, 0.2, 2, 0.2)
x = torch.randint(0, 32, (1, 16)); valid = torch.ones_like(x, dtype=torch.bool); valid[:, -2:] = False
with torch.no_grad(): codes = model.encode(x)
trials = propose(model, codes, x, valid, 4)
assert len(trials) == 4
for trial in trials:
    assert trial.shape == codes.shape
    assert int((trial != codes).sum()) == 1
    assert torch.equal(trial[:, :4], codes[:, :4])
    with torch.no_grad():
        errors, ce = score(model.decode(trial), x, valid)
    assert errors.shape == ce.shape == (1,) and torch.isfinite(ce).all()
print('Encoder-side VQ index-search checks passed.')
