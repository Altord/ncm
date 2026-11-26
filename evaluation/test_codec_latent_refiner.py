"""Check identity initialization and frozen-decoder gradient flow."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.latent_refiner import LatentRefiner
from evaluation.codec_latent_refiner_screen import benchmark_decode

cfg = OmegaConf.create({'in_vocab_size': 32, 'context_length': 16, 'embed_dim': 8,
 'vqvae_vocab_size': [16, 16], 'n_layers': 2, 'compression_factor': 1, 'dropout': 0.0,
 'vector_quantizer_config': {'clss': 'multiscale_residual_vector_quantizer', 'decay': 0.95,
  'epsilon': 1e-5, 'commitment_cost': 0.25, 'levels': {'use_manual_levels': True, 'manual_levels': [4, 16]},
  'quant_resi': {'enabled': True, 'ratio': 0.5, 'share_mode': 1}, 'aux': {'fine_drop': {'prob': 0.0}}}})
model = VQVAE(**cfg).eval().requires_grad_(False)
adapter = LatentRefiner(8, 16)
x = torch.randint(0, 32, (2, 16))
with torch.no_grad(): q = model.vq.idx_to_z_q(model.encode(x))
assert torch.equal(adapter(q), q)
assert torch.equal(model.decoder(adapter(q)), model.decoder(q))
model.decoder(adapter(q)).square().mean().backward()
assert adapter.net[-1].weight.grad is not None and torch.isfinite(adapter.net[-1].weight.grad).all()
assert all(p.grad is None for p in model.parameters())
timing = benchmark_decode(model, adapter, q, torch.device('cpu'), warmup=1, repeats=2)
assert timing['batch_size'] == 2 and timing['baseline_median_seconds'] > 0 and timing['adapter_median_seconds'] > 0
print('Latent refiner identity and gradient checks passed.')
