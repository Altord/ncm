"""Verify the weight changes VQ loss while retaining discrete forward outputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE

cfg = OmegaConf.create({'in_vocab_size': 32, 'context_length': 16, 'embed_dim': 8,
    'vqvae_vocab_size': [16, 16], 'n_layers': 2, 'compression_factor': 1, 'dropout': 0.0,
    'vector_quantizer_config': {'clss': 'multiscale_residual_vector_quantizer', 'decay': 0.95,
      'epsilon': 1e-5, 'commitment_cost': 0.25, 'levels': {'use_manual_levels': True, 'manual_levels': [4, 16]},
      'quant_resi': {'enabled': True, 'ratio': 0.5, 'share_mode': 1}, 'aux': {'fine_drop': {'prob': 0.0}}}})
torch.manual_seed(7)
base = VQVAE(**cfg).eval()
cfg.vector_quantizer_config.final_level_loss_weight = 8.0
candidate = VQVAE(**cfg).eval()
candidate.load_state_dict(base.state_dict())
x = torch.randint(0, 32, (2, 16))
normal, loss_a, *_ = base(x)
weighted, loss_b, *_ = candidate(x)
assert torch.equal(normal, weighted)
assert not torch.isclose(loss_a, loss_b)
(weighted.mean() + loss_b).backward()
assert torch.isfinite(candidate.encoder.wte.weight.grad).all()
print('Final-level VQ integration checks passed.')
