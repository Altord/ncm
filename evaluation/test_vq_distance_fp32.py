"""Verify FP32 VQ distances remain FP32 under autocast and default remains compatible."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE

cfg = OmegaConf.create({'in_vocab_size': 32, 'context_length': 16, 'embed_dim': 8,
    'vqvae_vocab_size': [16, 16], 'n_layers': 2, 'compression_factor': 1,
    'dropout': 0.0, 'vector_quantizer_config': {
        'clss': 'multiscale_residual_vector_quantizer', 'decay': 0.95,
        'epsilon': 1e-5, 'commitment_cost': 0.25,
        'levels': {'use_manual_levels': True, 'manual_levels': [4, 16]},
        'quant_resi': {'enabled': True, 'ratio': 0.5, 'share_mode': 1},
        'aux': {'fine_drop': {'prob': 0.0}}, 'distance_fp32': True}})
torch.manual_seed(19)
model = VQVAE(**cfg)
x = torch.randn(4, 8)
emb = torch.randn(16, 8)
expected = (x.square().sum(1, keepdim=True) + emb.square().sum(1)
            - 2 * (x @ emb.t()))
with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
    distances = model.vq._distances(x, emb)
assert distances.dtype == torch.float32
assert torch.allclose(distances, expected, atol=1e-6)
model.vq.distance_fp32 = False
with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
    default_distances = model.vq._distances(x, emb)
assert not torch.equal(default_distances, distances), 'test did not exercise autocast'
model.vq.distance_fp32 = True
model.train()
tokens = torch.randint(0, 32, (2, 16))
with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
    logits, vq_loss, *_ = model(tokens)
    loss = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), tokens) + vq_loss
loss.backward()
assert torch.isfinite(loss) and model.encoder.wte.weight.grad is not None
print('FP32 VQ distance and gradient checks passed.')
