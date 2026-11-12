"""Small CPU checks for compact VQ decoding and causal teacher forcing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE

torch.manual_seed(7)
cfg = OmegaConf.create({
    'in_vocab_size': 32, 'context_length': 16, 'embed_dim': 8,
    'vqvae_vocab_size': [16], 'n_layers': 2, 'compression_factor': 4,
    'enc_layers': 1, 'dec_layers': 2, 'dropout': 0.0, 'dec_positional': True,
    'autoregressive_decoder': True,
    'vector_quantizer_config': {
        'clss': 'multiscale_residual_vector_quantizer', 'decay': 0.95,
        'epsilon': 1e-5, 'commitment_cost': 0.25,
        'quant_resi': {'enabled': True, 'ratio': 0.5, 'share_mode': 1},
        'aux': {'fine_drop': {'prob': 0.0, 'min_keep': 1}},
        'levels': {'use_manual_levels': True, 'manual_levels': [4]},
    },
})
model = VQVAE(**cfg)
x = torch.randint(0, 32, (2, 16))
model.train()
logits, vq_loss, *_ = model(x)
(torch.nn.functional.cross_entropy(logits.reshape(-1, 32), x.reshape(-1)) + vq_loss).backward()
assert model.decoder.token_embedding.weight.grad is not None
model.eval()
with torch.no_grad():
    codes = model.encode(x)
    assert codes.shape == (2, 4)
    z = model.vq.idx_to_z_q(codes)
    a = model.decoder(z, targets=x)
    changed = x.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % 32
    b = model.decoder(z, targets=changed)
    assert torch.allclose(a[:, :9], b[:, :9]), 'future tokens leaked into teacher forcing'
    greedy = model.decode(codes, max_tokens=9)
    assert greedy.shape == (2, 9, 32)
    previous = torch.cat((torch.zeros((2, 1), dtype=torch.long), greedy.argmax(-1)[:, :-1]), 1)
    teacher = model.decoder._logits(model.decoder._features(z), previous)
    assert torch.allclose(greedy, teacher, atol=1e-5), 'greedy decode does not use its own past'
print('Compact autoregressive CPU checks passed.')
