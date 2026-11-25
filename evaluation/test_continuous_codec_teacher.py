"""Check continuous teacher detachment and discrete student backpropagation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from training.reconstruction_loss import topk_teacher_distillation_loss

cfg = OmegaConf.create({'in_vocab_size': 32, 'context_length': 16, 'embed_dim': 8,
 'vqvae_vocab_size': [16, 16], 'n_layers': 2, 'compression_factor': 1, 'dropout': 0.0,
 'vector_quantizer_config': {'clss': 'multiscale_residual_vector_quantizer', 'decay': 0.95,
  'epsilon': 1e-5, 'commitment_cost': 0.25, 'levels': {'use_manual_levels': True, 'manual_levels': [4, 16]},
  'quant_resi': {'enabled': True, 'ratio': 0.5, 'share_mode': 1}, 'aux': {'fine_drop': {'prob': 0.0}}}})
torch.manual_seed(9)
model = VQVAE(**cfg).eval()
x = torch.randint(0, 32, (2, 16))
baseline = model(x)
result = model(x, continuous_teacher_topk=4)
assert len(baseline) == 5 and len(result) == 6
assert torch.equal(baseline[0], result[0])
indices, probabilities = result[5]
assert indices.shape == probabilities.shape == (2, 16, 4)
assert not probabilities.requires_grad
assert torch.allclose(probabilities.sum(-1), torch.ones(2, 16))
# Force eligibility with the teacher's best target; only mismatched student tokens contribute.
targets = indices[..., 0]
valid = torch.ones_like(targets, dtype=torch.bool)
loss, count = topk_teacher_distillation_loss(result[0], targets, valid, [result[5]], student_errors_only=True)
assert int(count) == int((result[0].argmax(-1) != targets).sum())
loss.backward()
assert model.encoder.wte.weight.grad is not None
assert torch.isfinite(model.encoder.wte.weight.grad).all()
print('Continuous codec teacher checks passed.')
