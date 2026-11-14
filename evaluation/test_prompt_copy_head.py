"""CPU checks for pointer mass, masking, gradients and unchanged decoder output."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from models.prompt_copy_head import PromptCopyHead
from models.vae import Decoder
from training.train_prompt_copy_screen import frozen_codec
from models.vqvae import VQVAE
from omegaconf import OmegaConf

torch.manual_seed(7)
head = PromptCopyHead(4)
features = torch.randn(1, 2, 4)
embeddings = torch.randn(1, 3, 4)
ids = torch.tensor([[2, 2, 3]])
mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
base = torch.randn(1, 2, 5)
base[0, 0] = torch.tensor([4., 0., 0., 0., 0.])  # Wrong, but prompt includes target ID 2.
target = torch.tensor([[2, 4]])
valid = torch.tensor([[1, 1]], dtype=torch.bool)
attention, gate = head.distribution(features, embeddings, mask)
assert torch.allclose(attention.sum(-1), torch.ones(1, 2))
assert attention[:, :, 2].sum() == 0
loss = head.nll(features, base, embeddings, ids, mask, target, valid)
loss.backward()
assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
predicted, _ = head.predict(features, base, embeddings, ids, mask)
p = base.softmax(-1) * (1 - gate.sigmoid().unsqueeze(-1))
p.scatter_add_(-1, ids[:, None, :].expand(-1, 2, -1), attention * gate.sigmoid().unsqueeze(-1))
assert torch.equal(predicted, p.argmax(-1))
assert torch.allclose(p.sum(-1), torch.ones(1, 2), atol=1e-6)
expected = -p.gather(-1, target.unsqueeze(-1)).squeeze(-1).log().mean()
assert torch.allclose(loss.detach(), expected.detach(), atol=1e-6)
boosted = head.nll(features, base, embeddings, ids, mask, target, valid, copyable_error_boost=20)
per_token = -p.gather(-1, target.unsqueeze(-1)).squeeze(-1).log()
expected_boosted = (20 * per_token[0, 0] + per_token[0, 1]) / 21
assert torch.allclose(boosted.detach(), expected_boosted.detach(), atol=1e-6)
combined_loss, components = head.nll(features, base, embeddings, ids, mask, target, valid,
    copyable_error_boost=20, attention_aux_weight=0.2, gate_aux_weight=0.1,
    gate_positive_weight=20, return_components=True)
expected_attention = -attention[0, 0, :2].sum().log()
gate_labels = torch.tensor([[1., 0.]])
gate_bce = torch.nn.functional.binary_cross_entropy_with_logits(gate, gate_labels, reduction='none')
expected_gate = (20 * gate_bce[0, 0] + gate_bce[0, 1]) / 21
assert torch.allclose(components['copy_attention_nll'], expected_attention.detach(), atol=1e-6)
assert torch.allclose(components['copy_gate_bce'], expected_gate.detach(), atol=1e-6)
assert components['copyable_errors'].item() == 1
assert torch.allclose(combined_loss.detach(),
                      (expected_boosted + .2 * expected_attention + .1 * expected_gate).detach(), atol=1e-6)
none_loss, none_components = head.nll(features, base, embeddings, ids, mask,
    torch.full_like(target, 4), valid, attention_aux_weight=.2, gate_aux_weight=.1,
    return_components=True)
assert none_components['copyable_errors'].item() == 0 and torch.isfinite(none_loss)
try:
    head.nll(features, base, embeddings, ids, mask, target, valid, copyable_error_boost=0.5)
except ValueError:
    pass
else:
    raise AssertionError('Invalid boost accepted')
try:
    head.distribution(features, embeddings, torch.zeros_like(mask))
except ValueError:
    pass
else:
    raise AssertionError('Empty prompt mask accepted')
decoder = Decoder(in_vocab_size=5, embed_dim=4, n_layers=1, compression_factor=1,
                  dropout=0, dec_layers=2, context_length=4)
decoder.eval()
z = torch.randn(1, 4, 4)
assert torch.equal(decoder(z), decoder.output_layer(decoder.forward_features(z)))
class FakeVQ:
    def idx_to_z_q(self, codes):
        return torch.nn.functional.one_hot(codes, num_classes=4).float()
class FakeCodec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder(in_vocab_size=5, embed_dim=4, n_layers=1, compression_factor=1,
                               dropout=0, dec_layers=1, context_length=2)
        self.vq = FakeVQ()
    def encode(self, x):
        return x.remainder(4)
codec = FakeCodec().eval().requires_grad_(False)
x = torch.tensor([[1, 2]])
f, logits, emb = frozen_codec(codec, x, ids)
copy_head = PromptCopyHead(4)
opt = torch.optim.Adam(copy_head.parameters(), lr=0.01)
for _ in range(2):
    opt.zero_grad()
    loss = copy_head.nll(f, logits, emb, ids, mask, x, valid)
    loss.backward()
    opt.step()
assert all(p.grad is None for p in codec.parameters())
assert torch.isfinite(loss)
cfg = OmegaConf.create({'in_vocab_size': 32, 'context_length': 16, 'embed_dim': 8,
    'vqvae_vocab_size': [16, 16], 'n_layers': 2, 'compression_factor': 1,
    'dropout': 0.0, 'vector_quantizer_config': {
        'clss': 'multiscale_residual_vector_quantizer', 'decay': 0.95,
        'epsilon': 1e-5, 'commitment_cost': 0.25,
        'levels': {'use_manual_levels': True, 'manual_levels': [4, 16]},
        'quant_resi': {'enabled': True, 'ratio': 0.5, 'share_mode': 1},
        'aux': {'fine_drop': {'prob': 0.0}}}})
real = VQVAE(**cfg).eval().requires_grad_(False)
tokens = torch.randint(0, 32, (2, 16))
prompts = torch.randint(0, 32, (2, 4))
f, logits, emb = frozen_codec(real, tokens, prompts)
assert torch.allclose(logits, real.decode(real.encode(tokens)))
real_head = PromptCopyHead(8)
loss = real_head.nll(f, logits, emb, prompts, torch.ones_like(prompts, dtype=torch.bool),
                     tokens, torch.ones_like(tokens, dtype=torch.bool))
loss.backward()
assert torch.isfinite(loss) and all(p.grad is None for p in real.parameters())
print('Prompt copy head CPU checks passed.')
