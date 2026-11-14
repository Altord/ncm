"""Small CPU checks for refinement identity, masking, and loss."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from models.context_refinement_head import ContextRefinementHead, refinement_loss

torch.manual_seed(7)
head = ContextRefinementHead(8, context_heads=2, max_prompt_length=4, bottleneck=4)
features = torch.randn(2, 3, 8)
logits = torch.randn(2, 3, 11)
prompt = torch.randn(2, 4, 8)
mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
weight = torch.randn(11, 8)
assert torch.equal(head(features, logits, prompt, mask, weight), logits)
with torch.no_grad():
    head.up.weight.normal_()
modified = prompt.clone()
modified[~mask] = 1000
head.eval()
with torch.no_grad():
    a = head(features, logits, prompt, mask, weight)
    b = head(features, logits, modified, mask, weight)
assert torch.allclose(a, b, atol=1e-5), 'Padded context changed predictions'
targets = torch.randint(0, 11, (2, 3))
valid = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
loss, parts = refinement_loss(a, logits, targets, valid, error_boost=1, trust_weight=1)
ce = F.cross_entropy(a.transpose(1, 2), targets, reduction='none')
kl = F.kl_div(F.log_softmax(a, -1), F.softmax(logits, -1), reduction='none').sum(-1)
expected = ((ce + kl) * valid).sum() / valid.sum()
assert torch.allclose(loss, expected, atol=1e-5)
assert torch.isfinite(parts['trust_kl'])
try:
    head(features, logits, prompt, torch.zeros_like(mask), weight)
    raise AssertionError('Empty prompts must fail')
except ValueError:
    pass
print('Context refinement checks passed.')
