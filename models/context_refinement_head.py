"""Context-conditioned, vocabulary-wide refinement for a frozen codec."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextRefinementHead(nn.Module):
    def __init__(self, width, context_heads=4, max_prompt_length=256, bottleneck=64):
        super().__init__()
        if width % context_heads or min(context_heads, max_prompt_length, bottleneck) < 1:
            raise ValueError('Invalid refinement head dimensions')
        self.position = nn.Embedding(max_prompt_length, width)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=context_heads,
            dim_feedforward=4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.context_encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.query_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, context_heads, dropout=0.0, batch_first=True)
        self.down = nn.Linear(2 * width, bottleneck)
        self.up = nn.Linear(bottleneck, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, features, base_logits, prompt_embeddings, prompt_mask, output_weight):
        if prompt_mask.ndim != 2 or not prompt_mask.bool().any(dim=1).all():
            raise ValueError('Each prompt must have at least one unmasked token')
        if prompt_embeddings.shape[1] > self.position.num_embeddings:
            raise ValueError('Prompt exceeds position limit')
        positions = torch.arange(prompt_embeddings.shape[1], device=prompt_embeddings.device)
        context = prompt_embeddings.float() + self.position(positions)[None, :, :]
        context = self.context_encoder(context, src_key_padding_mask=~prompt_mask.bool())
        attention, _ = self.cross_attention(self.query_norm(features.float()), context, context,
                                           key_padding_mask=~prompt_mask.bool(), need_weights=False)
        delta = self.up(F.gelu(self.down(torch.cat((features.float(), attention), dim=-1))))
        return base_logits.float() + F.linear(delta, output_weight.float())


def refinement_loss(refined, baseline, target, valid, error_boost=20.0, trust_weight=0.05):
    """Weighted ground-truth CE plus KL trust term; labels are training-only."""
    if not math.isfinite(error_boost) or error_boost < 1 or not math.isfinite(trust_weight) or trust_weight < 0:
        raise ValueError('Invalid refinement loss weights')
    valid = valid.bool()
    errors = (baseline.argmax(-1) != target) & valid
    weights = valid.float() * torch.where(errors, error_boost, 1.0)
    ce = F.cross_entropy(refined.transpose(1, 2), target, reduction='none')
    weighted_ce = (ce * weights).sum() / weights.sum().clamp_min(1)
    if trust_weight:
        baseline = baseline.float()
        with torch.no_grad():
            baseline_probs = baseline.softmax(dim=-1)
            baseline_log_z = torch.logsumexp(baseline, dim=-1)
        kl_per_token = (baseline_probs * (baseline - refined)).sum(dim=-1) + torch.logsumexp(refined, dim=-1) - baseline_log_z
        kl = (kl_per_token * valid.float()).sum() / valid.float().sum().clamp_min(1)
    else:
        kl = weighted_ce.new_zeros(())
    loss = weighted_ce + trust_weight * kl
    return loss, {'weighted_ce': weighted_ce.detach(), 'trust_kl': kl.detach(),
                  'codec_errors': errors.sum().detach()}
