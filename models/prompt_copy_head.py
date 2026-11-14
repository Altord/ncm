"""Prompt-token pointer mixture for a frozen parallel codec decoder."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptCopyHead(nn.Module):
    def __init__(self, width, confidence_aware_gate=False, context_layers=0,
                 context_heads=4, context_max_length=256):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)
        self.confidence_gate = nn.Linear(3, 1, bias=False) if confidence_aware_gate else None
        if self.confidence_gate is not None:
            nn.init.zeros_(self.confidence_gate.weight)
        if int(context_layers) < 0 or int(context_heads) < 1 or int(context_max_length) < 1:
            raise ValueError('Invalid prompt context encoder configuration')
        if int(context_layers) and width % int(context_heads):
            raise ValueError('Prompt context width must be divisible by the head count')
        self.context_position = None
        self.context_encoder = None
        if int(context_layers):
            self.context_position = nn.Embedding(int(context_max_length), width)
            nn.init.normal_(self.context_position.weight, mean=0.0, std=0.02)
            layer = nn.TransformerEncoderLayer(d_model=width, nhead=int(context_heads),
                dim_feedforward=4 * width, dropout=0.0, batch_first=True, norm_first=True)
            self.context_encoder = nn.TransformerEncoder(layer, num_layers=int(context_layers),
                                                         enable_nested_tensor=False)
        self.scale = width ** -0.5

    def distribution(self, features, prompt_embeddings, prompt_mask, base_logits=None, prompt_ids=None):
        if prompt_mask.ndim != 2 or not prompt_mask.bool().any(dim=1).all():
            raise ValueError('Each prompt must have at least one unmasked token')
        query = self.query(features.float())
        prompt_features = prompt_embeddings.float()
        if self.context_encoder is not None:
            length = prompt_features.shape[1]
            if length > self.context_position.num_embeddings:
                raise ValueError('Prompt exceeds the context encoder position limit')
            positions = torch.arange(length, device=prompt_features.device)
            prompt_features = prompt_features + self.context_position(positions)[None, :, :]
            prompt_features = self.context_encoder(prompt_features, src_key_padding_mask=~prompt_mask.bool())
        keys = self.key(prompt_features)
        scores = torch.matmul(query, keys.transpose(1, 2)) * self.scale
        scores = scores.masked_fill(~prompt_mask[:, None, :].bool(), float('-inf'))
        attention = scores.softmax(dim=-1)
        gate = self.gate(features.float()).squeeze(-1)
        if self.confidence_gate is not None:
            if base_logits is None or prompt_ids is None:
                raise ValueError('Confidence-aware gate requires codec logits and prompt token IDs')
            top = base_logits.float().topk(2, dim=-1)
            margin = (top.values[..., 0] - top.values[..., 1]).clamp(0, 20) / 20
            concentration = attention.max(dim=-1).values
            base_matches = (prompt_ids[:, None, :] == top.indices[..., 0, None]) & prompt_mask[:, None, :].bool()
            support = (attention * base_matches).sum(dim=-1)
            signals = torch.stack((margin, concentration, support), dim=-1)
            gate = gate + self.confidence_gate(signals).squeeze(-1)
        return attention, gate

    def nll(self, features, base_logits, prompt_embeddings, prompt_ids, prompt_mask, targets, valid,
            copyable_error_boost=1.0, attention_aux_weight=0.0, gate_aux_weight=0.0,
            gate_positive_weight=1.0, return_components=False):
        if not math.isfinite(copyable_error_boost) or copyable_error_boost < 1:
            raise ValueError('copyable_error_boost must be finite and at least 1')
        if (not math.isfinite(attention_aux_weight) or attention_aux_weight < 0 or
            not math.isfinite(gate_aux_weight) or gate_aux_weight < 0 or
            not math.isfinite(gate_positive_weight) or gate_positive_weight < 1):
            raise ValueError('Auxiliary weights must be finite and nonnegative; gate positive weight must be at least 1')
        attention, gate_logits = self.distribution(features, prompt_embeddings, prompt_mask,
                                                   base_logits=base_logits, prompt_ids=prompt_ids)
        matches = (prompt_ids[:, None, :] == targets[:, :, None]) & prompt_mask[:, None, :].bool()
        copy_prob = (attention * matches).sum(dim=-1)
        logits = base_logits.float()
        base_logp = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        combined = torch.logaddexp(F.logsigmoid(-gate_logits) + base_logp,
                                   F.logsigmoid(gate_logits) + copy_prob.clamp_min(1e-30).log())
        eligible = matches.any(dim=-1) & (base_logits.argmax(dim=-1) != targets) & valid.bool()
        mask = valid.float() * torch.where(eligible, copyable_error_boost, 1.0)
        main = -(combined * mask).sum() / mask.sum().clamp_min(1)
        eligible_count = eligible.sum()
        attention_loss = -(copy_prob.clamp_min(1e-30).log() * eligible.float()).sum() / eligible_count.clamp_min(1)
        gate_weights = valid.float() * torch.where(eligible, gate_positive_weight, 1.0)
        gate_loss = (F.binary_cross_entropy_with_logits(gate_logits, eligible.float(), reduction='none') * gate_weights).sum() / gate_weights.sum().clamp_min(1)
        loss = main + attention_aux_weight * attention_loss + gate_aux_weight * gate_loss
        if return_components:
            return loss, {'mixture_nll': main.detach(), 'copy_attention_nll': attention_loss.detach(),
                          'copy_gate_bce': gate_loss.detach(), 'copyable_errors': eligible_count.detach()}
        return loss

    def predict(self, features, base_logits, prompt_embeddings, prompt_ids, prompt_mask):
        attention, gate_logits = self.distribution(features, prompt_embeddings, prompt_mask,
                                                   base_logits=base_logits, prompt_ids=prompt_ids)
        gate = gate_logits.sigmoid().unsqueeze(-1)
        probs = base_logits.float().softmax(dim=-1) * (1 - gate)
        indices = prompt_ids[:, None, :].expand(-1, features.size(1), -1)
        probs.scatter_add_(-1, indices, attention * gate)
        return probs.argmax(dim=-1), gate.squeeze(-1)
