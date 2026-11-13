"""Prompt-token pointer mixture for a frozen parallel codec decoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptCopyHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)
        self.scale = width ** -0.5

    def distribution(self, features, prompt_embeddings, prompt_mask):
        if prompt_mask.ndim != 2 or not prompt_mask.bool().any(dim=1).all():
            raise ValueError('Each prompt must have at least one unmasked token')
        query = self.query(features.float())
        keys = self.key(prompt_embeddings.float())
        scores = torch.matmul(query, keys.transpose(1, 2)) * self.scale
        scores = scores.masked_fill(~prompt_mask[:, None, :].bool(), float('-inf'))
        return scores.softmax(dim=-1), self.gate(features.float()).squeeze(-1)

    def nll(self, features, base_logits, prompt_embeddings, prompt_ids, prompt_mask, targets, valid):
        attention, gate_logits = self.distribution(features, prompt_embeddings, prompt_mask)
        matches = (prompt_ids[:, None, :] == targets[:, :, None]) & prompt_mask[:, None, :].bool()
        copy_prob = (attention * matches).sum(dim=-1)
        logits = base_logits.float()
        base_logp = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        combined = torch.logaddexp(F.logsigmoid(-gate_logits) + base_logp,
                                   F.logsigmoid(gate_logits) + copy_prob.clamp_min(1e-30).log())
        mask = valid.float()
        return -(combined * mask).sum() / mask.sum().clamp_min(1)

    def predict(self, features, base_logits, prompt_embeddings, prompt_ids, prompt_mask):
        attention, gate_logits = self.distribution(features, prompt_embeddings, prompt_mask)
        gate = gate_logits.sigmoid().unsqueeze(-1)
        probs = base_logits.float().softmax(dim=-1) * (1 - gate)
        indices = prompt_ids[:, None, :].expand(-1, features.size(1), -1)
        probs.scatter_add_(-1, indices, attention * gate)
        return probs.argmax(dim=-1), gate.squeeze(-1)
