"""Token-weight helpers for code-focused tokenizer reconstruction."""

from __future__ import annotations

import re

import numpy as np
import torch


CSHARP_KEYWORDS = """abstract as base bool break byte case catch char checked class const
continue decimal default delegate do double else enum event explicit extern false
finally fixed float for foreach goto if implicit in int interface internal is lock
long namespace new null object operator out override params private protected public
readonly ref return sbyte sealed short sizeof stackalloc static string struct switch
this throw true try typeof uint ulong unchecked unsafe ushort using virtual void
volatile while var async await record required init get set add remove partial
""".split()

SYNTAX_MARKERS = "{}[]();,.?:=+-*/%!&|<>~^#@\\\"'`$"
CSHARP_KEYWORD_SET = set(CSHARP_KEYWORDS)


def training_token_frequency_weights(cache, vocab_size, threshold=32, max_weight=2.0):
    """Count valid training-cache tokens and return bounded inverse-frequency weights."""
    if threshold < 1 or not 1 <= max_weight <= 10:
        raise ValueError('threshold must be positive and max_weight must be between 1 and 10')
    if 'tokens' not in cache._arrays or 'mask' not in cache._arrays:
        raise ValueError('Training cache must contain tokens and mask')
    counts = np.zeros(int(vocab_size), dtype=np.int64)
    for begin in range(0, len(cache), 1024):
        tokens = np.asarray(cache._arrays['tokens'][begin:begin+1024])
        mask = np.asarray(cache._arrays['mask'][begin:begin+1024], dtype=bool)
        valid = tokens[mask].astype(np.int64, copy=False)
        if valid.size and (valid.min() < 0 or valid.max() >= vocab_size):
            raise ValueError('Training cache contains token IDs outside the codec vocabulary')
        counts += np.bincount(valid, minlength=int(vocab_size))
    weights = np.ones(int(vocab_size), dtype=np.float32)
    observed = counts > 0
    weights[observed] = np.clip(np.sqrt(float(threshold) / counts[observed]), 1.0, max_weight)
    return torch.from_numpy(weights), counts


def repeated_identifier_piece_factors(targets, valid, occurrence_weights,
                                      identifier_weight, multiplier=1.5):
    """Extra weights for repeated identifier/literal-labeled token IDs within each chunk.

    This is a token-piece heuristic, not symbol resolution or an inference copy path.
    """
    if not 1 <= multiplier <= 4:
        raise ValueError('repeat multiplier must be between 1 and 4')
    if targets.ndim != 2 or valid.shape != targets.shape or occurrence_weights.shape != targets.shape:
        raise ValueError('targets, valid, and occurrence weights must share [batch, sequence] shape')
    eligible = valid.bool() & (occurrence_weights == float(identifier_weight))
    same = targets.unsqueeze(2) == targets.unsqueeze(1)
    counts = (same & eligible.unsqueeze(1)).sum(dim=2)
    repeated = eligible & (counts > 1)
    return torch.where(repeated, torch.full_like(occurrence_weights, float(multiplier)),
                       torch.ones_like(occurrence_weights))


def topk_teacher_distillation_loss(student_logits, targets, valid, teacher_topk):
    """Train toward teacher top-k distributions only where that teacher's top-1 is correct.

    Teachers supply detached (indices, normalized probabilities) tensors. The
    teacher distribution is renormalized over its top-k, not the full vocabulary.
    """
    if not teacher_topk:
        raise ValueError('At least one teacher is required')
    log_normalizer = torch.logsumexp(student_logits.float(), dim=-1)
    numerator = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    denominator = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    for indices, probabilities in teacher_topk:
        if indices.shape != probabilities.shape or indices.shape[:2] != targets.shape:
            raise ValueError('Teacher top-k shape must match targets')
        mask = valid.bool() & (indices[..., 0] == targets)
        chosen = student_logits.gather(-1, indices).float()
        token_loss = log_normalizer - (probabilities * chosen).sum(dim=-1)
        numerator = numerator + (token_loss * mask).sum()
        denominator = denominator + mask.sum()
    return numerator / denominator.clamp_min(1), denominator


def make_code_token_weights(tokenizer, vocab_size: int, syntax_weight: float,
                            identifier_weight: float):
    """Build per-token loss weights from the configured HF tokenizer.

    This is a lexical heuristic: syntax punctuation and C# reserved words get
    the syntax weight; alphanumeric pieces get the identifier/literal weight;
    other pieces retain weight 1. It changes only training CE, never targets or
    evaluation metrics.
    """
    hf_tokenizer = getattr(tokenizer, "hf_tokenizer", tokenizer)
    weights = [1.0] * int(vocab_size)

    for token_id in range(min(int(vocab_size), int(hf_tokenizer.vocab_size))):
        piece = hf_tokenizer.decode(
            [token_id], skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if piece.strip() in CSHARP_KEYWORD_SET or any(
            marker in piece for marker in SYNTAX_MARKERS
        ):
            weights[token_id] = max(weights[token_id], float(syntax_weight))
        elif re.search(r"[A-Za-z_0-9]", piece):
            weights[token_id] = max(weights[token_id], float(identifier_weight))

    return weights


def select_reconstruction_weights(targets, vocabulary_weights=None,
                                 occurrence_weights=None):
    """Choose cached per-position weights or fall back to vocabulary weights."""
    if occurrence_weights is not None:
        return occurrence_weights.to(device=targets.device, dtype=torch.float32)
    if vocabulary_weights is not None:
        return vocabulary_weights[targets]
    return None
