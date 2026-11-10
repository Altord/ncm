"""Token-weight helpers for code-focused tokenizer reconstruction."""

from __future__ import annotations

import re

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
