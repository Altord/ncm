"""Prepare fixed-shape memory-mapped token or frozen-VQ code datasets."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from models.vqvae import VQVAE
from training.data import (
    encode_text, fixed_sequence, iter_records, load_base_tokenizer,
    save_cache_manifest, sha256_file, write_rows,
)
from training.csharp_structure import token_occurrence_weights


def _load_config(path):
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return cfg


def prepare_tokenizer(cfg):
    dcfg = cfg.data
    tok = load_base_tokenizer(cfg.tokenizer)
    seq_len = int(cfg.model.context_length)
    pad_id = int(tok.pad_token_id)
    eos_id = int(tok.eos_token_id) if tok.eos_token_id is not None else None
    out = Path(dcfg.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    text_key = str(dcfg.get("text_key", "text"))
    append_eos = bool(dcfg.get("append_eos", True))
    weight_cfg = cfg.train.get("reconstruction_token_weights", None)
    occurrence_weights = bool(weight_cfg and str(weight_cfg.get("mode", "")) == "per_occurrence")
    hf_tokenizer = getattr(tok, "hf_tokenizer", None)
    if occurrence_weights and (hf_tokenizer is None or not getattr(hf_tokenizer, "is_fast", False)):
        raise ValueError("Per-occurrence code weights require a fast Hugging Face tokenizer")

    def rows():
        for record in iter_records(dcfg.input, str(dcfg.get("format", "jsonl"))):
            if text_key not in record or not isinstance(record[text_key], str):
                raise ValueError(f"Expected string field {text_key!r} in each input record")
            ids = encode_text(tok, record[text_key], append_eos)
            ids, mask = fixed_sequence(ids, seq_len, pad_id, eos_id)
            if max(ids, default=0) >= int(cfg.model.in_vocab_size):
                raise ValueError(
                    f"Encoded token id exceeds model.in_vocab_size={cfg.model.in_vocab_size}; "
                    "check added special tokens and vocabulary size"
                )
            if occurrence_weights:
                text = record[text_key]
                encoded = hf_tokenizer(text, return_offsets_mapping=True)
                offset_ids = list(map(int, encoded["input_ids"]))
                base_ids = list(map(int, tok.encode(text)))
                if offset_ids != base_ids:
                    raise ValueError("Tokenizer offset encoding does not match training token IDs")
                if Path(str(record.get("source", ""))).suffix.lower() == ".cs":
                    token_weights = token_occurrence_weights(
                        text, [tuple(map(int, pair)) for pair in encoded["offset_mapping"]],
                        float(weight_cfg.get("syntax", 3.0)),
                        float(weight_cfg.get("identifier", 1.5)),
                    )
                else:
                    # Keep project/XML records in the corpus without pretending they
                    # have C# token classes.
                    token_weights = [1.0] * len(base_ids)
                if append_eos and eos_id is not None and (not base_ids or base_ids[-1] != eos_id):
                    token_weights.append(1.0)
                token_weights = token_weights[:seq_len]
                if len(token_weights) < seq_len:
                    token_weights.extend([1.0] * (seq_len - len(token_weights)))
                # fixed_sequence replaces the final token with EOS on truncation.
                if len(base_ids) + int(append_eos and eos_id is not None) >= seq_len:
                    token_weights[seq_len - 1] = 1.0
                yield ids, mask, token_weights
            else:
                yield ids, mask

    token_path, mask_path = out / "tokens.bin", out / "mask.bin"
    weights_path = out / "token_weights.bin"
    # Stream into temporary files, then publish a manifest only after complete writes.
    token_tmp, mask_tmp = out / "tokens.bin.tmp", out / "mask.bin.tmp"
    weights_tmp = out / "token_weights.bin.tmp"
    count = 0
    with ExitStack() as stack:
        ft = stack.enter_context(token_tmp.open("wb"))
        fm = stack.enter_context(mask_tmp.open("wb"))
        fw = stack.enter_context(weights_tmp.open("wb")) if occurrence_weights else None
        for row in rows():
            ids, mask = row[:2]
            np.asarray(ids, dtype=np.int32).tofile(ft)
            np.asarray(mask, dtype=np.uint8).tofile(fm)
            if fw is not None:
                np.asarray(row[2], dtype=np.float32).tofile(fw)
            count += 1
    if count == 0:
        raise ValueError("No training records were found")
    token_tmp.replace(token_path)
    mask_tmp.replace(mask_path)
    arrays = {
        "tokens": {"file": token_path.name, "dtype": "int32", "shape": [seq_len]},
        "mask": {"file": mask_path.name, "dtype": "uint8", "shape": [seq_len]},
    }
    if occurrence_weights:
        weights_tmp.replace(weights_path)
        arrays["token_weights"] = {
            "file": weights_path.name, "dtype": "float32", "shape": [seq_len],
        }
    save_cache_manifest(out, "tokenizer", count, arrays, {
        "source": str(Path(dcfg.input).resolve()),
        "source_sha256": sha256_file(dcfg.input),
        "tokenizer": str(cfg.tokenizer.name),
        "tokenizer_model_id": str(cfg.tokenizer.get("model_id", "")),
        "pad_id": pad_id,
        "eos_id": eos_id,
        "sequence_length": seq_len,
        "occurrence_weighted": occurrence_weights,
    })
    print(f"Prepared {count:,} sequences at {out}", flush=True)


def prepare_generator(cfg, device):
    dcfg = cfg.data
    tok = load_base_tokenizer(cfg.tokenizer)
    vcfg = OmegaConf.to_container(cfg.vqvae, resolve=True)
    checkpoint = Path(vcfg.pop("checkpoint_path"))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = VQVAE(**OmegaConf.create(vcfg))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    context = int(model.context_length)
    prefix_len = int(cfg.model.prefix_len)
    pad_id, eos_id = int(tok.pad_token_id), int(tok.eos_token_id)
    prompt_key = str(dcfg.get("prompt_key", "prompt"))
    target_key = str(dcfg.get("target_key", "target"))
    append_eos = bool(dcfg.get("append_eos", True))
    out = Path(dcfg.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokens_tmp, prefixes_tmp, pmask_tmp = [out / (n + ".bin.tmp") for n in
                                             ("tokens", "prefix", "prefix_mask")]
    tokens_path, prefixes_path, pmask_path = [out / (n + ".bin") for n in
                                                ("tokens", "prefix", "prefix_mask")]
    total_positions = sum(model.vq.levels)
    count = 0

    def pad_prefix(text):
        ids = encode_text(tok, text, append_eos=False)[:prefix_len]
        if not ids:
            ids = [eos_id]
        mask = [1] * len(ids)
        n = prefix_len - len(ids)
        ids += [pad_id] * n
        mask += [0] * n
        return ids, mask

    with tokens_tmp.open("wb") as ft, prefixes_tmp.open("wb") as fp, pmask_tmp.open("wb") as fm:
        batch_tokens, batch_prefix = [], []

        def flush():
            nonlocal count
            if not batch_tokens:
                return
            x = torch.tensor(batch_tokens, dtype=torch.long, device=device)
            with torch.no_grad():
                codes = model.encode(x).cpu().numpy().astype(np.int32, copy=False)
            np.asarray(codes).tofile(ft)
            np.asarray(batch_prefix, dtype=np.int32).tofile(fp)
            np.asarray(batch_pmasks, dtype=np.uint8).tofile(fm)
            count += len(batch_tokens)
            batch_tokens.clear()
            batch_prefix.clear()
            batch_pmasks.clear()

        batch_pmasks = []
        for record in iter_records(dcfg.input, str(dcfg.get("format", "jsonl"))):
            if target_key not in record or not isinstance(record[target_key], str):
                raise ValueError(f"Expected string field {target_key!r} in each input record")
            target = encode_text(tok, record[target_key], append_eos)
            target, _ = fixed_sequence(target, context, pad_id, eos_id)
            if max(target, default=0) >= model.in_vocab_size:
                raise ValueError("A token ID is outside the VQ-VAE input vocabulary")
            if prompt_key in record and record[prompt_key] is not None:
                prefix, pmask = pad_prefix(str(record[prompt_key]))
            else:
                # Unconditional generation: provide EOS followed by padding.
                prefix, pmask = [eos_id] + [pad_id] * (prefix_len - 1), [1] + [0] * (prefix_len - 1)
            batch_tokens.append(target)
            batch_prefix.append(prefix)
            batch_pmasks.append(pmask)
            if len(batch_tokens) >= int(dcfg.get("encode_batch_size", 64)):
                flush()
        flush()

    if count == 0:
        raise ValueError("No training records were found")
    for tmp, final in ((tokens_tmp, tokens_path), (prefixes_tmp, prefixes_path),
                       (pmask_tmp, pmask_path)):
        tmp.replace(final)
    save_cache_manifest(out, "generator", count, {
        "codes": {"file": tokens_path.name, "dtype": "int32", "shape": [total_positions]},
        "prefix": {"file": prefixes_path.name, "dtype": "int32", "shape": [prefix_len]},
        "prefix_mask": {"file": pmask_path.name, "dtype": "uint8", "shape": [prefix_len]},
    }, {
        "source": str(Path(dcfg.input).resolve()),
        "source_sha256": sha256_file(dcfg.input),
        "tokenizer_model_id": str(cfg.tokenizer.get("model_id", "")),
        "vqvae_checkpoint": str(checkpoint.resolve()),
        "vqvae_checkpoint_sha256": sha256_file(checkpoint),
        "levels": list(map(int, model.vq.levels)),
        "prefix_len": prefix_len,
    })
    print(f"Prepared {count:,} code sequences at {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=("tokenizer", "generator"), required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--input", default=None, help="Override data.input for a separate split")
    ap.add_argument("--cache-dir", default=None, help="Override data.cache_dir")
    args = ap.parse_args()
    cfg = _load_config(args.config)
    if args.input:
        cfg.data.input = args.input
    if args.cache_dir:
        cfg.data.cache_dir = args.cache_dir
    prepare_tokenizer(cfg) if args.stage == "tokenizer" else prepare_generator(cfg, args.device)


if __name__ == "__main__":
    main()
