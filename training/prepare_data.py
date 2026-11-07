"""Prepare fixed-shape memory-mapped token or frozen-VQ code datasets."""

from __future__ import annotations

import argparse
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
            yield ids, mask

    token_path, mask_path = out / "tokens.bin", out / "mask.bin"
    # Stream into temporary files, then publish a manifest only after complete writes.
    token_tmp, mask_tmp = out / "tokens.bin.tmp", out / "mask.bin.tmp"
    count = 0
    with token_tmp.open("wb") as ft, mask_tmp.open("wb") as fm:
        for ids, mask in rows():
            np.asarray(ids, dtype=np.int32).tofile(ft)
            np.asarray(mask, dtype=np.uint8).tofile(fm)
            count += 1
    if count == 0:
        raise ValueError("No training records were found")
    token_tmp.replace(token_path)
    mask_tmp.replace(mask_path)
    save_cache_manifest(out, "tokenizer", count, {
        "tokens": {"file": token_path.name, "dtype": "int32", "shape": [seq_len]},
        "mask": {"file": mask_path.name, "dtype": "uint8", "shape": [seq_len]},
    }, {
        "source": str(Path(dcfg.input).resolve()),
        "source_sha256": sha256_file(dcfg.input),
        "tokenizer": str(cfg.tokenizer.name),
        "tokenizer_model_id": str(cfg.tokenizer.get("model_id", "")),
        "pad_id": pad_id,
        "eos_id": eos_id,
        "sequence_length": seq_len,
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
