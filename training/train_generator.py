"""Train the Next Concept Transformer over frozen VQ code sequences."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE
from training.data import sha256_file
from training.engine import (Run, autocast_context, checkpoint_state, device_from,
                             export_artifact, load_resume, lr_for_step,
                             make_loader, make_scaler, seed_all)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.device:
        cfg.train.device = args.device
    seed_all(int(cfg.train.seed))
    device = device_from(str(cfg.train.device))
    run = Run(cfg, "generator", args.config)
    loader = make_loader(cfg.data.cache_dir, cfg.train.batch_size,
                         cfg.train.num_workers, True, cfg.train.seed)
    val_loader = None
    if cfg.train.get("validation_cache_dir"):
        val_loader = make_loader(cfg.train.validation_cache_dir, cfg.train.batch_size,
                                 cfg.train.num_workers, False, cfg.train.seed)

    vq_cfg = OmegaConf.to_container(cfg.vqvae, resolve=True)
    tokenizer_path = Path(vq_cfg.pop("checkpoint_path"))
    tokenizer_ckpt = torch.load(tokenizer_path, map_location="cpu", weights_only=False)
    vqvae = VQVAE(**OmegaConf.create(vq_cfg))
    vqvae.load_state_dict(tokenizer_ckpt["model"], strict=True)
    vqvae.to(device).eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = NextConceptTransformer(vqvae_local=vqvae, config=OmegaConf.create(model_cfg)).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=float(cfg.train.lr),
                              betas=tuple(cfg.train.betas),
                              weight_decay=float(cfg.train.weight_decay))
    scaler = make_scaler(device, cfg.train.precision)
    step = load_resume(args.resume, model, optim, scaler, device)
    model.train()
    vqvae.eval()
    max_steps = int(cfg.train.max_steps)
    accum = int(cfg.train.grad_accum_steps)
    log_every = int(cfg.train.log_every)
    save_every = int(cfg.train.save_every)
    eval_every = int(cfg.train.get("eval_every", save_every))
    alpha = float(cfg.train.level_alpha)
    corrupt_start = float(cfg.train.get("corruption_start", 0.0))
    corrupt_end = float(cfg.train.get("corruption_end", 0.0))
    levels = list(map(int, vqvae.vq.levels))
    level_offsets = [0]
    for length in levels:
        level_offsets.append(level_offsets[-1] + length)
    target_first = 0 if (bool(model.query_level0) or bool(model.predict_level0)) else 1
    total_target_positions = sum(levels[target_first:])
    weights = [length ** (1.0 - alpha) for length in levels[target_first:]]
    weights = [w / sum(weights) for w in weights]
    optim.zero_grad(set_to_none=True)
    running = {"loss": 0.0, **{f"level_{i + target_first}_ce": 0.0
                                for i in range(len(levels[target_first:]))}}
    micro = 0
    since_log = 0
    last_log_time = time.perf_counter()
    best_val = float("inf")

    def _model_input(prefix_ids, codes, prefix_pad_mask):
        target_vecs = vqvae.vq.idx_to_ncm_input(codes)
        if bool(model.query_level0):
            q = model.build_l0_query_input(codes.shape[0], target_vecs.dtype)
            target_vecs = torch.cat([q, target_vecs], dim=1)
        if bool(model.use_ctx_encoder):
            return target_vecs
        if bool(getattr(model.config, "raw_token_prefix", False)):
            enc = vqvae.encoder
            if bool(getattr(model.config, "contextual_prefix", False)) and len(getattr(enc, "blocks", [])) > 0:
                prefix_vecs = enc.forward_nopos(prefix_ids)[:, :model.prefix_len, :]
            else:
                prefix_vecs = enc.wte(prefix_ids)[:, :model.prefix_len, :]
        else:
            prefix_vecs = vqvae.encode_z(prefix_ids)[:, :model.prefix_len, :]
        return torch.cat([prefix_vecs, target_vecs], dim=1)

    def _level_ce(logits, labels):
        values = []
        offset = 0
        for i, level_len in enumerate(levels[target_first:]):
            vocab = int(vqvae.vq.vqvae_vocab_sizes[target_first + i])
            part = logits[:, offset:offset + level_len, :vocab].float()
            truth = labels[:, offset:offset + level_len]
            values.append(F.cross_entropy(part.reshape(-1, vocab), truth.reshape(-1)))
            offset += level_len
        return values

    def validate(current_step):
        nonlocal best_val
        if val_loader is None:
            return
        if len(val_loader) == 0:
            raise RuntimeError("Validation cache is empty")
        model.eval()
        sums = [0.0] * len(weights)
        seen = 0
        with torch.no_grad():
            for vb in val_loader:
                vcodes = vb["codes"].to(device, non_blocking=True).long()
                vpfx = vb["prefix"].to(device, non_blocking=True).long()
                vmask = ~vb["prefix_mask"].to(device, non_blocking=True).bool()
                vin = _model_input(vpfx, vcodes, vmask)
                vlabels = vcodes[:, level_offsets[target_first]:]
                with autocast_context(device, cfg.train.precision):
                    vlogits = model(vin, prefix_ids=vpfx if model.use_ctx_encoder else None,
                                    prefix_pad_mask=vmask)
                    vlosses = _level_ce(vlogits, vlabels)
                    vtotal = sum(w * l for w, l in zip(weights, vlosses))
                n = int(vcodes.shape[0])
                seen += n
                for i, val in enumerate(vlosses):
                    sums[i] += float(val) * n
        model.train()
        per_level = [v / max(1, seen) for v in sums]
        val_loss = sum(w * val for w, val in zip(weights, per_level))
        record = {"stage": "generator", "step": current_step, "examples": seen,
                  "val_loss": val_loss,
                  **{f"val_level_{i + target_first}_ce": value
                     for i, value in enumerate(per_level)}}
        run.log_validation(record)
        if val_loss < best_val:
            best_val = val_loss
            run.checkpoint("best.pt", checkpoint_state(
                model, optim, scaler, current_step - 1, cfg, "generator",
                {"best_validation_loss": best_val}))

    while step < max_steps:
        progressed = False
        for batch in loader:
            progressed = True
            clean_codes = batch["codes"].to(device, non_blocking=True).long()
            prefix_ids = batch["prefix"].to(device, non_blocking=True).long()
            prefix_pad_mask = ~batch["prefix_mask"].to(device, non_blocking=True).bool()
            # Corrupt only conditioning codes; the prediction targets stay clean.
            corrupted = clean_codes.clone()
            n_levels = len(levels)
            for level_idx, (begin, end) in enumerate(zip(level_offsets[:-1], level_offsets[1:])):
                rate = corrupt_start + (corrupt_end - corrupt_start) * (
                    level_idx / max(n_levels - 1, 1))
                if rate <= 0:
                    continue
                replace = torch.rand(corrupted[:, begin:end].shape, device=device) < rate
                vocab = int(vqvae.vq.vqvae_vocab_sizes[level_idx])
                random_codes = torch.randint(vocab, corrupted[:, begin:end].shape, device=device)
                corrupted[:, begin:end] = torch.where(replace, random_codes,
                                                       corrupted[:, begin:end])

            with torch.no_grad():
                model_input = _model_input(prefix_ids, corrupted, prefix_pad_mask)
            labels = clean_codes[:, level_offsets[target_first]:]

            with autocast_context(device, cfg.train.precision):
                logits = model(model_input, prefix_ids=prefix_ids if model.use_ctx_encoder else None,
                               prefix_pad_mask=prefix_pad_mask)
                if logits.shape[1] != labels.shape[1]:
                    raise RuntimeError(f"Logit/label length mismatch: {logits.shape[1]} vs {labels.shape[1]}")
                level_losses = _level_ce(logits, labels)
                loss = sum(w * item for w, item in zip(weights, level_losses))

            scaler.scale(loss / accum).backward()
            micro += 1
            n_examples = int(clean_codes.shape[0])
            running["loss"] += float(loss.detach()) * n_examples
            for i, value in enumerate(level_losses):
                running[f"level_{i + target_first}_ce"] += float(value.detach()) * n_examples
            since_log += n_examples
            if micro % accum:
                continue
            lr = lr_for_step(step, max_steps, int(cfg.train.warmup_steps),
                             cfg.train.lr, cfg.train.min_lr)
            for group in optim.param_groups:
                group["lr"] = lr
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, float(cfg.train.grad_clip))
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % log_every == 0 or step == 1:
                denom = max(1, since_log)
                elapsed = max(1e-9, time.perf_counter() - last_log_time)
                run.log({"stage": "generator", "step": step, "lr": lr,
                         **{k: v / denom for k, v in running.items()},
                         "level_alpha": alpha,
                         "corruption_start": corrupt_start,
                         "corruption_end": corrupt_end,
                         "target_positions": total_target_positions,
                         "examples_per_second": denom / elapsed,
                         "code_positions_per_second": denom * total_target_positions / elapsed,
                         "peak_vram_gib": (torch.cuda.max_memory_allocated(device) / 1024**3
                                           if device.type == "cuda" else None)})
                running = {k: 0.0 for k in running}
                since_log = 0
                last_log_time = time.perf_counter()
            if val_loader is not None and (step % eval_every == 0 or step == max_steps):
                validate(step)
            if step % save_every == 0 or step == max_steps:
                state = checkpoint_state(model, optim, scaler, step - 1, cfg, "generator")
                run.checkpoint(f"step-{step:08d}.pt", state)
            if step >= max_steps:
                break
        if not progressed:
            raise RuntimeError("Prepared cache is empty")

    # Existing evaluator seeds level zero from this empirical training-code unigram.
    ds = loader.dataset
    counts = np.bincount(
        np.asarray(ds._arrays["codes"][:, :levels[0]]).reshape(-1),
        minlength=int(vqvae.vq.vqvae_vocab_sizes[0]),
    )
    unigram = torch.from_numpy(counts.astype(np.float32))
    unigram /= unigram.sum().clamp_min(1)
    final = run.checkpoint("final.pt", checkpoint_state(
        model, optim, scaler, max_steps - 1, cfg, "generator",
        {"inference": {"level0_unigram": unigram}}))
    export_artifact(run, cfg, str(cfg.train.get("artifact_filename", "generator.pt")), final, {
        "stage": "generator", "step": max_steps, "level_alpha": alpha,
        "tokenizer_checkpoint": str(tokenizer_path.resolve()),
        "tokenizer_checkpoint_sha256": sha256_file(tokenizer_path),
        "cache_manifest": str(Path(cfg.data.cache_dir, "manifest.json").resolve()),
        "code": "Next Concept Transformer per-level code prediction",
    })
    print(f"Training complete. Checkpoints: {run.path / 'checkpoints'}; artifact: {cfg.train.artifact_dir}", flush=True)


if __name__ == "__main__":
    main()
