"""Train the multi-scale VQ tokenizer reconstruction model."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.vqvae import VQVAE
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
    run = Run(cfg, "tokenizer", args.config)
    loader = make_loader(cfg.data.cache_dir, cfg.train.batch_size,
                         cfg.train.num_workers, True, cfg.train.seed)
    val_loader = None
    if cfg.train.get("validation_cache_dir"):
        val_loader = make_loader(cfg.train.validation_cache_dir, cfg.train.batch_size,
                                 cfg.train.num_workers, False, cfg.train.seed)
    model = VQVAE(**OmegaConf.create(OmegaConf.to_container(cfg.vqvae, resolve=True))).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=float(cfg.train.lr),
                              betas=tuple(cfg.train.betas),
                              weight_decay=float(cfg.train.weight_decay))
    scaler = make_scaler(device, cfg.train.precision)
    step = load_resume(args.resume, model, optim, scaler, device)
    model.train()
    accum = int(cfg.train.grad_accum_steps)
    max_steps = int(cfg.train.max_steps)
    log_every = int(cfg.train.log_every)
    save_every = int(cfg.train.save_every)
    eval_every = int(cfg.train.get("eval_every", save_every))
    mask_padding = bool(cfg.train.get("mask_padding", True))
    vq_weight = float(cfg.train.get("vq_loss_weight", 1.0))
    optim.zero_grad(set_to_none=True)
    running = {"loss": 0.0, "recon_ce": 0.0, "vq": 0.0}
    micro = 0
    since_log = 0
    last_log_time = time.perf_counter()
    best_val = float("inf")
    last_util = None

    def validate(current_step):
        nonlocal best_val
        if val_loader is None:
            return
        if len(val_loader) == 0:
            raise RuntimeError("Validation cache is empty")
        model.eval()
        totals = {"loss": 0.0, "recon_ce": 0.0, "vq": 0.0}
        seen = 0
        with torch.no_grad():
            for vb in val_loader:
                vx = vb["tokens"].to(device, non_blocking=True).long()
                valid = vb["mask"].to(device, non_blocking=True).bool()
                with autocast_context(device, cfg.train.precision):
                    logits, vq_loss, _, _, _ = model(vx)
                    ce = F.cross_entropy(logits.float().transpose(1, 2), vx, reduction="none")
                    ce = ((ce * valid).sum() / valid.sum().clamp_min(1)
                          if mask_padding else ce.mean())
                    loss = ce + vq_weight * vq_loss.float()
                n = int(vx.shape[0])
                seen += n
                totals["loss"] += float(loss) * n
                totals["recon_ce"] += float(ce) * n
                totals["vq"] += float(vq_loss) * n
        model.train()
        record = {"stage": "tokenizer", "step": current_step, "examples": seen,
                  **{f"val_{k}": v / max(1, seen) for k, v in totals.items()}}
        run.log_validation(record)
        if record["val_loss"] < best_val:
            best_val = record["val_loss"]
            run.checkpoint("best.pt", checkpoint_state(
                model, optim, scaler, current_step - 1, cfg, "tokenizer",
                {"best_validation_loss": best_val}))

    while step < max_steps:
        progressed = False
        for batch in loader:
            progressed = True
            x = batch["tokens"].to(device, non_blocking=True).long()
            valid = batch["mask"].to(device, non_blocking=True).bool()
            with autocast_context(device, cfg.train.precision):
                logits, vq_loss, per_level_util, global_util, _ = model(x)
                ce = F.cross_entropy(logits.float().transpose(1, 2), x,
                                     reduction="none")
                if mask_padding:
                    ce = (ce * valid).sum() / valid.sum().clamp_min(1)
                else:
                    ce = ce.mean()
                loss = ce + vq_weight * vq_loss.float()
            last_util = {"per_level_codebook_utilization": per_level_util,
                         "global_codebook_utilization": global_util}
            scaler.scale(loss / accum).backward()
            micro += 1
            n_examples = int(x.shape[0])
            for key, val in (("loss", loss), ("recon_ce", ce), ("vq", vq_loss)):
                running[key] += float(val.detach()) * n_examples
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
                run.log({"stage": "tokenizer", "step": step, "lr": lr,
                         **{k: v / denom for k, v in running.items()},
                         "examples_per_second": denom / elapsed,
                         "tokens_per_second": denom * int(x.shape[1]) / elapsed,
                         **(last_util or {}),
                         "peak_vram_gib": (torch.cuda.max_memory_allocated(device) / 1024**3
                                           if device.type == "cuda" else None)})
                running = {k: 0.0 for k in running}
                since_log = 0
                last_log_time = time.perf_counter()
            if val_loader is not None and (step % eval_every == 0 or step == max_steps):
                validate(step)
            if step % save_every == 0 or step == max_steps:
                state = checkpoint_state(model, optim, scaler, step - 1, cfg, "tokenizer")
                run.checkpoint(f"step-{step:08d}.pt", state)
            if step >= max_steps:
                break
        if not progressed:
            raise RuntimeError("Prepared cache is empty")
    final = run.checkpoint("final.pt", checkpoint_state(model, optim, scaler,
                                                        max_steps - 1, cfg, "tokenizer"))
    export_artifact(run, cfg, str(cfg.train.get("artifact_filename", "tokenizer.pt")), final, {
        "stage": "tokenizer", "step": max_steps,
        "cache_manifest": str(Path(cfg.data.cache_dir, "manifest.json").resolve()),
        "code": "NCM tokenizer reconstruction and VQ objective",
    })
    print(f"Training complete. Checkpoints: {run.path / 'checkpoints'}; artifact: {cfg.train.artifact_dir}", flush=True)


if __name__ == "__main__":
    main()
