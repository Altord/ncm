"""Train the multi-scale VQ tokenizer reconstruction model."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time

# Set before importing torch. Explicit user settings, including 0, take priority.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.vqvae import VQVAE
from training.engine import (Run, autocast_context, checkpoint_state, device_from,
                             export_artifact, load_resume, lr_for_step,
                             make_loader, make_scaler, seed_all)
from training.data import load_base_tokenizer
from training.reconstruction_loss import make_code_token_weights, select_reconstruction_weights


def reconstruction_ce(logits, targets, valid, token_weights=None, mask_padding=True,
                      hard_fraction=0.25, hard_mix=0.0):
    """Return unweighted CE for reporting and (optionally) weighted train CE."""
    per_token = F.cross_entropy(logits.float().transpose(1, 2), targets,
                                reduction="none")
    valid_weights = valid.to(per_token.dtype) if mask_padding else torch.ones_like(per_token)
    raw_ce = (per_token * valid_weights).sum() / valid_weights.sum().clamp_min(1)
    if not 0 < hard_fraction <= 1 or not 0 <= hard_mix <= 1:
        raise ValueError('hard_fraction must be in (0, 1] and hard_mix in [0, 1]')
    weights = token_weights
    if weights is not None and weights.ndim == 1:
        weights = weights[targets]
    combined_weights = valid_weights if weights is None else weights * valid_weights
    weighted_ce = (per_token * combined_weights).sum() / combined_weights.sum().clamp_min(1)
    if hard_mix:
        denominators = combined_weights.sum(dim=1)
        valid_rows = denominators > 0
        if valid_rows.any():
            row_loss = ((per_token * combined_weights).sum(dim=1) /
                        denominators.clamp_min(1))[valid_rows]
            k = max(1, math.ceil(row_loss.numel() * hard_fraction))
            hard_ce = row_loss.topk(k).values.mean()
            weighted_ce = (1 - hard_mix) * weighted_ce + hard_mix * hard_ce
    return raw_ce, weighted_ce


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--stop-after-steps", type=int, default=None,
                    help="Stop at this total update count while keeping max-steps as the learning-rate schedule horizon.")
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
    if torch.version.hip:
        print("ROCm experimental AOTriton attention setting: "
              + os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"], flush=True)
    run = Run(cfg, "tokenizer", args.config)
    loader = make_loader(cfg.data.cache_dir, cfg.train.batch_size,
                         cfg.train.num_workers, True, cfg.train.seed)
    val_loader = None
    if cfg.train.get("validation_cache_dir"):
        val_loader = make_loader(cfg.train.validation_cache_dir, cfg.train.batch_size,
                                 cfg.train.num_workers, False, cfg.train.seed)
    model = VQVAE(**OmegaConf.create(OmegaConf.to_container(cfg.vqvae, resolve=True))).to(device)
    if model.autoregressive_decoder:
        print('Autoregressive decoder: training and checkpoint validation are teacher-forced; '
              'run codec evaluation for free-running accuracy and compilation.', flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=float(cfg.train.lr),
                              betas=tuple(cfg.train.betas),
                              weight_decay=float(cfg.train.weight_decay))
    scaler = make_scaler(device, cfg.train.precision)
    step = load_resume(args.resume, model, optim, scaler, device)
    model.train()
    accum = int(cfg.train.grad_accum_steps)
    max_steps = int(cfg.train.max_steps)
    stop_step = min(max_steps, args.stop_after_steps) if args.stop_after_steps is not None else max_steps
    if stop_step <= step:
        raise ValueError(f"stop-after-steps must be greater than loaded step {step} and positive")
    log_every = int(cfg.train.log_every)
    save_every = int(cfg.train.save_every)
    eval_every = int(cfg.train.get("eval_every", save_every))
    mask_padding = bool(cfg.train.get("mask_padding", True))
    vq_weight = float(cfg.train.get("vq_loss_weight", 1.0))
    hard_cfg = cfg.train.get('hard_examples', {})
    hard_mix = float(hard_cfg.get('mix', 0.0))
    hard_fraction = float(hard_cfg.get('fraction', 0.25))
    if not 0 <= hard_mix <= 1 or not 0 < hard_fraction <= 1:
        raise ValueError('hard_examples mix must be in [0, 1] and fraction in (0, 1]')
    if hard_mix:
        print(f'Hard-example loss: top {hard_fraction:.0%} of nonempty training chunks; '
              f'blend={hard_mix:.0%}. Validation is unweighted.', flush=True)
    weight_cfg = cfg.train.get("reconstruction_token_weights", None)
    token_weights = None
    occurrence_weighting = bool(
        weight_cfg and str(weight_cfg.get("mode", "")) == "per_occurrence"
    )
    if weight_cfg:
        if occurrence_weighting:
            cache_manifest = Path(cfg.data.cache_dir) / "manifest.json"
            if not cache_manifest.is_file():
                raise FileNotFoundError(f"Prepared cache manifest not found: {cache_manifest}")
            import json
            manifest = json.loads(cache_manifest.read_text(encoding="utf-8"))
            if "token_weights" not in manifest.get("arrays", {}):
                raise ValueError(
                    "Per-occurrence loss weights are enabled, but the prepared cache "
                    "has no token_weights.bin; rerun training.prepare_data"
                )
            print(
                "Per-occurrence reconstruction weighting enabled: "
                f"syntax={float(weight_cfg.get('syntax', 3.0))}, "
                f"identifier/literal={float(weight_cfg.get('identifier', 1.5))}; "
                "validation metrics remain unweighted.", flush=True
            )
        else:
            tokenizer = load_base_tokenizer(cfg.tokenizer)
            token_weights = torch.tensor(make_code_token_weights(
                tokenizer,
                int(cfg.vqvae.in_vocab_size),
                float(weight_cfg.get("syntax", 3.0)),
                float(weight_cfg.get("identifier", 1.5)),
            ), dtype=torch.float32, device=device)
            syntax_count = int((token_weights == float(weight_cfg.get("syntax", 3.0))).sum())
            identifier_count = int((token_weights == float(weight_cfg.get("identifier", 1.5))).sum())
            print(
                "Reconstruction token weighting enabled: "
                f"syntax={float(weight_cfg.get('syntax', 3.0))}, "
                f"identifier/literal={float(weight_cfg.get('identifier', 1.5))}; "
                f"weighted vocabulary ids={syntax_count + identifier_count} "
                f"(syntax={syntax_count}, identifier/literal={identifier_count}); "
                "validation metrics remain unweighted.", flush=True
            )
    optim.zero_grad(set_to_none=True)
    running = {"loss": 0.0, "recon_ce": 0.0, "weighted_recon_ce": 0.0, "vq": 0.0}
    micro = 0
    since_log = 0
    last_log_time = time.perf_counter()
    selection_metric = str(cfg.train.get("checkpoint_metric", "val_loss"))
    selection_mode = str(cfg.train.get("checkpoint_metric_mode", "min")).lower()
    if selection_mode not in {"min", "max"}:
        raise ValueError("train.checkpoint_metric_mode must be 'min' or 'max'")
    best_selection_value = float("inf") if selection_mode == "min" else -float("inf")
    best_tiebreak_value = -float("inf")
    best_checkpoint_path = None
    best_validation_step = None
    best_validation_record = None
    last_util = None

    def validate(current_step):
        nonlocal best_selection_value, best_tiebreak_value
        nonlocal best_checkpoint_path, best_validation_step, best_validation_record
        if val_loader is None:
            return
        if len(val_loader) == 0:
            raise RuntimeError("Validation cache is empty")
        model.eval()
        totals = {"loss": 0.0, "recon_ce": 0.0, "vq": 0.0}
        seen = 0
        valid_tokens = 0
        correct_tokens = 0
        exact_sequences = 0
        with torch.no_grad():
            for vb in val_loader:
                vx = vb["tokens"].to(device, non_blocking=True).long()
                valid = vb["mask"].to(device, non_blocking=True).bool()
                with autocast_context(device, cfg.train.precision):
                    logits, vq_loss, _, _, _ = model(vx)
                    ce, _ = reconstruction_ce(logits, vx, valid, None, mask_padding)
                    loss = ce + vq_weight * vq_loss.float()
                predictions = logits.argmax(dim=-1)
                valid_tokens += int(valid.sum())
                correct_tokens += int(((predictions == vx) & valid).sum())
                exact_sequences += int((((predictions == vx) | ~valid).all(dim=1)).sum())
                n = int(vx.shape[0])
                seen += n
                totals["loss"] += float(loss) * n
                totals["recon_ce"] += float(ce) * n
                totals["vq"] += float(vq_loss) * n
        model.train()
        record = {"stage": "tokenizer", "step": current_step, "examples": seen,
                  **{f"val_{k}": v / max(1, seen) for k, v in totals.items()},
                  "val_token_accuracy": correct_tokens / max(1, valid_tokens),
                  "val_exact_sequence_match_rate": exact_sequences / max(1, seen)}
        run.log_validation(record)
        if selection_metric not in record:
            raise ValueError(
                f"Unknown train.checkpoint_metric {selection_metric!r}; "
                f"available metrics: {sorted(record)}"
            )
        selected_value = float(record[selection_metric])
        tie_value = float(record["val_token_accuracy"])
        improved = (selected_value < best_selection_value if selection_mode == "min"
                    else selected_value > best_selection_value)
        tied_better = selected_value == best_selection_value and tie_value > best_tiebreak_value
        if improved or tied_better:
            best_selection_value = selected_value
            best_tiebreak_value = tie_value
            best_validation_step = current_step
            best_validation_record = dict(record)
            best_checkpoint_path = run.checkpoint("best.pt", checkpoint_state(
                model, optim, scaler, current_step - 1, cfg, "tokenizer",
                {"best_validation_metric": selection_metric,
                 "best_validation_value": best_selection_value,
                 "best_validation_token_accuracy": best_tiebreak_value}))

    while step < stop_step:
        progressed = False
        for batch in loader:
            progressed = True
            x = batch["tokens"].to(device, non_blocking=True).long()
            valid = batch["mask"].to(device, non_blocking=True).bool()
            with autocast_context(device, cfg.train.precision):
                logits, vq_loss, per_level_util, global_util, _ = model(x)
                occurrence_weights = (
                    batch["token_weights"].to(device, non_blocking=True).float()
                    if occurrence_weighting else None
                )
                reconstruction_weights = select_reconstruction_weights(
                    x, token_weights, occurrence_weights
                )
                ce, train_ce = reconstruction_ce(
                    logits, x, valid, reconstruction_weights, mask_padding,
                    hard_fraction=hard_fraction, hard_mix=hard_mix,
                )
                loss = train_ce + vq_weight * vq_loss.float()
            last_util = {"per_level_codebook_utilization": per_level_util,
                         "global_codebook_utilization": global_util}
            scaler.scale(loss / accum).backward()
            micro += 1
            n_examples = int(x.shape[0])
            for key, val in (("loss", loss), ("recon_ce", ce),
                             ("weighted_recon_ce", train_ce), ("vq", vq_loss)):
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
            if val_loader is not None and (step % eval_every == 0 or step == stop_step):
                validate(step)
            if step % save_every == 0 or step == stop_step:
                state = checkpoint_state(model, optim, scaler, step - 1, cfg, "tokenizer")
                run.checkpoint(f"step-{step:08d}.pt", state)
            if step >= stop_step:
                break
        if not progressed:
            raise RuntimeError("Prepared cache is empty")
    final = run.checkpoint("final.pt", checkpoint_state(model, optim, scaler,
                                                        step - 1, cfg, "tokenizer"))
    # Export the checkpoint selected by held-out loss when this invocation
    # performed validation. Keep final.pt as the resumable end-of-run state.
    artifact_checkpoint = best_checkpoint_path or final
    export_artifact(run, cfg, str(cfg.train.get("artifact_filename", "tokenizer.pt")), artifact_checkpoint, {
        "stage": "tokenizer", "step": best_validation_step or step,
        "selected_checkpoint": str(artifact_checkpoint.name),
        "best_validation_metric": selection_metric if best_checkpoint_path else None,
        "best_validation_value": best_selection_value if best_checkpoint_path else None,
        "best_validation_token_accuracy": best_tiebreak_value if best_checkpoint_path else None,
        "cache_manifest": str(Path(cfg.data.cache_dir, "manifest.json").resolve()),
        "code": "NCM tokenizer reconstruction and VQ objective",
    })
    print(f"Training complete. Checkpoints: {run.path / 'checkpoints'}; artifact: {cfg.train.artifact_dir}", flush=True)
    print("\n=== Human-readable tokenizer summary ===")
    print(f"Updates completed: {step} (schedule horizon: {max_steps})")
    if best_validation_record is not None:
        print(
            f"Selected checkpoint: {artifact_checkpoint.name} at step {best_validation_step} "
            f"({selection_metric}={best_selection_value:.4f})"
        )
        print(
            f"Validation: token accuracy {best_validation_record['val_token_accuracy']:.2%}; "
            f"exact chunks {best_validation_record['val_exact_sequence_match_rate']:.2%}; "
            f"reconstruction CE {best_validation_record['val_recon_ce']:.4f}"
        )
        print("Next: run held-out codec evaluation and compile checks before generator training.")
    else:
        print(f"Selected checkpoint: {artifact_checkpoint.name} (no validation checkpoint was produced)")
        print("Next: prepare validation data and run codec evaluation before generator training.")
    print(f"Artifact: {Path(cfg.train.artifact_dir) / cfg.train.get('artifact_filename', 'tokenizer.pt')}", flush=True)


if __name__ == "__main__":
    main()
