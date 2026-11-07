"""Shared optimizer, logging, checkpoint, and device helpers."""

from __future__ import annotations

import json
import math
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from training.data import MemmapCache


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested but torch.cuda.is_available() is false")
    return device


def make_loader(path, batch_size, workers, shuffle, seed):
    ds = MemmapCache(path)
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(ds, batch_size=int(batch_size), shuffle=shuffle,
                      num_workers=int(workers), pin_memory=torch.cuda.is_available(),
                      persistent_workers=int(workers) > 0, drop_last=False,
                      generator=generator)


class Run:
    def __init__(self, cfg, stage, config_path):
        root = Path(cfg.train.run_dir)
        self.path = root
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (root / "samples").mkdir(parents=True, exist_ok=True)
        self.log_path = root / "logs" / "train.jsonl"
        self.validation_path = root / "logs" / "validation.jsonl"
        resolved = OmegaConf.to_container(cfg, resolve=True)
        (root / "config.resolved.yaml").write_text(
            OmegaConf.to_yaml(OmegaConf.create(resolved), resolve=True), encoding="utf-8")
        (root / "manifest.json").write_text(json.dumps({
            "stage": stage, "config_source": str(Path(config_path).resolve()),
            "created_unix": time.time(), "run_dir": str(root.resolve()),
            "repo_revision": self._repo_revision(),
            "python": platform.python_version(), "platform": platform.platform(),
            "pytorch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "rocm_runtime": getattr(torch.version, "hip", None),
            "device": str(cfg.train.device),
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        }, indent=2), encoding="utf-8")

    @staticmethod
    def _repo_revision():
        repo_root = Path(__file__).resolve().parents[1]
        try:
            return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                                  check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def log(self, record):
        record = {**record, "time_unix": time.time()}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    def log_validation(self, record):
        record = {**record, "time_unix": time.time()}
        with self.validation_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    def checkpoint(self, name, state):
        path = self.path / "checkpoints" / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, tmp)
        tmp.replace(path)
        if name != "best.pt":
            latest = self.path / "checkpoints" / "latest.pt"
            shutil.copy2(path, latest.with_suffix(".pt.tmp"))
            latest.with_suffix(".pt.tmp").replace(latest)
        return path


def lr_for_step(step, total, warmup, peak, floor):
    if warmup > 0 and step < warmup:
        return float(peak) * (step + 1) / warmup
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / span))
    return float(floor) + 0.5 * (float(peak) - float(floor)) * (1 + math.cos(math.pi * progress))


def load_resume(path, model, optimizer, scaler, device):
    if not path:
        return 0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if "rng" in ckpt:
        torch.set_rng_state(ckpt["rng"]["torch"])
        if torch.cuda.is_available() and ckpt["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
        random.setstate(ckpt["rng"]["python"])
        np.random.set_state(ckpt["rng"]["numpy"])
    return int(ckpt.get("step", -1)) + 1


def checkpoint_state(model, optimizer, scaler, step, cfg, stage, extra=None):
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": int(step), "stage": stage,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None,
                "python": random.getstate(), "numpy": np.random.get_state()},
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if extra:
        state.update(extra)
    return state


def autocast_context(device, precision):
    precision = str(precision).lower()
    if precision in ("none", "fp32", "float32") or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if precision in ("bf16", "bfloat16") else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_scaler(device, precision):
    enabled = device.type == "cuda" and str(precision).lower() in ("fp16", "float16")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def export_artifact(run, cfg, name, checkpoint, metadata):
    out = Path(cfg.train.artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / name
    shutil.copy2(checkpoint, target)
    cache_manifest = Path(cfg.data.cache_dir) / "manifest.json"
    if cache_manifest.is_file():
        metadata = {**metadata, "data_cache": json.loads(cache_manifest.read_text(encoding="utf-8"))}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    shutil.copy2(run.path / "config.resolved.yaml", out / "training.config.yaml")
    evaluation_config = cfg.train.get("evaluation_config")
    if evaluation_config and Path(evaluation_config).is_file():
        shutil.copy2(evaluation_config, out / "config.yaml")
    else:
        shutil.copy2(run.path / "config.resolved.yaml", out / "config.yaml")
    return target
