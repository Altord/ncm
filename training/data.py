"""Dataset and tokenizer helpers used by the NCM trainer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from utils.registry import tokenizers as TOKENIZER_REGISTRY
import hydra


def load_base_tokenizer(config):
    name = str(config.name)
    if name not in TOKENIZER_REGISTRY:
        raise ValueError(f"Unknown tokenizer name {name!r}; available: {sorted(TOKENIZER_REGISTRY)}")
    cls = hydra.utils.get_class(TOKENIZER_REGISTRY[name])
    return cls.from_config(config)


def iter_records(path: str | Path, fmt: str = "jsonl") -> Iterator[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    if fmt == "jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL: {e}") from e
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{line_no}: each JSONL row must be an object")
                yield item
    elif fmt == "text":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield {"text": line.rstrip("\r\n")}
    else:
        raise ValueError(f"Unsupported data format {fmt!r}; use 'jsonl' or 'text'")


def encode_text(tokenizer, text: str, append_eos: bool) -> list[int]:
    ids = list(map(int, tokenizer.encode(text)))
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if append_eos and eos_id is not None and (not ids or ids[-1] != int(eos_id)):
        ids.append(int(eos_id))
    return ids


def fixed_sequence(ids: list[int], length: int, pad_id: int, eos_id: int | None = None):
    ids = ids[:length]
    if len(ids) == length and eos_id is not None and length > 0:
        # Preserve an end marker when a long example is truncated.
        ids[-1] = int(eos_id)
    mask = [1] * len(ids)
    if len(ids) < length:
        n_pad = length - len(ids)
        ids.extend([int(pad_id)] * n_pad)
        mask.extend([0] * n_pad)
    return ids, mask


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_rows(path: Path, rows, dtype) -> int:
    count = 0
    with path.open("wb") as f:
        for row in rows:
            np.asarray(row, dtype=dtype).tofile(f)
            count += 1
    return count


def save_cache_manifest(directory: Path, kind: str, count: int, arrays: dict, metadata: dict):
    manifest = {
        "format_version": 1,
        "kind": kind,
        "count": int(count),
        "arrays": arrays,
        **metadata,
    }
    tmp = directory / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(directory / "manifest.json")


class MemmapCache(Dataset):
    """Read fixed-shape integer arrays without loading a whole cache into RAM."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Prepared cache missing at {self.directory}; run training.prepare_data first"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.count = int(self.manifest["count"])
        self._arrays = {}
        for key, spec in self.manifest["arrays"].items():
            dtype = np.dtype(spec["dtype"])
            shape = (self.count, *map(int, spec["shape"]))
            path = self.directory / spec["file"]
            self._arrays[key] = np.memmap(path, mode="r", dtype=dtype, shape=shape)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {
            key: torch.from_numpy(np.array(array[index], copy=True))
            for key, array in self._arrays.items()
        }


def load_resolved_config(path):
    return OmegaConf.load(path)
