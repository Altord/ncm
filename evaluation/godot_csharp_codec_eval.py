"""Measure frozen VQ-VAE round-trip quality on held-out Godot/C# snippets."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from models.vqvae import VQVAE
from training.data import encode_text, fixed_sequence, iter_records, load_base_tokenizer


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def visible_text(tokenizer, ids: list[int], eos_id: int, pad_id: int) -> str:
    if eos_id in ids:
        ids = ids[:ids.index(eos_id)]
    ids = [token_id for token_id in ids if token_id != pad_id]
    return tokenizer.decode(ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/train/godot-csharp-generator-pilot.yaml",
        help="Config whose tokenizer and VQ-VAE architecture match the trained artifact.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Tokenizer artifact; defaults to vqvae.checkpoint_path in the config.",
    )
    parser.add_argument(
        "--input", default="examples/godot_csharp_generator_pilot/test.jsonl",
    )
    parser.add_argument(
        "--output", default="runs/godot-csharp/generator-pilot/test/codec_roundtrip.jsonl",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = OmegaConf.load(repo_path(args.config))
    OmegaConf.resolve(cfg)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device requested but torch.cuda.is_available() is false")

    tokenizer = load_base_tokenizer(cfg.tokenizer)
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise SystemExit("Base tokenizer must define both PAD and EOS token IDs")
    pad_id, eos_id = int(tokenizer.pad_token_id), int(tokenizer.eos_token_id)

    vq_config = OmegaConf.to_container(cfg.vqvae, resolve=True)
    configured_checkpoint = vq_config.pop("checkpoint_path")
    checkpoint_path = repo_path(args.checkpoint or configured_checkpoint)
    if not checkpoint_path.is_file():
        raise SystemExit(f"Tokenizer checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = VQVAE(**OmegaConf.create(vq_config))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    sequence_length = int(cfg.model.context_length)
    output_rows = []
    total_valid = 0
    total_correct = 0
    total_exact = 0
    total_char_similarity = 0.0
    started = time.perf_counter()

    for record in iter_records(repo_path(args.input), "jsonl"):
        if not isinstance(record.get("target"), str):
            raise ValueError(f"Record {record.get('id', '<unknown>')} has no string target")
        target_ids = encode_text(tokenizer, record["target"], append_eos=True)
        padded_ids, mask = fixed_sequence(target_ids, sequence_length, pad_id, eos_id)
        tokens = torch.tensor([padded_ids], dtype=torch.long, device=device)
        valid_count = sum(mask)

        with torch.inference_mode():
            codes = model.encode(tokens)
            logits = model.decode(codes)
            predicted_ids = logits.argmax(dim=-1)[0].tolist()

        valid_truth = padded_ids[:valid_count]
        valid_prediction = predicted_ids[:valid_count]
        correct = sum(a == b for a, b in zip(valid_truth, valid_prediction))
        exact = valid_truth == valid_prediction
        source_text = visible_text(tokenizer, valid_truth.copy(), eos_id, pad_id)
        reconstructed_text = visible_text(tokenizer, valid_prediction.copy(), eos_id, pad_id)
        similarity = difflib.SequenceMatcher(
            a=source_text, b=reconstructed_text, autojunk=False,
        ).ratio()

        total_valid += valid_count
        total_correct += correct
        total_exact += int(exact)
        total_char_similarity += similarity
        output_rows.append({
            "id": record.get("id"),
            "prompt": record.get("prompt", ""),
            "target": record["target"],
            "reconstructed": reconstructed_text,
            "valid_token_count": valid_count,
            "correct_tokens": correct,
            "valid_token_accuracy": correct / max(1, valid_count),
            "token_exact_match": exact,
            "character_similarity": similarity,
        })

    output_path = repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "purpose": "Frozen VQ-VAE round-trip diagnostic on held-out short C# targets.",
        "checkpoint": str(checkpoint_path),
        "input": str(repo_path(args.input)),
        "device": str(device),
        "sequence_length": sequence_length,
        "examples": len(output_rows),
        "valid_tokens": total_valid,
        "valid_token_accuracy": total_correct / max(1, total_valid),
        "exact_sequence_matches": total_exact,
        "exact_sequence_match_rate": total_exact / max(1, len(output_rows)),
        "mean_character_similarity": total_char_similarity / max(1, len(output_rows)),
        "elapsed_seconds": time.perf_counter() - started,
        "reconstructions": str(output_path),
        "interpretation": (
            "This tests only codec reconstruction, not generator quality or code correctness. "
            "Character similarity is descriptive and is not a pass/fail threshold."
        ),
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
