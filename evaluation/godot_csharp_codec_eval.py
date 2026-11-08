"""Measure frozen VQ-VAE round-trip quality on held-out Godot/C# snippets."""

from __future__ import annotations

import argparse
from collections import defaultdict
import difflib
import hashlib
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
from evaluation.godot_csharp_generator_pilot import compile_candidate


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def visible_text(tokenizer, ids: list[int], eos_id: int, pad_id: int) -> str:
    if eos_id in ids:
        ids = ids[:ids.index(eos_id)]
    ids = [token_id for token_id in ids if token_id != pad_id]
    return tokenizer.decode(ids, skip_special_tokens=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument("--text-key", default="target",
                        help="JSONL string field to reconstruct (for example target or text).")
    parser.add_argument("--code-only", action="store_true",
                        help="Keep only rows whose source has a C# code extension.")
    parser.add_argument("--code-extensions", default=".cs,.csproj",
                        help="Comma-separated extensions used by --code-only.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional maximum number of matching records (0 means all).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--compile", action="store_true",
                        help="Compile original and reconstructed snippets; never executes them.")
    parser.add_argument(
        "--compare-report", default="",
        help="Optional prior .report.json from the same input for a concise before/after comparison.",
    )
    parser.add_argument("--dotnet-framework", default="net8.0")
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
    extension_stats = defaultdict(lambda: {"examples": 0, "valid_tokens": 0,
                                           "correct_tokens": 0, "exact_sequences": 0})
    started = time.perf_counter()

    accepted_extensions = {ext.strip().lower() for ext in args.code_extensions.split(",") if ext.strip()}
    records = []
    input_path = repo_path(args.input)
    input_digest = file_sha256(input_path)
    for record in iter_records(input_path, "jsonl"):
        text = record.get(args.text_key)
        if not isinstance(text, str):
            raise ValueError(f"Record {record.get('id', '<unknown>')} has no string field {args.text_key!r}")
        source_path = str(record.get("source", ""))
        extension = Path(source_path).suffix.lower() if source_path else "no-source"
        if args.code_only and extension not in accepted_extensions:
            continue
        records.append((record, text, extension))
        if args.limit > 0 and len(records) >= args.limit:
            break
    if not records:
        raise SystemExit("No matching records found; check --text-key and --code-only filters")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least one")

    for begin in range(0, len(records), args.batch_size):
        batch = records[begin:begin + args.batch_size]
        padded_batch, masks = [], []
        for _, text, _ in batch:
            ids = encode_text(tokenizer, text, append_eos=True)
            padded, mask = fixed_sequence(ids, sequence_length, pad_id, eos_id)
            padded_batch.append(padded)
            masks.append(mask)
        tokens = torch.tensor(padded_batch, dtype=torch.long, device=device)

        with torch.inference_mode():
            codes = model.encode(tokens)
            logits = model.decode(codes)
            predictions = logits.argmax(dim=-1).tolist()

        for (record, text, extension), padded_ids, mask, predicted_ids in zip(
                batch, padded_batch, masks, predictions):
            valid_count = sum(mask)
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
            group = extension_stats[extension]
            group["examples"] += 1
            group["valid_tokens"] += valid_count
            group["correct_tokens"] += correct
            group["exact_sequences"] += int(exact)
            result_row = {
                "id": record.get("id"),
                "source": record.get("source"),
                "prompt": record.get("prompt", ""),
                "text_key": args.text_key,
                "target": text,
                "reconstructed": reconstructed_text,
                "valid_token_count": valid_count,
                "correct_tokens": correct,
                "valid_token_accuracy": correct / max(1, valid_count),
                "token_exact_match": exact,
                "character_similarity": similarity,
            }
            if args.compile:
                result_row["original_compile"] = compile_candidate(text, args.dotnet_framework)
                result_row["reconstruction_compile"] = compile_candidate(
                    reconstructed_text, args.dotnet_framework,
                ) if reconstructed_text.strip() else {
                    "status": "empty", "detail": "codec reconstructed no code",
                }
            output_rows.append(result_row)

    output_path = repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "purpose": "Frozen VQ-VAE round-trip diagnostic; measures representation only.",
        "checkpoint": str(checkpoint_path),
        "input": str(input_path),
        "input_sha256": input_digest,
        "text_key": args.text_key,
        "code_only": args.code_only,
        "code_extensions": sorted(accepted_extensions),
        "device": str(device),
        "sequence_length": sequence_length,
        "examples": len(output_rows),
        "valid_tokens": total_valid,
        "valid_token_accuracy": total_correct / max(1, total_valid),
        "exact_sequence_matches": total_exact,
        "exact_sequence_match_rate": total_exact / max(1, len(output_rows)),
        "mean_character_similarity": total_char_similarity / max(1, len(output_rows)),
        "compile_requested": args.compile,
        "original_compile_passes": sum(
            row.get("original_compile", {}).get("status") == "passed" for row in output_rows
        ),
        "reconstruction_compile_passes": sum(
            row.get("reconstruction_compile", {}).get("status") == "passed" for row in output_rows
        ),
        "compile_unavailable_rows": sum(
            row.get("original_compile", {}).get("status") == "unavailable" or
            row.get("reconstruction_compile", {}).get("status") == "unavailable"
            for row in output_rows
        ),
        "by_extension": {
            ext: {
                **stats,
                "valid_token_accuracy": stats["correct_tokens"] / max(1, stats["valid_tokens"]),
                "exact_sequence_match_rate": stats["exact_sequences"] / max(1, stats["examples"]),
            }
            for ext, stats in sorted(extension_stats.items())
        },
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
    print("\n=== Human-readable codec summary ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Examples:   {report['examples']} ({report['input']})")

    previous = None
    if args.compare_report:
        compare_path = repo_path(args.compare_report)
        if not compare_path.is_file():
            raise SystemExit(f"Comparison report not found: {compare_path}")
        candidate_previous = json.loads(compare_path.read_text(encoding="utf-8"))
        matching_fields = ("input_sha256", "text_key", "code_only", "sequence_length", "code_extensions")
        mismatches = [
            field for field in matching_fields
            if field not in candidate_previous or candidate_previous[field] != report.get(field)
        ]
        if mismatches:
            print(
                "Comparison skipped: the reports do not prove they use the same input/config "
                f"({', '.join(mismatches)} differ or are missing). Prior input: "
                f"{candidate_previous.get('input', '<unknown>')}"
            )
        else:
            previous = candidate_previous

    def show_metric(label, key, suffix="%", digits=2, scale=100):
        current = float(report[key]) * scale
        line = f"{label}: {current:.{digits}f}{suffix}"
        if previous is not None and key in previous:
            old = float(previous[key]) * scale
            delta = current - old
            direction = "improved" if delta > 1e-9 else "declined" if delta < -1e-9 else "unchanged"
            delta_unit = " pp" if suffix == "%" else suffix
            line += f" (vs prior {old:.{digits}f}{suffix}: {direction} {delta:+.{digits}f}{delta_unit})"
        print(line)

    show_metric("Token accuracy", "valid_token_accuracy")
    exact = f"{report['exact_sequence_matches']}/{report['examples']}"
    exact_line = f"Exact 64-token chunks: {exact} ({report['exact_sequence_match_rate'] * 100:.2f}%)"
    if previous is not None and "exact_sequence_match_rate" in previous:
        old = float(previous["exact_sequence_match_rate"]) * 100
        delta = report["exact_sequence_match_rate"] * 100 - old
        direction = "improved" if delta > 1e-9 else "declined" if delta < -1e-9 else "unchanged"
        exact_line += f" (vs prior {old:.2f}%: {direction} {delta:+.2f} pp)"
    print(exact_line)
    show_metric("Mean character similarity", "mean_character_similarity")
    if args.compile:
        print(
            f"Compile: originals {report['original_compile_passes']}/{report['examples']}; "
            f"reconstructions {report['reconstruction_compile_passes']}/{report['examples']}"
        )
        failures = report["examples"] - report["reconstruction_compile_passes"]
        if failures:
            print(f"Still needs work: {failures} reconstructed snippets did not compile.")
        else:
            print("Compile gate: every reconstructed pilot snippet compiled.")
    else:
        print("Compile: not run; use --compile for the pilot syntax check.")
    print(f"Detailed rows: {output_path}")
    print(f"Full report:   {report_path}", flush=True)


if __name__ == "__main__":
    main()
