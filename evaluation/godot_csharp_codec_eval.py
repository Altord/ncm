"""Measure frozen VQ-VAE round-trip quality on held-out Godot/C# snippets."""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import difflib
import hashlib
import json
import math
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


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint value must be nonnegative")
    output = bytearray()
    while value >= 128:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def _read_varint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(payload):
        byte = payload[offset]
        offset += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("residual varint is too long")
    raise ValueError("truncated residual varint")


def exact_residual(truth: list[int], prediction: list[int]) -> bytes:
    """Encode count, then position deltas and original token IDs for mismatches."""
    if len(truth) != len(prediction):
        raise ValueError("truth and prediction lengths differ")
    errors = [(position, token_id) for position, (token_id, guess) in
              enumerate(zip(truth, prediction)) if token_id != guess]
    payload = bytearray(_varint(len(errors)))
    previous = -1
    for position, token_id in errors:
        payload.extend(_varint(position - previous))
        payload.extend(_varint(token_id))
        previous = position
    return bytes(payload)


def apply_exact_residual(prediction: list[int], payload: bytes) -> list[int]:
    corrected = prediction.copy()
    count, offset = _read_varint(payload, 0)
    previous = -1
    for _ in range(count):
        delta, offset = _read_varint(payload, offset)
        token_id, offset = _read_varint(payload, offset)
        position = previous + delta
        if delta < 1 or position >= len(corrected):
            raise ValueError("invalid residual position")
        corrected[position] = token_id
        previous = position
    if offset != len(payload):
        raise ValueError("trailing bytes in residual")
    return corrected


def pack_codes(values: list[int], widths: list[int]) -> bytes:
    if len(values) != len(widths):
        raise ValueError("code count does not match level configuration")
    out = bytearray()
    buffer = bits = 0
    for value, width in zip(values, widths):
        if value < 0 or value >= 1 << width:
            raise ValueError("code index is out of range")
        buffer |= value << bits
        bits += width
        while bits >= 8:
            out.append(buffer & 255)
            buffer >>= 8
            bits -= 8
    if bits:
        out.append(buffer)
    return bytes(out)


def unpack_codes(payload: bytes, widths: list[int]) -> list[int]:
    result = []
    buffer = bits = offset = 0
    for width in widths:
        while bits < width:
            if offset >= len(payload):
                raise ValueError("truncated code payload")
            buffer |= payload[offset] << bits
            offset += 1
            bits += 8
        result.append(buffer & ((1 << width) - 1))
        buffer >>= width
        bits -= width
    if offset != len(payload) or buffer:
        raise ValueError("unexpected code payload padding")
    return result


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
    parser.add_argument("--progress", action="store_true",
                        help="Print progress and an estimated remaining time after each example.")
    parser.add_argument("--exact-residual", action="store_true",
                        help="Measure a lossless token correction sidecar for known input text.")
    parser.add_argument("--measure-packed", action="store_true",
                        help="Pack VQ codes, verify unpacking and measure decode time; requires --exact-residual.")
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
    if args.measure_packed and not args.exact_residual:
        parser.error("--measure-packed requires --exact-residual")

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
    configured_checkpoint = vq_config.pop("checkpoint_path", None)
    checkpoint_value = args.checkpoint or configured_checkpoint
    if not checkpoint_value:
        raise SystemExit(
            "No tokenizer checkpoint was specified; pass --checkpoint or use a config "
            "with vqvae.checkpoint_path."
        )
    checkpoint_path = repo_path(checkpoint_value)
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
    residual_bytes = 0
    residual_token_errors = 0
    residual_exact_texts = 0
    residual_compile_passes = 0
    input_bytes = 0
    truncated_inputs = 0
    packed_bytes = framing_bytes = 0
    decode_seconds = residual_apply_seconds = 0.0
    widths = [max(1, math.ceil(math.log2(int(vocab))))
              for length, vocab in zip(model.vq.levels, model.vq.vqvae_vocab_sizes)
              for _ in range(int(length))] if args.measure_packed else []
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
            if args.measure_packed:
                packed = [pack_codes(row, widths) for row in codes.cpu().tolist()]
                unpacked = [unpack_codes(row, widths) for row in packed]
                if unpacked != codes.cpu().tolist():
                    raise AssertionError("packed VQ codes failed to round-trip")
                codes = torch.tensor(unpacked, dtype=codes.dtype, device=device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                decode_started = time.perf_counter()
            logits = model.decode(codes)
            predictions = logits.argmax(dim=-1).tolist()
            if args.measure_packed:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                decode_seconds += time.perf_counter() - decode_started

        for row_index, ((record, text, extension), padded_ids, mask, predicted_ids) in enumerate(zip(
                batch, padded_batch, masks, predictions)):
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
            if args.exact_residual:
                payload = exact_residual(valid_truth, valid_prediction)
                apply_started = time.perf_counter()
                corrected_ids = apply_exact_residual(valid_prediction, payload)
                if args.measure_packed:
                    residual_apply_seconds += time.perf_counter() - apply_started
                    code_bytes = len(packed[row_index])
                    header_bytes = len(_varint(valid_count)) + len(_varint(len(payload)))
                    packed_bytes += code_bytes
                    framing_bytes += header_bytes
                    result_row["packed_measurement"] = {
                        "vq_code_bytes": code_bytes,
                        "framing_bytes": header_bytes,
                        "total_bytes_with_residual": code_bytes + header_bytes + len(payload),
                    }
                corrected_text = visible_text(tokenizer, corrected_ids, eos_id, pad_id)
                if corrected_ids != valid_truth:
                    raise AssertionError("residual failed to restore original token IDs")
                error_count = valid_count - correct
                encoded_bytes = len(text.encode("utf-8"))
                residual_bytes += len(payload)
                residual_token_errors += error_count
                input_bytes += encoded_bytes
                residual_exact_texts += corrected_text == text
                truncated_inputs += source_text != text
                result_row["exact_residual"] = {
                    "payload_base64": base64.b64encode(payload).decode("ascii"),
                    "binary_bytes": len(payload),
                    "changed_tokens": error_count,
                    "restored_text_exact": corrected_text == text,
                    "input_utf8_bytes": encoded_bytes,
                }
            if args.compile:
                result_row["original_compile"] = compile_candidate(text, args.dotnet_framework)
                result_row["reconstruction_compile"] = compile_candidate(
                    reconstructed_text, args.dotnet_framework,
                ) if reconstructed_text.strip() else {
                    "status": "empty", "detail": "codec reconstructed no code",
                }
                if args.exact_residual:
                    result_row["residual_compile"] = compile_candidate(
                        corrected_text, args.dotnet_framework,
                    ) if corrected_text.strip() else {
                        "status": "empty", "detail": "residual restored no code",
                    }
                    residual_compile_passes += (
                        result_row["residual_compile"]["status"] == "passed"
                    )
            output_rows.append(result_row)
            if args.progress:
                done = len(output_rows)
                elapsed = time.perf_counter() - started
                remaining = elapsed / done * (len(records) - done)
                print(f"Progress: {done}/{len(records)} ({100 * done / len(records):.1f}%); "
                      f"elapsed {elapsed:.1f}s; estimated remaining {remaining:.1f}s", flush=True)

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
        "exact_residual": ({
            "scope": "known-input round trip only; generator outputs have no source-token sidecar",
            "binary_bytes": residual_bytes,
            "input_utf8_bytes": input_bytes,
            "residual_to_input_byte_ratio": residual_bytes / max(1, input_bytes),
            "changed_tokens": residual_token_errors,
            "restored_text_exact": residual_exact_texts,
            "truncated_inputs": truncated_inputs,
            "restored_compile_passes": residual_compile_passes if args.compile else None,
            "format": "varint error count, then varint position deltas and original token IDs",
        } if args.exact_residual else None),
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
    if args.measure_packed:
        total = packed_bytes + residual_bytes + framing_bytes
        report["packed_measurement"] = {
            "vq_code_bytes": packed_bytes,
            "residual_bytes": residual_bytes,
            "framing_bytes": framing_bytes,
            "total_bytes": total,
            "input_utf8_bytes": input_bytes,
            "total_to_input_ratio": total / max(1, input_bytes),
            "decode_and_argmax_seconds": decode_seconds,
            "residual_apply_seconds": residual_apply_seconds,
            "decode_batch_size": args.batch_size,
            "timing_scope": "VQ decode and argmax with GPU synchronization, excluding code unpacking, text decoding, encoding and compilation; residual application measured separately on CPU",
            "size_scope": "fixed-width bit-packed code indices plus per-record valid length, residual length and residual; excludes shared model/config and container metadata",
        }
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
    if args.exact_residual:
        residual = report["exact_residual"]
        print(
            f"Exact residual: {residual['changed_tokens']} token corrections, "
            f"{residual['binary_bytes']} binary bytes "
            f"({residual['residual_to_input_byte_ratio']:.1%} of input UTF-8 bytes)"
        )
        print(f"Restored text: {residual['restored_text_exact']}/{report['examples']}")
        if args.compile:
            print(f"Restored compile: {residual_compile_passes}/{report['examples']}")
        if truncated_inputs:
            print(f"Truncated inputs: {truncated_inputs}; residual cannot recover omitted text")
        print("This is an exact correction sidecar for known inputs, not a generation result.")
    if args.measure_packed:
        measure = report["packed_measurement"]
        print(f"Packed VQ codes: {packed_bytes} bytes; residual: {residual_bytes} bytes; framing: {framing_bytes} bytes")
        print(f"Total encoded: {measure['total_bytes']} bytes / {input_bytes} original UTF-8 bytes ({measure['total_to_input_ratio']:.2f}x)")
        print(f"VQ decode + argmax: {decode_seconds:.3f}s; CPU residual apply: {residual_apply_seconds:.3f}s")
        print("Shared model/config and container metadata are excluded; timing excludes code unpacking and text decoding.")
    print(f"Detailed rows: {output_path}")
    print(f"Full report:   {report_path}", flush=True)


if __name__ == "__main__":
    main()
