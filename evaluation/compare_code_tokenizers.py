"""Compare base tokenizer behavior on code snippets before retraining an NCM codec."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer


IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*(?![A-Za-z0-9_])")


def load_records(path: Path, text_key: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(row.get(text_key), str):
                raise ValueError(f"{path}:{line_no}: missing string field {text_key!r}")
            rows.append(row)
    return rows


def analyze(model_id: str, rows: list[dict], text_key: str, sequence_length: int) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=False,
    )
    original_vocab_size = int(tokenizer.vocab_size)
    added_pad = False
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        added_pad = True

    token_counts = []
    exact_roundtrips = 0
    over_context = 0
    identifier_piece_counts = []
    rows_by_id = []
    token_ids_digest = hashlib.sha256()
    for row in rows:
        text = row[text_key]
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids_digest.update(json.dumps(ids, separators=(",", ":")).encode("utf-8"))
        token_ids_digest.update(b"\n")
        decoded = tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )
        count = len(ids)
        token_counts.append(count)
        exact_roundtrips += int(decoded == text)
        over_context += int(count > sequence_length)
        for match in IDENTIFIER.finditer(text):
            pieces = tokenizer.encode(match.group(0), add_special_tokens=False)
            identifier_piece_counts.append(len(pieces))
        rows_by_id.append({"id": row.get("id"), "token_count": count,
                           "roundtrip_exact": decoded == text})

    ordered_counts = sorted(token_counts)
    median_count = ordered_counts[len(ordered_counts) // 2]
    return {
        "model_id": model_id,
        "fast_tokenizer": bool(tokenizer.is_fast),
        "base_vocab_size": original_vocab_size,
        "vocab_size_after_pad_setup": len(tokenizer),
        "added_pad_token": added_pad,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "sample_token_ids_sha256": token_ids_digest.hexdigest(),
        "examples": len(rows),
        "sequence_length": sequence_length,
        "mean_tokens_per_example": sum(token_counts) / max(1, len(token_counts)),
        "median_tokens_per_example": median_count,
        "max_tokens_per_example": max(token_counts, default=0),
        "examples_over_context": over_context,
        "exact_text_roundtrips": exact_roundtrips,
        "mean_pieces_per_identifier_like_span": (
            sum(identifier_piece_counts) / max(1, len(identifier_piece_counts))
        ),
        "single_piece_identifier_like_spans": sum(
            count == 1 for count in identifier_piece_counts
        ),
        "identifier_like_spans": len(identifier_piece_counts),
        "rows": rows_by_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl",
    )
    parser.add_argument("--text-key", default="target")
    parser.add_argument("--models", default=(
                        "Qwen/Qwen2.5-Coder-0.5B,gpt2,microsoft/codebert-base,"
                        "Salesforce/codet5-base,bigcode/starcoder2-3b,"
                        "deepseek-ai/deepseek-coder-1.3b-base,"
                        "deepseek-ai/DeepSeek-Coder-V2-Base,"
                        "Qwen/Qwen3-Coder-30B-A3B-Instruct,"
                        "ibm-granite/granite-8b-code-base-4k,"
                        "google/codegemma-7b,meta-llama/CodeLlama-7b-hf"),
                        help="Comma-separated Hugging Face tokenizer model IDs.")
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    rows = load_records(input_path, args.text_key)
    results = []
    for raw_model_id in args.models.split(","):
        model_id = raw_model_id.strip()
        if not model_id:
            continue
        try:
            result = analyze(model_id, rows, args.text_key, args.sequence_length)
            result["status"] = "ok"
        except Exception as exc:
            # A candidate tokenizer can be incompatible with the installed
            # Transformers/tokenizers pair. Record it and keep evaluating the
            # remaining candidates instead of losing their measurements.
            result = {
                "model_id": model_id,
                "status": "load_or_analysis_error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
        results.append(result)
    report = {
        "purpose": "Tokenizer-only diagnostic; does not train a codec or compare NCM quality.",
        "input": str(input_path),
        "text_key": args.text_key,
        "models": results,
        "interpretation": (
            "Token counts and identifier-like span splits are proxies only. A shorter or more "
            "code-oriented tokenizer must still be trained and evaluated with a matched NCM codec."
        ),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print("\n=== Tokenizer comparison ===")
    baseline = next((row for row in results if row.get("model_id") == "Qwen/Qwen2.5-Coder-0.5B"
                     and row.get("status") == "ok"), None)
    baseline_digest = baseline.get("sample_token_ids_sha256") if baseline else None
    for result in results:
        if result.get("status") != "ok":
            print(f"{result['model_id']}: unavailable ({result['error_type']}: {result['error']})")
            continue
        print(
            f"{result['model_id']}: mean={result['mean_tokens_per_example']:.1f} tokens, "
            f"over-{args.sequence_length}={result['examples_over_context']}/{result['examples']}, "
            f"exact text round-trip={result['exact_text_roundtrips']}/{result['examples']}, "
            f"identifier-like spans={result['mean_pieces_per_identifier_like_span']:.2f} pieces/span"
        )
        if baseline and result is not baseline:
            base_tokens = baseline["mean_tokens_per_example"]
            token_delta_pct = 100.0 * (result["mean_tokens_per_example"] - base_tokens) / max(base_tokens, 1e-12)
            base_identifiers = baseline["mean_pieces_per_identifier_like_span"]
            identifier_delta_pct = 100.0 * (
                result["mean_pieces_per_identifier_like_span"] - base_identifiers
            ) / max(base_identifiers, 1e-12)
            print(f"  vs Qwen2.5: tokens {token_delta_pct:+.1f}%, "
                  f"identifier pieces/span {identifier_delta_pct:+.1f}%")
            if result.get("sample_token_ids_sha256") == baseline_digest:
                print("  Same token IDs as Qwen2.5 on every comparison sample.")
    if args.output:
        print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
