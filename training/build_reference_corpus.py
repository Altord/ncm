"""Build license-traceable, short-sequence text data from synced game references.

This produces tokenizer-reconstruction data only. It does not create prompt/solution
examples for training the Next Concept generator.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import random

from transformers import AutoTokenizer

from training.csharp_structure import byte_to_char_offsets, syntax_boundary_bytes


CODE_SUFFIXES = {".cs", ".csproj", ".tscn", ".tres", ".godot", ".gdshader"}
DOC_SUFFIXES = {".md", ".rst"}
SKIP_PARTS = {".git", "bin", "obj", ".godot", "node_modules", "_build"}


def is_validation(path: Path, root: Path, fraction: float) -> bool:
    relative = path.relative_to(root).as_posix().encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(relative).digest()[:8], "big") % 10_000
    return bucket < int(fraction * 10_000)


def chunk_file(path: Path, tokenizer, max_tokens: int, ast_aware: bool = False):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    if not text.strip():
        return
    ids = tokenizer.encode(text, add_special_tokens=False)
    offsets = None
    boundaries = set()
    if ast_aware and path.suffix.lower() == ".cs":
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoded["offset_mapping"]
            byte_to_char = byte_to_char_offsets(text)
            char_boundaries = {
                byte_to_char[min(pos, len(byte_to_char) - 1)]
                for pos in syntax_boundary_bytes(text)
            }
            boundaries = {
                index + 1 for index, (_, end_char) in enumerate(offsets)
                if end_char in char_boundaries
            }
        except (RuntimeError, ValueError, TypeError):
            # Keep corpus creation robust for malformed/unparseable source files.
            boundaries = set()
    boundaries = sorted(boundaries)
    cursor = 0
    while cursor < len(ids):
        limit = min(cursor + max_tokens, len(ids))
        boundary_index = bisect_right(boundaries, limit) - 1
        end = boundaries[boundary_index] if boundary_index >= 0 and boundaries[boundary_index] > cursor else limit
        while end > cursor:
            piece = tokenizer.decode(ids[cursor:end], clean_up_tokenization_spaces=False)
            # prepare_data appends EOS; retain one token of headroom for that marker.
            if len(tokenizer.encode(piece, add_special_tokens=False)) < max_tokens + 1:
                break
            end -= 1
        if end <= cursor:
            cursor += 1
            continue
        if ast_aware or len(tokenizer.encode(piece, add_special_tokens=False)) >= 8:
            yield piece
        cursor = end


def source_license(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    repo = parts[0] if parts else ""
    suffix = path.suffix.lower()
    if repo == "godot-docs":
        return "CC-BY-3.0; see godot-docs/LICENSE.txt"
    if repo == "dotnet-docs":
        return ("MIT; see dotnet/docs/LICENSE-CODE" if suffix in CODE_SUFFIXES
                else "CC-BY-4.0; see dotnet/docs/LICENSE")
    if repo in {"godot-demo-projects", "godot-csharp-essentials"}:
        return "MIT; see source repository LICENSE file"
    if repo == "google-eng-practices":
        return "CC-BY-3.0; see google/eng-practices LICENSE"
    if repo == "game-design-concepts":
        return "CC-BY-3.0; see game-design-concepts/ATTRIBUTION.md"
    return "Unknown; review source license before using this corpus"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/reference"))
    ap.add_argument("--output", type=Path, default=Path("data/mycode/godot-csharp-reference"))
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--sequence-length", type=int, default=64)
    ap.add_argument("--validation-fraction", type=float, default=0.1)
    ap.add_argument("--max-doc-fraction", type=float, default=0.25,
                     help="Cap prose chunks to this fraction of each split; remaining data is code/config.")
    ap.add_argument(
        "--include-extensions", default="",
        help="Optional comma-separated file extensions to include (for example .cs,.csproj).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunking", choices=("token", "ast"), default="token",
                     help="Use flat fixed token windows or C# syntax-boundary windows.")
    args = ap.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Reference directory not found: {root}; run the reference sync script first")
    if args.sequence_length < 16 or not 0 < args.validation_fraction < 0.5:
        raise SystemExit("Use sequence_length >= 16 and validation_fraction between 0 and 0.5")
    if not 0 <= args.max_doc_fraction < 1:
        raise SystemExit("max-doc-fraction must be in [0, 1)")
    include_extensions = {
        ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
        for ext in args.include_extensions.split(",") if ext.strip()
    }
    supported_extensions = CODE_SUFFIXES | DOC_SUFFIXES
    unknown_extensions = include_extensions - supported_extensions
    if unknown_extensions:
        raise SystemExit(
            "Unsupported include extensions: " + ", ".join(sorted(unknown_extensions))
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    max_tokens = args.sequence_length - 2
    buckets = {split: {"code": [], "docs": []} for split in ("train", "validation")}
    file_counts = {split: {"code": 0, "docs": 0} for split in buckets}
    seen_hashes = set()

    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.name.lower().startswith("license") or path.suffix.lower() not in supported_extensions:
            continue
        if include_extensions and path.suffix.lower() not in include_extensions:
            continue
        try:
            if path.stat().st_size > 2_000_000:
                continue
            raw = path.read_bytes()
        except OSError:
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        category = "code" if path.suffix.lower() in CODE_SUFFIXES else "docs"
        split = "validation" if is_validation(path, root, args.validation_fraction) else "train"
        chunks = list(chunk_file(path, tokenizer, max_tokens,
                                 ast_aware=args.chunking == "ast"))
        if chunks:
            relative = path.relative_to(root).as_posix()
            license_name = source_license(path, root)
            buckets[split][category].extend(
                {"text": chunk, "source": relative, "license": license_name}
                for chunk in chunks
            )
            file_counts[split][category] += 1

    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, groups in buckets.items():
        code = groups["code"]
        docs = groups["docs"]
        # Keep documentation as a minority so the tokenizer sees lots of actual syntax.
        max_docs = int(len(code) * args.max_doc_fraction / max(1 - args.max_doc_fraction, 1e-9))
        if len(docs) > max_docs:
            docs = rng.sample(docs, max_docs)
        rows = code + docs
        rng.shuffle(rows)
        if split == "train" and not code:
            raise SystemExit("No code samples found; check synced sources and file filters")
        if split == "validation" and not rows:
            raise SystemExit("No validation samples found; lower validation-fraction or add sources")
        out_path = args.output / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = {"code_chunks": len(code), "doc_chunks": len(docs), "total": len(rows)}

    manifest = {
        "purpose": "VQ-VAE tokenizer reconstruction only; not instruction-target generator data",
        "tokenizer": args.tokenizer,
        "sequence_length": args.sequence_length,
        "chunking": args.chunking,
        "source_manifest": str((root / "SOURCES.tsv").resolve()),
        "file_counts": file_counts,
        "chunk_counts": counts,
        "deduplicated_files": len(seen_hashes),
        "include_extensions": sorted(include_extensions) if include_extensions else None,
        "split_rule": "stable SHA-256 of relative file path; no chunks from one file cross splits",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
