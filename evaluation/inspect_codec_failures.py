"""Print compact token-level diffs and compiler diagnostics for codec failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from difflib import SequenceMatcher

ROOT = Path(__file__).resolve().parents[1]


def lex(text: str) -> list[str]:
    """Split code into identifier/number/whitespace/punctuation pieces."""
    return re.findall(r"\s+|[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?[fFdDmM]?|.", text)


def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield row


def changed_pieces(original: str, reconstructed: str) -> list[str]:
    expected = lex(original)
    actual = lex(reconstructed)
    output = []
    matcher = SequenceMatcher(a=expected, b=actual, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        before = "".join(expected[i1:i2])
        after = "".join(actual[j1:j2])
        if tag == "insert":
            output.append(f"  inserted: {after!r}")
        elif tag == "delete":
            output.append(f"  deleted:  {before!r}")
        else:
            output.append(f"  changed:  {before!r} -> {after!r}")
    return output


def compiler_errors(result: dict) -> list[str]:
    detail = str(result.get("detail", ""))
    error_lines = [line.strip() for line in detail.splitlines()
                   if " error " in line or "error CS" in line or "Build FAILED" in line]
    return error_lines[-12:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="runs/godot-csharp-ncm-aligned/pilot-test/codec-compiled.jsonl",
        help="JSONL produced by godot_csharp_codec_eval.py --compile",
    )
    parser.add_argument("--all", action="store_true", help="Show passing rows too")
    args = parser.parse_args()
    path = Path(args.input)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        parser.error(f"input file not found: {path}")

    rows = list(read_rows(path))
    failures = 0
    for row in rows:
        original = str(row.get("target", ""))
        reconstructed = str(row.get("reconstructed", ""))
        original_result = row.get("original_compile", {})
        decoded_result = row.get("reconstruction_compile", {})
        passed = decoded_result.get("status") == "passed"
        if passed and not args.all:
            continue
        failures += int(not passed)
        print(f"\n--- {row.get('id', '<no id>')} ---")
        print(
            f"compile: original={original_result.get('status', 'unknown')}, "
            f"reconstruction={decoded_result.get('status', 'unknown')}; "
            f"token_accuracy={row.get('valid_token_accuracy', 'n/a')}; "
            f"exact={row.get('token_exact_match', False)}"
        )
        print("reference:      " + original)
        print("reconstruction: " + reconstructed)
        differences = changed_pieces(original, reconstructed)
        if differences:
            print("changed pieces:")
            print("\n".join(differences))
        else:
            print("changed pieces: none")
        errors = compiler_errors(decoded_result)
        if errors:
            print("compiler diagnostics:")
            print("\n".join(f"  {line}" for line in errors))

    print(f"\nRows: {len(rows)}; reconstruction compile failures shown: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
