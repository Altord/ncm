"""Print a compact reconstruction/compile summary for the 8-level codec screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CANDIDATES = (
    "qwen", "starcoder2", "granite-code", "deepseek-coder-1.3b", "deepseek-coder-v2"
)


def load_report(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def pct(value: object) -> str:
    return f"{float(value) * 100:.2f}%" if isinstance(value, (int, float)) else "—"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="NCM repository root (default: current directory)")
    args = parser.parse_args()
    root = args.root.resolve()

    print("NCM 8-level codec screen — concise results for sharing")
    print("candidate             val-token  val-exact  val-char   stress-char  compile")
    print("--------------------  ---------  ---------  ---------  -----------  -------")
    for candidate in CANDIDATES:
        base = root / "runs" / "codec-tokenizer-screen" / candidate / "8level"
        val = load_report(base / "validation" / "reconstruction.report.json")
        stress = load_report(base / "stress" / "compiled.report.json")
        if val is None and stress is None:
            print(f"{candidate:20}  pending (no evaluation reports found)")
            continue
        val_token = pct(val.get("valid_token_accuracy")) if val else "—"
        val_exact = pct(val.get("exact_sequence_match_rate")) if val else "—"
        val_char = pct(val.get("mean_character_similarity")) if val else "—"
        stress_char = pct(stress.get("mean_character_similarity")) if stress else "—"
        compile_result = "—"
        if stress and stress.get("compile_requested"):
            compile_result = (
                f"{stress.get('reconstruction_compile_passes', 0)}/"
                f"{stress.get('examples', 0)}"
            )
        print(f"{candidate:20}  {val_token:9}  {val_exact:9}  {val_char:9}  "
              f"{stress_char:11}  {compile_result}")

    print("\nCompile denominator is the shared stress-set size; originals should all pass.")
    print("Paste this summary plus any errors or warnings that changed the run.")


if __name__ == "__main__":
    main()
