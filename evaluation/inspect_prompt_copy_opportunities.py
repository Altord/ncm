"""Measure exact identifier overlap between generator prompts and targets.

This is an oracle coverage diagnostic, not a pointer model or a quality score.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
KEYWORDS = set("abstract as base bool break byte case catch char checked class const continue decimal default delegate do double else enum event explicit extern false finally fixed float for foreach goto if implicit in int interface internal is lock long namespace new null object operator out override params private protected public readonly ref return sbyte sealed short sizeof stackalloc static string struct switch this throw true try typeof uint ulong unchecked unsafe ushort using virtual void volatile while var".split())


def identifiers(text: str) -> list[str]:
    return [m.group() for m in IDENTIFIER.finditer(text) if m.group() not in KEYWORDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("Input has no records")
    total = copied = 0
    details = []
    repeated = Counter()
    for row in rows:
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("target"), str):
            raise ValueError("Every row must contain string prompt and target fields")
        prompt_ids = set(identifiers(row["prompt"]))
        target_ids = identifiers(row["target"])
        matches = [name for name in target_ids if name in prompt_ids]
        total += len(target_ids)
        copied += len(matches)
        repeated.update(matches)
        details.append({"id": row.get("id"), "target_identifier_occurrences": len(target_ids),
                        "copyable_occurrences": len(matches), "copyable_identifiers": sorted(set(matches))})
    report = {"input": str(Path(args.input).resolve()), "examples": len(rows),
              "target_identifier_occurrences": total, "exact_prompt_overlap_occurrences": copied,
              "exact_prompt_overlap_rate": copied / total if total else 0,
              "most_common_overlap": repeated.most_common(20), "rows": details,
              "interpretation": "Case-sensitive lexical upper bound only; includes method and type names, excludes C# keywords, and does not establish a trained copy mechanism."}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Examples: {len(rows)}; target identifier occurrences: {total}")
    print(f"Exact identifiers also present in prompt: {copied}/{total} ({report['exact_prompt_overlap_rate']:.1%})")
    print("This is an oracle overlap estimate, not a trained copy model.")
    print(f"Report: {output.resolve()}")


if __name__ == "__main__":
    main()
