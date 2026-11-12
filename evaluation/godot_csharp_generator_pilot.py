"""Generate candidates for the held-out Godot/C# generator pilot set.

This reports strict normalized reference matches and preserves human acceptance
checks. It does not execute model-generated code or claim semantic correctness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE
from training.data import load_base_tokenizer


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
                raise ValueError(f"{path}:{line_no}: expected an object with a prompt")
            rows.append(row)
    return rows


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def compile_candidate(candidate: str, framework: str, timeout: int = 90):
    """Compile a method snippet without executing the generated code."""
    if shutil.which("dotnet") is None:
        return {"status": "unavailable", "detail": "dotnet SDK not found on PATH"}
    with tempfile.TemporaryDirectory(prefix="ncm-csharp-compile-") as tmp:
        root = Path(tmp)
        (root / "Candidate.csproj").write_text(textwrap.dedent(f"""
            <Project Sdk="Microsoft.NET.Sdk">
              <PropertyGroup>
                <TargetFramework>{framework}</TargetFramework>
                <ImplicitUsings>enable</ImplicitUsings>
                <Nullable>disable</Nullable>
                <OutputType>Library</OutputType>
              </PropertyGroup>
            </Project>
        """).strip() + "\n", encoding="utf-8")
        (root / "Candidate.cs").write_text(textwrap.dedent(f"""
            using System;
            using Godot;

            namespace Godot
            {{
                public readonly struct Vector2
                {{
                    public float X {{ get; }}
                    public float Y {{ get; }}
                    public Vector2(float x, float y) {{ X = x; Y = y; }}
                    public float LengthSquared() => X * X + Y * Y;
                    public Vector2 Normalized()
                    {{
                        var length = MathF.Sqrt(LengthSquared());
                        return length > 0f ? new Vector2(X / length, Y / length) : this;
                    }}
                    public float DistanceSquaredTo(Vector2 other)
                    {{
                        var dx = X - other.X; var dy = Y - other.Y;
                        return dx * dx + dy * dy;
                    }}
                    public Vector2 MoveToward(Vector2 target, float delta)
                    {{
                        var dx = target.X - X; var dy = target.Y - Y;
                        var distance = MathF.Sqrt(dx * dx + dy * dy);
                        if (distance <= delta || distance == 0f) return target;
                        var scale = delta / distance;
                        return new Vector2(X + dx * scale, Y + dy * scale);
                    }}
                }}
            }}

            public static class Candidate
            {{
            {textwrap.indent(candidate, '    ')}
            }}
        """), encoding="utf-8")
        try:
            result = subprocess.run(
                ["dotnet", "build", str(root / "Candidate.csproj"), "--nologo", "--verbosity", "quiet"],
                cwd=root, capture_output=True, text=True, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "detail": f"dotnet build exceeded {timeout}s"}
        output = (result.stdout + "\n" + result.stderr).strip()
        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "exit_code": result.returncode,
            "detail": output[-6000:],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/train/godot-csharp-generator-pilot.yaml",
        help="Generator pilot YAML used to construct the matching architectures.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Generator artifact; defaults to the artifact path in the YAML config.",
    )
    parser.add_argument(
        "--input", default="examples/godot_csharp_generator_pilot/test.jsonl",
        help="Held-out prompt/target JSONL.",
    )
    parser.add_argument(
        "--output", default="runs/godot-csharp/generator-pilot/test/predictions.jsonl",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Zero uses greedy code choice at each generated level.")
    parser.add_argument("--compile", action="store_true",
                        help="Compile candidates in temporary .NET projects; never executes them.")
    parser.add_argument("--dotnet-framework", default="net8.0",
                        help="Target framework used by the optional compile-only check.")
    args = parser.parse_args()

    config_path = repo_path(args.config)
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device requested but torch.cuda.is_available() is false")

    tokenizer = load_base_tokenizer(cfg.tokenizer)
    vq_config = OmegaConf.to_container(cfg.vqvae, resolve=True)
    tokenizer_checkpoint_path = repo_path(vq_config.pop("checkpoint_path"))
    if not tokenizer_checkpoint_path.is_file():
        raise SystemExit(f"Tokenizer checkpoint not found: {tokenizer_checkpoint_path}")
    tokenizer_checkpoint = torch.load(tokenizer_checkpoint_path, map_location="cpu",
                                      weights_only=False)
    # VQVAE's quantizer factory reads nested config entries as attributes.
    vqvae = VQVAE(**OmegaConf.create(vq_config))
    vqvae.load_state_dict(tokenizer_checkpoint["model"], strict=True)
    vqvae.to(device).eval()
    for parameter in vqvae.parameters():
        parameter.requires_grad_(False)

    checkpoint_path = (repo_path(args.checkpoint) if args.checkpoint else
                       repo_path(Path(cfg.train.artifact_dir) / cfg.train.artifact_filename))
    if not checkpoint_path.is_file():
        raise SystemExit(f"Generator checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = NextConceptTransformer(
        vqvae_local=vqvae,
        config=OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    inference = checkpoint.get("inference", {})
    level0_probs = inference.get("level0_unigram")
    if not (model.predict_level0 or model.query_level0):
        if level0_probs is None:
            raise SystemExit("Generator artifact has no inference.level0_unigram bootstrap")
        level0_probs = torch.as_tensor(level0_probs, dtype=torch.float32, device=device)
        level0_probs = level0_probs.clamp_min(0)
        if level0_probs.numel() != int(vqvae.vq.vqvae_vocab_sizes[0]) or level0_probs.sum() <= 0:
            raise SystemExit("Generator artifact has an invalid level-0 unigram")
        level0_probs = level0_probs / level0_probs.sum()

    records = read_jsonl(repo_path(args.input))
    prefix_length = int(cfg.model.prefix_len)
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise SystemExit("Base tokenizer must define both PAD and EOS token IDs")
    pad_id = int(tokenizer.pad_token_id)
    eos_id = int(tokenizer.eos_token_id)

    output_rows = []
    started = time.perf_counter()
    for index, record in enumerate(records):
        prompt_ids = list(map(int, tokenizer.encode(record["prompt"])))
        original_prompt_length = len(prompt_ids)
        prefix = prompt_ids[:prefix_length]
        if not prefix:
            prefix = [eos_id]
        prefix += [pad_id] * (prefix_length - len(prefix))
        prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)

        seed = args.seed + index
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        level0_length = int(vqvae.vq.levels[0])
        level0 = (None if model.predict_level0 or model.query_level0 else
                  torch.multinomial(level0_probs, level0_length, replacement=True).view(1, -1))

        with torch.inference_mode():
            codes = model.generate(
                level0, prefix_tensor,
                temperature=float(args.temperature), top_k=1, top_p=1.0,
            )
            _, z_levels = vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
            token_logits = vqvae.decoder(z_levels[-1])
            generated_ids = token_logits.argmax(dim=-1)[0].tolist()

        if eos_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(eos_id)]
        generated_ids = [token_id for token_id in generated_ids if token_id != pad_id]
        candidate = tokenizer.decode(generated_ids, skip_special_tokens=True)
        reference = record.get("target", "")
        exact_match = bool(reference) and normalized(candidate) == normalized(reference)
        row = {
            "id": record.get("id", f"example-{index}"),
            "prompt": record["prompt"],
            "prompt_tokens": original_prompt_length,
            "prompt_tokens_used": min(original_prompt_length, prefix_length),
            "candidate": candidate,
            "reference": reference,
            "normalized_reference_exact_match": exact_match,
            "acceptance": record.get("acceptance", []),
            "cases": record.get("cases", []),
            "acceptance_execution": "not_run; generated code is not executed by this script",
        }
        if args.compile:
            row["compile_check"] = (
                compile_candidate(candidate, args.dotnet_framework)
                if candidate.strip() else {"status": "empty", "detail": "model emitted no code"}
            )
        output_rows.append(row)
        print(f"Generated {len(output_rows)}/{len(records)} examples; "
              f"elapsed {time.perf_counter() - started:.1f}s", flush=True)

    output_path = repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    exact = sum(row["normalized_reference_exact_match"] for row in output_rows)
    compile_rows = [row["compile_check"] for row in output_rows if "compile_check" in row]
    report = {
        "task": str(cfg.task),
        "input": str(repo_path(args.input)),
        "generator_checkpoint": str(checkpoint_path),
        "tokenizer_checkpoint": str(tokenizer_checkpoint_path),
        "device": str(device),
        "seed": args.seed,
        "temperature": args.temperature,
        "prompt_prefix_length": prefix_length,
        "output_count": len(output_rows),
        "normalized_reference_exact_matches": exact,
        "normalized_reference_exact_match_rate": exact / max(1, len(output_rows)),
        "compile_checks": len(compile_rows),
        "compile_passes": sum(row["status"] == "passed" for row in compile_rows),
        "compile_failures": sum(row["status"] == "failed" for row in compile_rows),
        "compile_empty_candidates": sum(row["status"] == "empty" for row in compile_rows),
        "compile_unavailable_or_timeout": sum(row["status"] in {"unavailable", "timeout"}
                                               for row in compile_rows),
        "acceptance_cases_executed": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "predictions": str(output_path),
        "interpretation": (
            "Reference exact-match is a strict diagnostic, not semantic code correctness. "
            "The optional .NET check compiles only; acceptance cases remain unexecuted."
        ),
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
