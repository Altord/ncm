"""Live terminal progress view for the sequential 8-level tokenizer screen."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


CANDIDATES = (
    "qwen",
    "starcoder2",
    "granite-code",
    "deepseek-coder-1.3b",
    "deepseek-coder-v2",
)
BATCH_SIZE = 16
PASSES = 3
EVAL_CANDIDATES = CANDIDATES


@dataclass
class Run:
    name: str
    target: int
    step: int
    status: str
    rate: float | None
    seconds_per_step: float | None
    elapsed: float | None
    log_path: Path
    final_path: Path


def count_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def read_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("step"), (int, float)):
                rows.append(row)
    return rows


def estimate_rate(rows: list[dict]) -> tuple[float | None, float | None]:
    timed = [row for row in rows if isinstance(row.get("time_unix"), (int, float))]
    if len(timed) < 2:
        return None, None
    window = timed[-6:]
    first, last = window[0], window[-1]
    step_delta = float(last["step"]) - float(first["step"])
    time_delta = float(last["time_unix"]) - float(first["time_unix"])
    if step_delta <= 0 or time_delta <= 0:
        return None, None
    seconds_per_step = time_delta / step_delta
    return 1.0 / seconds_per_step, seconds_per_step


def inspect(root: Path, name: str, now: float, run_dir: Path | None = None,
            target: int | None = None) -> Run:
    if run_dir is None:
        data = root / "data" / "mycode" / f"godot-csharp-{name}-strict" / "train.jsonl"
        rows_count = count_rows(data)
        target = ((rows_count + BATCH_SIZE - 1) // BATCH_SIZE) * PASSES if rows_count else 0
        run_dir = root / "runs" / "codec-tokenizer-screen" / name / "8level"
    else:
        run_dir = run_dir if run_dir.is_absolute() else root / run_dir
    log_path = run_dir / "logs" / "train.jsonl"
    final_path = run_dir / "artifacts" / "tokenizer.pt"
    rows = read_log(log_path)
    step = int(rows[-1]["step"]) if rows else 0
    rate, seconds_per_step = estimate_rate(rows)
    timed_rows = [row for row in rows if isinstance(row.get("time_unix"), (int, float))]
    elapsed = max(0.0, now - float(timed_rows[0]["time_unix"])) if timed_rows else None

    final_checkpoint = run_dir / "checkpoints" / "final.pt"
    if ((target and step >= target) or
            (final_path.is_file() and (not log_path.exists() or final_path.stat().st_mtime >= log_path.stat().st_mtime)) or
            (final_checkpoint.is_file() and (not log_path.exists() or final_checkpoint.stat().st_mtime >= log_path.stat().st_mtime))):
        status = "done"
        step = target or step
    elif not rows:
        status = "pending"
    elif now - log_path.stat().st_mtime <= 300:
        status = "running"
    else:
        status = "stale log"

    return Run(name, target, step, status, rate, seconds_per_step, elapsed,
               log_path, final_path)


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "estimating"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02}m"
    if minutes:
        return f"{minutes}m {seconds:02}s"
    return f"{seconds}s"


def render(root: Path, custom_runs: list[tuple[str, Path, int]] | None = None) -> str:
    now = time.time()
    runs = ([inspect(root, name, now, path, target)
             for name, path, target in custom_runs]
            if custom_runs else [inspect(root, name, now) for name in CANDIDATES])
    total_target = sum(run.target for run in runs)
    total_done = sum(run.target if run.status == "done" else min(run.step, run.target)
                     for run in runs)
    overall_pct = 100.0 * total_done / total_target if total_target else 0.0

    rates = [run.seconds_per_step for run in runs if run.seconds_per_step is not None]
    fallback_rate = statistics.median(rates) if rates else None
    total_eta = 0.0 if fallback_rate is not None else None
    for run in runs:
        if run.status == "done":
            continue
        sec_per_step = run.seconds_per_step or fallback_rate
        if sec_per_step is None or run.target <= 0:
            total_eta = None
            break
        remaining = max(0, run.target - run.step)
        total_eta += remaining * sec_per_step

    width = 22
    lines = [
        "NCM training progress — live read-only log monitor",
        f"Overall step progress: {total_done:,}/{total_target:,} ({overall_pct:.1f}%)",
        f"Rough time remaining: {format_duration(total_eta)} "
        "(extrapolated from observed candidate speeds)",
        "",
    ]
    for run in runs:
        pct = 100.0 * min(run.step, run.target) / run.target if run.target else 0.0
        filled = round(width * pct / 100.0)
        bar = "#" * filled + "." * (width - filled)
        speed = f"{run.rate:.1f} steps/s" if run.rate else "speed estimating"
        elapsed = format_duration(run.elapsed)
        remaining = max(0, run.target - run.step)
        eta = (remaining * run.seconds_per_step
               if run.seconds_per_step is not None else None)
        lines.append(
            f"{run.name:22} [{bar}] {min(run.step, run.target):,}/{run.target:,} "
            f"{pct:5.1f}%  {run.status:10} elapsed {elapsed:9} {speed:18} ETA {format_duration(eta)}"
        )
    lines += [
        "",
        "Log files are read only. ETA uses recent logged speed and is approximate.",
    ]
    return "\n".join(lines)


def active_codec_evaluation() -> tuple[str, float] | None:
    """Return active evaluator label and elapsed seconds, if it is running."""
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "etime=,args="], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in output.splitlines():
        if "evaluation/godot_csharp_codec_eval.py" not in line:
            continue
        elapsed_text, _, command = line.strip().partition(" ")
        name = next((candidate for candidate in EVAL_CANDIDATES
                     if f"{candidate}-8level.yaml" in command), "unknown candidate")
        stage = "compile stress" if "codec_compile_stress.jsonl" in command else "held-out reconstruction"
        return f"{name} — {stage}", parse_elapsed(elapsed_text)
    return None


def parse_elapsed(value: str) -> float:
    """Parse ps elapsed time values in [[dd-]hh:]mm:ss form."""
    try:
        days = 0
        if "-" in value:
            day_text, value = value.split("-", 1)
            days = int(day_text)
        parts = [int(part) for part in value.split(":")]
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours, (minutes, seconds) = 0, parts
        else:
            return 0.0
        return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)
    except ValueError:
        return 0.0


def render_codec_eval_sweep(root: Path) -> str:
    tasks = []
    for candidate in EVAL_CANDIDATES:
        base = root / "runs" / "codec-tokenizer-screen" / candidate / "8level"
        tasks.append((candidate, "held-out reconstruction",
                      base / "validation" / "reconstruction.report.json"))
        tasks.append((candidate, "compile stress",
                      base / "stress" / "compiled.report.json"))

    finished = []
    durations_by_stage: dict[str, list[float]] = {
        "held-out reconstruction": [], "compile stress": []
    }
    for candidate, stage, report_path in tasks:
        if not report_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            duration = float(report.get("elapsed_seconds", 0.0))
        except (OSError, ValueError, TypeError):
            duration = 0.0
        finished.append((candidate, stage))
        if duration > 0:
            durations_by_stage[stage].append(duration)

    active = active_codec_evaluation()
    completed = set(finished)
    total = len(tasks)
    eta_parts = []
    active_label = "No codec evaluator process detected"
    active_elapsed = None
    if active:
        active_label, active_elapsed = active
        active_candidate, _, active_stage = active_label.partition(" — ")
        stage_key = active_stage
        samples = durations_by_stage.get(stage_key, [])
        average = statistics.median(samples) if samples else None
        if average is not None:
            eta_parts.append(max(0.0, average - active_elapsed))

    for candidate, stage, _ in tasks:
        if (candidate, stage) in completed:
            continue
        if active and active_label == f"{candidate} — {stage}":
            continue
        samples = durations_by_stage[stage]
        if samples:
            eta_parts.append(statistics.median(samples))
        else:
            eta_parts = []
            break

    eta = sum(eta_parts) if eta_parts else None
    pct = 100.0 * len(finished) / total
    lines = [
        "NCM codec evaluation sweep — read-only monitor",
        f"Completed checks: {len(finished)}/{total} ({pct:.1f}%)",
        f"Active: {active_label}",
        f"Active process elapsed: {format_duration(active_elapsed)}",
        f"Rough time remaining: {format_duration(eta)}",
        "",
    ]
    for candidate, stage, report_path in tasks:
        done = (candidate, stage) in completed
        is_active = bool(active and active_label == f"{candidate} — {stage}")
        status = "done" if done else ("running" if is_active else "pending")
        lines.append(f"{candidate:22} {stage:26} {status:8} {report_path.relative_to(root)}")
    lines += [
        "",
        "Progress counts finished evaluation reports; the evaluator does not log per-snippet progress.",
        "ETA is approximate and improves after at least one run of each stage finishes.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="NCM repository root (default: current directory)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument("--codec-eval-sweep", action="store_true",
                        help="Monitor the current five-candidate reconstruction/compile sweep")
    parser.add_argument(
        "--run", action="append", default=[], metavar="NAME=RUN_DIR:TARGET_STEPS",
        help="Monitor any run; repeat for multiple runs. RUN_DIR contains logs/train.jsonl.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    custom_runs = []
    for item in args.run:
        try:
            name, run_spec = item.split("=", 1)
            run_path, target_text = run_spec.rsplit(":", 1)
            target = int(target_text)
            if not name or not run_path or target <= 0:
                raise ValueError
        except ValueError as exc:
            parser.error(f"invalid --run {item!r}; use NAME=RUN_DIR:TARGET_STEPS")
        custom_runs.append((name, Path(run_path), target))

    while True:
        output = (render_codec_eval_sweep(root) if args.codec_eval_sweep
                  else render(root, custom_runs or None))
        if args.once:
            print(output, flush=True)
            return
        if sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(output, flush=True)
        try:
            time.sleep(max(1.0, args.interval))
        except KeyboardInterrupt:
            print("\nMonitor stopped; training was not affected.")
            return


if __name__ == "__main__":
    main()
