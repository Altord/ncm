"""Bounded compiler-guided repair using identifiers in reconstructed signatures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.godot_csharp_generator_pilot import compile_candidate


def proposals(text, detail):
    """Propose local edits using only reconstructed text and compiler diagnostics.

    Signature extraction is deliberately conservative; it is not a C# parser.
    No source or target text is used to create or order candidates.
    """
    match = re.search(r'\b[A-Za-z_]\w*\s*\(([^()]*)\)', text)
    if not match:
        return []
    names = []
    for part in match.group(1).split(','):
        m = re.fullmatch(r'\s*(?:(?:ref|out|in|params)\s+)?[A-Za-z_]\w*(?:\[\])?\s+([A-Za-z_]\w*)\s*', part)
        if m and m.group(1) not in names:
            names.append(m.group(1))
    candidates = []
    # A declaration keyword in the body often indicates damaged punctuation.
    body = text[match.end():]
    for found in re.finditer(r'\bclass\b', body):
        start = match.end() + found.start()
        candidates.append((text[:start] + ';' + text[start + 5:], 'replace body class with semicolon'))
    unknown = list(dict.fromkeys(re.findall(r"The name '([^']+)' does not exist in the current context", detail)))
    for name in unknown:
        if not re.fullmatch(r'[A-Za-z_]\w*', name) or name in names:
            continue
        for replacement in names:
            changed = re.sub(r'\b' + re.escape(name) + r'\b', replacement, text)
            if changed != text:
                candidates.append((changed, f'replace undeclared {name} with parameter {replacement}'))
    return candidates


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    result = {r['id']: r for r in rows}
    if not rows or len(rows) != len(result):
        raise ValueError(f'Empty input or duplicate IDs: {path}')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--previous', required=True)
    p.add_argument('--candidate', action='append', default=[])
    p.add_argument('--output', required=True)
    p.add_argument('--max-attempts', type=int, default=24)
    p.add_argument('--framework', default='net8.0')
    args = p.parse_args()
    if args.max_attempts < 1:
        p.error('--max-attempts must be positive')
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists():
        raise SystemExit('Output exists; choose a new path')
    base, prior = load(args.baseline), load(args.previous)
    others = [load(path) for path in args.candidate]
    if set(base) != set(prior) or any(set(x) != set(base) for x in others):
        raise SystemExit('Inputs have different IDs')
    output, total, improved = [], 0, 0
    start = time.perf_counter()
    for i, (key, row) in enumerate(base.items(), 1):
        old = prior[key]
        if old['baseline'] != row['reconstructed']:
            raise SystemExit(f'Prior baseline mismatch: {key}')
        status = old['compile_status']
        if status not in ('passed', 'failed', 'empty'):
            raise SystemExit(f'Prior compile result unavailable: {key}')
        selected, chosen, attempts = old['selected'], old['selection'], 0
        if status != 'passed':
            queue = []
            for source in [base] + others:
                r = source[key]
                if r['target'] != row['target']:
                    raise SystemExit(f'Target mismatch: {key}')
                comp = r['reconstruction_compile']
                if comp['status'] not in ('passed', 'failed', 'empty'):
                    raise SystemExit(f'Compile result unavailable: {key}')
                if comp['status'] == 'passed':
                    selected, status, chosen = r['reconstructed'], 'passed', 'saved_candidate'
                    break
                queue.append((r['reconstructed'], comp.get('detail', ''), 0))
            seen = {item[0] for item in queue}
            while status != 'passed' and queue and attempts < args.max_attempts:
                text, detail, depth = queue.pop(0)
                if depth >= 2:
                    continue
                for candidate, why in proposals(text, detail):
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    result = compile_candidate(candidate, args.framework)
                    attempts += 1
                    total += 1
                    if result['status'] not in ('passed', 'failed', 'empty'):
                        raise SystemExit(f'Compiler unavailable or timed out: {key}: {result}')
                    if result['status'] == 'passed':
                        selected, status, chosen = candidate, 'passed', why
                        break
                    if depth == 0:
                        queue.insert(0, (candidate, result.get('detail', ''), 1))
                    if attempts >= args.max_attempts:
                        break
            if status == 'passed':
                improved += 1
        output.append({'id': key, 'baseline': row['reconstructed'], 'selected': selected,
                       'selection': chosen, 'compile_status': status, 'attempts': attempts,
                       'selected_text_exact': selected == row['target']})
        print(f'Processed {i}/{len(base)}; newly compiling={improved}; attempts={total}', flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in output:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = {'examples': len(output), 'prior_compiles': sum(r['compile_status'] == 'passed' for r in prior.values()),
              'selected_compiles': sum(r['compile_status'] == 'passed' for r in output),
              'selected_exact_text': sum(r['selected_text_exact'] for r in output),
              'newly_compiling': improved, 'compiler_attempts': total, 'elapsed_seconds': time.perf_counter()-start,
              'note': 'Known-input codec round trip. Candidates derive from reconstructed text and compiler diagnostics only; original used for offline scoring. Compilation does not prove exactness or correctness. Regex signature extraction is not C# scope resolution.'}
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('\n=== Scope-inspired repair ===\n' + json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
