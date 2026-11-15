"""Try supplied-signature restoration on previously failing refinement candidates."""
import argparse
import json
import time
from pathlib import Path
from training.data import sha256_file
from evaluation.codec_prompt_signature_copy import copy_signature
from evaluation.evaluate_pointer_stress import make_signature_prompt, compile_or_reuse


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    result = {row['id']: row for row in rows}
    if not rows or len(result) != len(rows):
        raise ValueError(f'Empty file or duplicate IDs: {path}')
    return result


def evaluate(source, hybrid, heads, framework='net8.0'):
    if any(set(data) != set(source) for data in (hybrid, *heads)):
        raise ValueError('Input IDs differ')
    results, calls = [], 0
    for key, original in source.items():
        prior = hybrid[key]['select_strict']
        if prior['compile_status'] not in ('passed', 'failed', 'empty'):
            raise ValueError(f'Unavailable compile result: {key}')
        baseline = hybrid[key]['baseline']['text']
        if any(data[key]['baseline'] != baseline for data in heads):
            raise ValueError(f'Different baseline: {key}')
        target = original['target']
        chosen = {'text': prior['text'], 'compile_status': prior['compile_status'], 'selection': 'earlier_hybrid'}
        tries = []
        if prior['compile_status'] != 'passed':
            # Target is used to create the supplied synthetic signature, as in the earlier stress tests.
            prompt = make_signature_prompt(target)
            cache = {prior['text']: prior['compile_status']}
            for number, data in enumerate(heads, 1):
                row = data[key]
                if row['selected_compile'] not in ('passed', 'failed', 'empty'):
                    raise ValueError(f'Unavailable refinement compile result: {key}')
                if row['selected'] in cache and cache[row['selected']] != row['selected_compile']:
                    raise ValueError(f'Conflicting compile results: {key}')
                cache[row['selected']] = row['selected_compile']
                for fallback in (False, True):
                    text, reason = copy_signature(prompt, row['selected'], fallback=fallback)
                    status, called = compile_or_reuse(text, cache, framework)
                    calls += called
                    tries.append({'head': number, 'fallback': fallback, 'reason': reason, 'text': text, 'compile_status': status})
                    if status == 'passed':
                        chosen = {'text': text, 'compile_status': status, 'selection': f'head{number}_signature_{reason}'}
                        break
                if chosen['compile_status'] == 'passed':
                    break
        results.append({'id': key, 'baseline': baseline, 'previous': prior, 'selected': chosen, 'attempts': tries,
                        'selected_exact': chosen['text'] == target, 'previous_exact': prior['text'] == target})
        print(f'Processed {len(results)}/{len(source)}; new compiler calls={calls}', flush=True)
    return results, calls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'hybrid', 'previous-refinement', 'hard-refinement', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--framework', default='net8.0')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists():
        raise SystemExit('Output already exists')
    source, hybrid, previous, hard = (load(path) for path in (args.input, args.hybrid, args.previous_refinement, args.hard_refinement))
    start = time.perf_counter()
    rows, calls = evaluate(source, hybrid, (previous, hard), args.framework)
    report = {'examples': len(rows), 'previous_compiles': sum(r['previous']['compile_status'] == 'passed' for r in rows),
        'selected_compiles': sum(r['selected']['compile_status'] == 'passed' for r in rows),
        'previous_exact': sum(r['previous_exact'] for r in rows), 'selected_exact': sum(r['selected_exact'] for r in rows),
        'new_compiles': sum(r['previous']['compile_status'] != 'passed' and r['selected']['compile_status'] == 'passed' for r in rows),
        'new_compiler_calls': calls, 'elapsed_seconds': time.perf_counter()-start,
        'input_sha256': sha256_file(args.input), 'hybrid_sha256': sha256_file(args.hybrid),
        'previous_refinement_sha256': sha256_file(args.previous_refinement), 'hard_refinement_sha256': sha256_file(args.hard_refinement),
        'scope': 'Known-target codec round trip; synthetic prompt supplies target signature. Selection uses compiler status only; target body used for offline scoring.',
        'limitations': 'Stress set repeatedly used; no independent evaluation, generation, or behavioral correctness test.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('\n=== Refinement + signature rescue results to share ===\n'+json.dumps(report, indent=2), flush=True)
    for row in rows:
        if row['previous']['compile_status'] != 'passed' and row['selected']['compile_status'] == 'passed':
            print(f"Rescued {row['id']}: exact={row['selected_exact']}; {row['selected']['text']}", flush=True)


if __name__ == '__main__':
    main()
