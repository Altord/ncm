"""Compile saved ensemble reconstructions, reusing identical saved results."""
import argparse
import json
import time
from pathlib import Path
from evaluation.ensemble_codec_methods import load_rows
from evaluation.godot_csharp_generator_pilot import compile_candidate
from training.data import sha256_file


def compile_rows(source, ensemble, a, b, framework='net8.0'):
    if any(set(d) != set(source) for d in (ensemble, a, b)): raise ValueError('IDs differ')
    rows, calls = [], 0
    for key, original in source.items():
        target, selected = original['target'], ensemble[key]['selected']
        if any(codec[key]['target'] != target for codec in (a, b)): raise ValueError(f'Target mismatch: {key}')
        cache = {}
        for codec in (a, b):
            row = codec[key]
            if row['original_compile']['status'] != 'passed': raise ValueError(f'Original did not compile: {key}')
            text, status = row['reconstructed'], row['reconstruction_compile']['status']
            if status not in ('passed', 'failed', 'empty'): raise ValueError(f'Compiler unavailable: {key}')
            if text in cache and cache[text] != status: raise ValueError(f'Conflicting compile results: {key}')
            cache[text] = status
        if selected in cache:
            status = cache[selected]
        elif not selected.strip():
            status = 'empty'
        else:
            result = compile_candidate(selected, framework)
            calls += 1
            status = result['status']
            if status not in ('passed', 'failed', 'empty'): raise RuntimeError(f'Compiler unavailable: {key}: {result}')
        rows.append({'id': key, 'selected': selected, 'compile_status': status,
                     'exact_text': selected == target, 'baseline_a_compile': a[key]['reconstruction_compile']['status'],
                     'baseline_b_compile': b[key]['reconstruction_compile']['status']})
        print(f'Processed {len(rows)}/{len(source)}; new compiler calls={calls}', flush=True)
    return rows, calls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'ensemble', 'baseline-a', 'baseline-b', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--framework', default='net8.0')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output exists')
    report = json.loads(Path(args.ensemble).with_suffix('.report.json').read_text(encoding='utf-8'))
    if report['input_sha256'] != sha256_file(args.input): raise SystemExit('Ensemble input hash differs')
    for path, key in ((args.baseline_a, 'checkpoint_a_sha256'), (args.baseline_b, 'checkpoint_b_sha256')):
        baseline_report = json.loads(Path(path).with_suffix('.report.json').read_text(encoding='utf-8'))
        if (not baseline_report.get('compile_requested') or baseline_report['input_sha256'] != report['input_sha256'] or
            sha256_file(baseline_report['checkpoint']) != report[key]):
            raise SystemExit(f'Saved compile report does not match ensemble: {path}')
    source, ensemble, a, b = (load_rows(p) for p in (args.input, args.ensemble, args.baseline_a, args.baseline_b))
    start = time.perf_counter()
    rows, calls = compile_rows(source, ensemble, a, b, args.framework)
    summary = {'examples': len(rows), 'baseline_a_compiles': sum(r['baseline_a_compile']=='passed' for r in rows),
        'baseline_b_compiles': sum(r['baseline_b_compile']=='passed' for r in rows),
        'ensemble_compiles': sum(r['compile_status']=='passed' for r in rows),
        'ensemble_exact_text': sum(r['exact_text'] for r in rows), 'new_compiler_calls': calls,
        'new_compiles_vs_a': sum(r['baseline_a_compile']!='passed' and r['compile_status']=='passed' for r in rows),
        'lost_compiles_vs_a': sum(r['baseline_a_compile']=='passed' and r['compile_status']!='passed' for r in rows),
        'seconds': time.perf_counter()-start, 'input_sha256': sha256_file(args.input),
        'ensemble_sha256': sha256_file(args.ensemble),
        'scope': 'Known-target codec round trip. Compiler checks are evaluated after fixed-logit ensemble prediction; target used only for offline scoring.',
        'limitations': ('No semantic or generator test. Reused development stress set. '
                        + ('Three' if report.get('checkpoint_c_sha256') else 'Two')
                        + ' codecs require separate latent streams and compute.')}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    out.with_suffix('.report.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print('\n=== Ensemble compile stress results to share ===\n'+json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
