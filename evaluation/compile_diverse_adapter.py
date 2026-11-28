"""Compile a latent adapter against a separately compiled codec baseline."""
import argparse
import json
import time
from pathlib import Path

from evaluation.ensemble_codec_methods import load_rows
from evaluation.godot_csharp_generator_pilot import compile_candidate
from training.data import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'adapter-output', 'baseline', 'compiled-baseline', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists():
        raise SystemExit('Output exists')
    def report(path):
        return json.loads(Path(path).with_suffix('.report.json').read_text(encoding='utf-8'))
    adapter_meta, base_meta, compiled_meta = map(report, (args.adapter_output, args.baseline, args.compiled_baseline))
    input_hash = sha256_file(args.input)
    if (any(r['input_sha256'] != input_hash for r in (adapter_meta, base_meta, compiled_meta))
        or sha256_file(base_meta['checkpoint']) != adapter_meta['codec_sha256']
        or compiled_meta['candidate_sha256'] != sha256_file(args.baseline)):
        raise SystemExit('Input, codec, or compiled baseline provenance differs')
    source, adapter, baseline, compiled = (load_rows(p) for p in (args.input, args.adapter_output, args.baseline, args.compiled_baseline))
    if any(rows.keys() != source.keys() for rows in (adapter, baseline, compiled)):
        raise SystemExit('IDs differ')
    rows, calls = [], 0
    started = time.perf_counter()
    for key, original in source.items():
        target = original['target']
        base, prior, selected = baseline[key], compiled[key], adapter[key]['selected']
        if base['target'] != target or prior['selected'] != base['reconstructed']:
            raise SystemExit(f'Baseline text or target differs: {key}')
        before = prior['compile_status']
        if before not in ('passed', 'failed', 'empty'):
            raise SystemExit(f'Baseline compile unavailable: {key}')
        if selected == prior['selected']:
            status = before
        elif not selected.strip():
            status = 'empty'
        else:
            status = compile_candidate(selected, 'net8.0')['status']
            calls += 1
            if status not in ('passed', 'failed', 'empty'):
                raise SystemExit(f'Compiler unavailable: {key}: {status}')
        rows.append({'id': key, 'selected': selected, 'compile_status': status,
                     'baseline_compile_status': before, 'exact_text': selected == target,
                     'baseline_exact_text': base['reconstructed'] == target})
        print(f'Processed {len(rows)}/{len(source)}; new compiler calls={calls}', flush=True)
    summary = {'examples': len(rows), 'baseline_compiles': sum(r['baseline_compile_status'] == 'passed' for r in rows),
               'adapter_compiles': sum(r['compile_status'] == 'passed' for r in rows),
               'baseline_exact_text': sum(r['baseline_exact_text'] for r in rows),
               'adapter_exact_text': sum(r['exact_text'] for r in rows),
               'compile_gains': sum(r['baseline_compile_status'] != 'passed' and r['compile_status'] == 'passed' for r in rows),
               'compile_losses': sum(r['baseline_compile_status'] == 'passed' and r['compile_status'] != 'passed' for r in rows),
               'new_compiler_calls': calls, 'seconds': time.perf_counter() - started,
               'input_sha256': input_hash, 'adapter_output_sha256': sha256_file(args.adapter_output),
               'compiled_baseline_sha256': sha256_file(args.compiled_baseline),
               'scope': 'Known-input codec; unchanged baseline text reuses its saved compiler status.',
               'limitations': 'Repeated stress set; compilation does not establish behavior or generation quality.'}
    with out.open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    with out.with_suffix('.report.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2); f.write('\n')
    print('\n=== Broader codec adapter compile comparison ===\n' + json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
