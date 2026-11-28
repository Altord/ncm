"""Compile a two-codec ensemble using verified saved single-codec results."""
import argparse
import json
from pathlib import Path
from evaluation.ensemble_codec_methods import load_rows
from evaluation.godot_csharp_generator_pilot import compile_candidate
from training.data import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'ensemble', 'baseline', 'compiled-baseline', 'second-baseline', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() or out.with_suffix('.report.json').exists():
        raise SystemExit('Output exists')
    def meta(path):
        return json.loads(Path(path).with_suffix('.report.json').read_text(encoding='utf-8'))
    ensemble_meta, base_meta, compiled_meta, second_meta = map(meta, (a.ensemble, a.baseline, a.compiled_baseline, a.second_baseline))
    source_hash = sha256_file(a.input)
    if (any(r['input_sha256'] != source_hash for r in (ensemble_meta, base_meta, compiled_meta, second_meta))
        or sha256_file(base_meta['checkpoint']) != ensemble_meta['checkpoint_a_sha256']
        or compiled_meta['candidate_sha256'] != sha256_file(a.baseline)
        or not second_meta.get('compile_requested')
        or sha256_file(second_meta['checkpoint']) != ensemble_meta['checkpoint_b_sha256']):
        raise SystemExit('Input, codec or saved compilation provenance differs')
    source, ens, baseline, compiled, second = (load_rows(path) for path in (a.input, a.ensemble, a.baseline, a.compiled_baseline, a.second_baseline))
    if any(rows.keys() != source.keys() for rows in (ens, baseline, compiled, second)):
        raise SystemExit('IDs differ')
    output, calls = [], 0
    for key, original in source.items():
        target, b, old, s = original['target'], baseline[key], compiled[key], second[key]
        if b['target'] != target or s['target'] != target or old['selected'] != b['reconstructed']:
            raise SystemExit(f'Target or baseline mismatch: {key}')
        if s['original_compile']['status'] != 'passed': raise SystemExit(f'Original compile unavailable: {key}')
        selected = ens[key]['selected']
        saved = [(old['selected'], old['compile_status']), (s['reconstructed'], s['reconstruction_compile']['status'])]
        if any(status not in ('passed', 'failed', 'empty') for _, status in saved):
            raise SystemExit(f'Saved compile unavailable: {key}')
        if saved[0][0] == saved[1][0] and saved[0][1] != saved[1][1]:
            raise SystemExit(f'Conflicting saved compile: {key}')
        cached = dict(saved)
        if selected in cached: status = cached[selected]
        elif not selected.strip(): status = 'empty'
        else:
            status = compile_candidate(selected, 'net8.0')['status']; calls += 1
            if status not in ('passed', 'failed', 'empty'): raise SystemExit(f'Compile unavailable: {key}')
        output.append({'id': key, 'selected': selected, 'compile_status': status,
                       'exact_text': selected == target, 'a_status': saved[0][1], 'b_status': saved[1][1]})
        print(f'Processed {len(output)}/{len(source)}; new compiler calls={calls}', flush=True)
    summary = {'examples': len(output), 'baseline_a_compiles': sum(r['a_status'] == 'passed' for r in output),
               'baseline_b_compiles': sum(r['b_status'] == 'passed' for r in output),
               'ensemble_compiles': sum(r['compile_status'] == 'passed' for r in output),
               'ensemble_exact_text': sum(r['exact_text'] for r in output), 'new_compiler_calls': calls,
               'input_sha256': source_hash, 'ensemble_sha256': sha256_file(a.ensemble),
               'scope': ensemble_meta['scope'],
               'limitations': 'Repeated stress set; compilation does not establish behavior or generation quality.'}
    with out.open('x', encoding='utf-8') as f:
        for r in output: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    with out.with_suffix('.report.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2); f.write('\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
