"""Compile a saved codec candidate while reusing identical baseline results."""
import argparse
import json
import time
from pathlib import Path
from evaluation.compile_codec_ensemble import compile_rows
from evaluation.ensemble_codec_methods import load_rows
from training.data import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input','candidate','baseline','output'): p.add_argument('--'+name, required=True)
    p.add_argument('--framework', default='net8.0')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output exists')
    candidate_meta = json.loads(Path(args.candidate).with_suffix('.report.json').read_text(encoding='utf-8'))
    baseline_meta = json.loads(Path(args.baseline).with_suffix('.report.json').read_text(encoding='utf-8'))
    if (candidate_meta['input_sha256'] != sha256_file(args.input) or
        baseline_meta['input_sha256'] != candidate_meta['input_sha256'] or not baseline_meta.get('compile_requested')):
        raise SystemExit('Input or baseline compile report differs')
    source, candidate, baseline = (load_rows(path) for path in (args.input, args.candidate, args.baseline))
    if candidate.keys() != source.keys() or baseline.keys() != source.keys(): raise SystemExit('IDs differ')
    for key, row in source.items():
        if candidate[key]['target'] != row['target']: raise SystemExit(f'Candidate target differs: {key}')
    selected = {key: {'selected': row['reconstructed']} for key, row in candidate.items()}
    started = time.perf_counter()
    rows, calls = compile_rows(source, selected, baseline, baseline, args.framework)
    result = {'examples':len(rows), 'baseline_compiles':sum(r['baseline_a_compile']=='passed' for r in rows),
              'candidate_compiles':sum(r['compile_status']=='passed' for r in rows),
              'candidate_exact_text':sum(r['exact_text'] for r in rows), 'new_compiler_calls':calls,
              'input_sha256':sha256_file(args.input), 'candidate_sha256':sha256_file(args.candidate),
              'seconds':time.perf_counter()-started,
              'scope':'Known-target codec reconstructions; identical text reuses a saved baseline compiler result.',
              'limitations':'Repeated stress dataset; compilation does not establish behavior. Not generation.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    with out.with_suffix('.report.json').open('x', encoding='utf-8') as f: json.dump(result, f, indent=2); f.write('\n')
    print('\n=== Saved codec compilation ===\n'+json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
