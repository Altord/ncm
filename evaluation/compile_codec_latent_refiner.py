"""Compile saved adapter outputs, reusing identical baseline compiler results."""
import argparse
import json
import time
from pathlib import Path
from evaluation.compile_codec_ensemble import compile_rows
from evaluation.ensemble_codec_methods import load_rows
from training.data import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'adapter-output', 'baseline', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--framework', default='net8.0')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output exists')
    adapter_report = json.loads(Path(args.adapter_output).with_suffix('.report.json').read_text(encoding='utf-8'))
    baseline_report = json.loads(Path(args.baseline).with_suffix('.report.json').read_text(encoding='utf-8'))
    if (adapter_report['input_sha256'] != sha256_file(args.input) or
        baseline_report['input_sha256'] != adapter_report['input_sha256'] or
        not baseline_report.get('compile_requested') or
        sha256_file(baseline_report['checkpoint']) != adapter_report['codec_sha256']):
        raise SystemExit('Input, codec or saved baseline compile report differs')
    source, adapter, baseline = (load_rows(p) for p in (args.input, args.adapter_output, args.baseline))
    started = time.perf_counter()
    rows, calls = compile_rows(source, adapter, baseline, baseline, args.framework)
    summary = {'examples': len(rows), 'baseline_compiles': sum(r['baseline_a_compile'] == 'passed' for r in rows),
               'adapter_compiles': sum(r['compile_status'] == 'passed' for r in rows),
               'adapter_exact_text': sum(r['exact_text'] for r in rows),
               'compile_gains': sum(r['baseline_a_compile'] != 'passed' and r['compile_status'] == 'passed' for r in rows),
               'compile_losses': sum(r['baseline_a_compile'] == 'passed' and r['compile_status'] != 'passed' for r in rows),
               'new_compiler_calls': calls, 'seconds': time.perf_counter()-started,
               'input_sha256': sha256_file(args.input), 'adapter_output_sha256': sha256_file(args.adapter_output),
               'scope': 'Known-target codec round trip; unchanged baseline text reuses saved compiler status.',
               'limitations': 'Reused stress set; compilation does not establish behavior. No generator or independent test.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    with out.with_suffix('.report.json').open('x', encoding='utf-8') as f: json.dump(summary, f, indent=2); f.write('\n')
    print('\n=== Latent adapter compile comparison ===\n'+json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
