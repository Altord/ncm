"""Evaluate supplied-signature copying on saved ensemble stress outputs."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path
from evaluation.codec_prompt_signature_copy import copy_signature
from evaluation.evaluate_pointer_stress import make_signature_prompt, compile_or_reuse
from training.data import sha256_file


def load(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    result = {r['id']: r for r in rows}
    if not rows or len(result) != len(rows): raise ValueError(f'Empty or duplicate IDs: {path}')
    return result


def evaluate(source, ensemble, hybrid, reference, framework='net8.0'):
    if any(set(source) != set(d) for d in (ensemble, hybrid, reference)): raise ValueError('Input IDs differ')
    rows, calls = [], 0
    for key, original in source.items():
        target = original['target']
        prev, old, ref = ensemble[key], hybrid[key], reference[key]
        if (ref['target'] != target or old['baseline']['text'] != ref['reconstructed'] or
            bool(prev['exact_text']) != (prev['selected'] == target)):
            raise ValueError(f'Saved baseline or target mismatch: {key}')
        if prev['compile_status'] not in ('passed', 'failed', 'empty'):
            raise ValueError(f'Unavailable ensemble compile result: {key}')
        # In this synthetic stress test the signature is constructed from the target;
        # the original body is never supplied to the copy or selection logic.
        prompt = make_signature_prompt(target)
        selected, reason = copy_signature(prompt, prev['selected'], fallback=True)
        status, called = compile_or_reuse(selected, {prev['selected']: prev['compile_status']}, framework)
        calls += called
        first = old['select_strict']
        if first['compile_status'] not in ('passed', 'failed', 'empty'):
            raise ValueError(f'Unavailable prior hybrid compile result: {key}')
        use_new = first['compile_status'] != 'passed' and status == 'passed'
        combined = selected if use_new else first['text']
        combined_status = status if use_new else first['compile_status']
        rows.append({'id':key, 'before':prev['selected'], 'before_compile':prev['compile_status'],
            'before_exact':prev['selected'] == target, 'selected':selected, 'selection':reason,
            'selected_compile':status, 'selected_exact':selected == target,
            'prior_hybrid_compile':first['compile_status'], 'prior_hybrid_exact':first['text'] == target,
            'combined':combined, 'combined_compile':combined_status, 'combined_exact':combined == target,
            'used_ensemble_fallback':use_new})
        print(f'Processed {len(rows)}/{len(source)}; new compiler calls={calls}', flush=True)
    return rows, calls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'ensemble', 'hybrid', 'reference', 'output'): p.add_argument('--'+name, required=True)
    p.add_argument('--framework', default='net8.0')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output exists')
    report = json.loads(Path(args.ensemble).with_suffix('.report.json').read_text(encoding='utf-8'))
    if report['input_sha256'] != sha256_file(args.input): raise SystemExit('Ensemble input differs')
    source, ensemble, hybrid, reference = (load(p) for p in (args.input, args.ensemble, args.hybrid, args.reference))
    if report['examples'] != len(source): raise SystemExit('Ensemble report count differs')
    start = time.perf_counter()
    rows, calls = evaluate(source, ensemble, hybrid, reference, args.framework)
    result = {'examples':len(rows), 'ensemble_compiles':sum(r['before_compile']=='passed' for r in rows),
        'ensemble_exact':sum(r['before_exact'] for r in rows),
        'signature_compiles':sum(r['selected_compile']=='passed' for r in rows),
        'signature_exact':sum(r['selected_exact'] for r in rows),
        'signature_compile_gains':sum(r['before_compile']!='passed' and r['selected_compile']=='passed' for r in rows),
        'signature_compile_losses':sum(r['before_compile']=='passed' and r['selected_compile']!='passed' for r in rows),
        'signature_exact_gains':sum(not r['before_exact'] and r['selected_exact'] for r in rows),
        'signature_exact_losses':sum(r['before_exact'] and not r['selected_exact'] for r in rows),
        'prior_hybrid_compiles':sum(r['prior_hybrid_compile']=='passed' for r in rows),
        'prior_hybrid_exact':sum(r['prior_hybrid_exact'] for r in rows),
        'combined_compiles':sum(r['combined_compile']=='passed' for r in rows),
        'combined_exact':sum(r['combined_exact'] for r in rows),
        'combined_rescues':sum(r['used_ensemble_fallback'] for r in rows),
        'selection_counts':dict(Counter(r['selection'] for r in rows)), 'new_compiler_calls':calls,
        'seconds':time.perf_counter()-start, 'input_sha256':sha256_file(args.input),
        'scope':'Known-target two-codec round trip; signature is synthetically extracted from target. Compiler status chooses hybrid candidate; original body used for offline scoring.',
        'limitations':'Reused stress set; not generation or behavioral correctness. Signature may not be available to a real user. Two codec passes and latent streams remain necessary.'}
    if result['ensemble_compiles'] != report['ensemble_compiles'] or result['ensemble_exact'] != report['ensemble_exact_text']:
        raise SystemExit('Saved ensemble report does not match rows')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    out.with_suffix('.report.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('\n=== Ensemble signature compile results to share ===\n'+json.dumps(result, indent=2), flush=True)
    for row in rows:
        if row['used_ensemble_fallback']:
            print(f"Rescued {row['id']}: exact={row['combined_exact']}; {row['combined']}")


if __name__ == '__main__': main()
