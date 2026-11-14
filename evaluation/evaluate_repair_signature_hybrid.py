"""Combine saved compile-guided repairs with supplied synthetic signatures.

Compiler-based selection is not a semantic guarantee. Targets verify the
synthetic task and score results offline; they do not choose candidates.
"""
from __future__ import annotations
import argparse
import difflib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.data import sha256_file
from evaluation.codec_prompt_signature_copy import copy_signature
from evaluation.compare_codec_method_names import method_name
from evaluation.evaluate_pointer_stress import make_signature_prompt, compile_or_reuse

NAMES = ('baseline', 'repair', 'repair_strict', 'repair_fallback',
         'pointer_strict', 'pointer_fallback', 'select_strict', 'select_fallback')


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    result = {r['id']: r for r in rows}
    if not rows or len(rows) != len(result):
        raise ValueError(f'Empty rows or duplicate IDs: {path}')
    return result


def put_cached(cache, text, status, identifier):
    if status not in ('passed', 'failed', 'empty'):
        raise ValueError(f'Unavailable saved compiler status: {identifier}')
    if text in cache and cache[text] != status:
        raise ValueError(f'Conflicting saved compiler statuses: {identifier}')
    cache[text] = status


def choose_compiling(first, second):
    """Prefer the first compiling candidate; no target or correctness oracle."""
    if first['compile_status'] == 'passed':
        return {**first, 'selection': 'repair_preferred'}
    if second['compile_status'] == 'passed':
        return {**second, 'selection': 'pointer_fallback'}
    return {**first, 'selection': 'neither_compiles'}


def evaluate(original, repaired, hybrid, pointer, framework='net8.0'):
    if any(set(rows) != set(original) for rows in (repaired, hybrid, pointer)):
        raise ValueError('Input rows have different IDs')
    totals = {name: {'exact_text': 0, 'compile_passes': 0, 'name_exact': 0,
                     'new_exact_vs_repair': 0, 'lost_exact_vs_repair': 0,
                     'new_compiles_vs_repair': 0, 'lost_compiles_vs_repair': 0}
              for name in NAMES}
    output, calls = [], 0
    for index, (key, source) in enumerate(original.items(), 1):
        old, mix, ptr = repaired[key], hybrid[key], pointer[key]
        if (old['baseline'] != mix['baseline']['text'] or ptr['baseline']['text'] != old['baseline'] or
            ptr['baseline']['compile_status'] != mix['baseline']['compile_status']):
            raise ValueError(f'Frozen codec baselines differ: {key}')
        if ptr.get('prompt') != make_signature_prompt(source['target']):
            raise ValueError(f'Synthetic prompt differs: {key}')
        if bool(old['selected_text_exact']) != (old['selected'] == source['target']):
            raise ValueError(f'Saved repair score differs: {key}')
        cache = {}
        put_cached(cache, mix['baseline']['text'], mix['baseline']['compile_status'], key)
        put_cached(cache, old['selected'], old['compile_status'], key)
        for name in ('signature_strict', 'signature_fallback'):
            item = mix[name]
            put_cached(cache, item['text'], item['compile_status'], key)
            if bool(item['exact_text']) != (item['text'] == source['target']):
                raise ValueError(f'Saved hybrid score differs: {key}/{name}')
        candidates = {
            'baseline': {'text': mix['baseline']['text'], 'selection': 'saved_codec'},
            'repair': {'text': old['selected'], 'selection': 'saved_repair'},
            'pointer_strict': {'text': mix['signature_strict']['text'], 'selection': 'saved_pointer_strict'},
            'pointer_fallback': {'text': mix['signature_fallback']['text'], 'selection': 'saved_pointer_fallback'},
        }
        for name, fallback in (('repair_strict', False), ('repair_fallback', True)):
            text, reason = copy_signature(ptr['prompt'], old['selected'], fallback=fallback)
            candidates[name] = {'text': text, 'selection': reason}
        for name, candidate in candidates.items():
            status, called = compile_or_reuse(candidate['text'], cache, framework)
            calls += called
            candidate['compile_status'] = status
        candidates['select_strict'] = choose_compiling(candidates['repair_strict'], candidates['pointer_strict'])
        candidates['select_fallback'] = choose_compiling(candidates['repair_fallback'], candidates['pointer_fallback'])
        expected_name = method_name(source['target'])
        old_exact = old['selected'] == source['target']
        old_compiles = old['compile_status'] == 'passed'
        result = {'id': key}
        for name in NAMES:
            candidate = candidates[name]
            text, status = candidate['text'], candidate['compile_status']
            exact = text == source['target']
            totals[name]['exact_text'] += exact
            totals[name]['compile_passes'] += status == 'passed'
            totals[name]['name_exact'] += expected_name is not None and method_name(text) == expected_name
            totals[name]['new_exact_vs_repair'] += not old_exact and exact
            totals[name]['lost_exact_vs_repair'] += old_exact and not exact
            totals[name]['new_compiles_vs_repair'] += not old_compiles and status == 'passed'
            totals[name]['lost_compiles_vs_repair'] += old_compiles and status != 'passed'
            result[name] = {**candidate, 'exact_text': exact,
                            'character_similarity': difflib.SequenceMatcher(None, source['target'], text, autojunk=False).ratio()}
        output.append(result)
        if index % 8 == 0 or index == len(original):
            print(f'Processed {index}/{len(original)}; new compiler calls={calls}', flush=True)
    for name in totals:
        totals[name]['mean_character_similarity'] = sum(r[name]['character_similarity'] for r in output) / len(output)
    return totals, output, calls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--repair', required=True)
    p.add_argument('--hybrid', required=True)
    p.add_argument('--pointer', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--framework', default='net8.0')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists():
        raise SystemExit('Output exists; choose a new path')
    source, repair, hybrid, pointer = map(Path, (args.input, args.repair, args.hybrid, args.pointer))
    h_report = json.loads(hybrid.with_suffix('.report.json').read_text(encoding='utf-8'))
    p_report = json.loads(pointer.with_suffix('.report.json').read_text(encoding='utf-8'))
    r_report = json.loads(repair.with_suffix('.report.json').read_text(encoding='utf-8'))
    if (h_report.get('input_sha256') != sha256_file(source) or
        p_report.get('input_sha256') != sha256_file(source) or
        h_report.get('pointer_results_sha256') != sha256_file(pointer)):
        raise SystemExit('Saved reports do not match stress input and pointer results')
    originals, repairs, hybrids, pointers = (load(p) for p in (source, repair, hybrid, pointer))
    if any(r.get('examples') != len(originals) for r in (h_report, p_report, r_report)):
        raise SystemExit('Saved reports have unexpected example counts')
    if sum(r['compile_status'] == 'passed' for r in repairs.values()) != r_report.get('selected_compiles'):
        raise SystemExit('Saved repair report does not match rows')
    if sum(r['selected_text_exact'] for r in repairs.values()) != r_report.get('selected_exact_text'):
        raise SystemExit('Saved repair exact-text report does not match rows')
    start = time.perf_counter()
    totals, rows, calls = evaluate(originals, repairs, hybrids, pointers, args.framework)
    for result_name, saved_name in (('baseline', 'baseline'), ('pointer_strict', 'signature_strict'),
                                    ('pointer_fallback', 'signature_fallback')):
        for field in ('exact_text', 'compile_passes', 'name_exact'):
            if totals[result_name][field] != h_report['results'][saved_name][field]:
                raise SystemExit(f'Hybrid report does not match saved rows: {saved_name}/{field}')
    report = {'examples': len(rows), 'results': totals, 'new_compiler_calls': calls,
              'elapsed_seconds': time.perf_counter()-start,
              'input_sha256': sha256_file(source), 'repair_sha256': sha256_file(repair),
              'hybrid_sha256': sha256_file(hybrid), 'pointer_sha256': sha256_file(pointer),
              'scope': 'Known-target codec round trip. Target signature is supplied in a synthetic prompt; candidate selection uses compiler results, not target text.',
              'limitations': 'Not unknown-target generation. Signature extracted from target to simulate supplied-signature task. Prior repair uses compiler-only selection. Compilation does not prove behavior. Development stress set has been repeatedly inspected.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('\n=== Repair + supplied signature ===\n' + json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
