"""Combine saved pointer reconstructions with copying an explicitly supplied signature.

The supplied synthetic prompt is verified against the stress task. The original
target is otherwise used only for offline scoring, never candidate selection.
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

ORIGINAL_NAMES = ('baseline', 'signature', 'contextual')
VARIANTS = ('signature_strict', 'signature_fallback', 'contextual_strict', 'contextual_fallback')


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    result = {r['id']: r for r in rows}
    if not rows or len(rows) != len(result):
        raise ValueError(f'Empty rows or duplicate IDs in {path}')
    return result


def evaluate(original, prior, framework='net8.0'):
    if set(original) != set(prior):
        raise ValueError('Stress input and saved pointer rows have different IDs')
    totals = {name: {'exact_text': 0, 'compile_passes': 0, 'name_exact': 0,
                     'new_exact_text': 0, 'lost_exact_text': 0,
                     'new_compiles': 0, 'lost_compiles': 0, 'changed': 0}
              for name in ORIGINAL_NAMES + VARIANTS}
    results, calls = [], 0
    for index, (key, source) in enumerate(original.items(), 1):
        saved = prior[key]
        if saved.get('prompt') != make_signature_prompt(source['target']):
            raise ValueError(f'Synthetic prompt mismatch for {key}')
        cache = {}
        for name in ORIGINAL_NAMES:
            item = saved[name]
            text, status = item['text'], item['compile_status']
            if status not in ('passed', 'failed', 'empty'):
                raise ValueError(f'Unavailable saved compiler result for {key}/{name}')
            if text in cache and cache[text] != status:
                raise ValueError(f'Conflicting saved compiler results for {key}')
            if bool(item['exact_text']) != (text == source['target']):
                raise ValueError(f'Saved exact-text score differs for {key}/{name}')
            cache[text] = status
        # Edits use the explicitly supplied prompt and saved output only.
        candidates = {name: (saved[name]['text'], 'saved') for name in ORIGINAL_NAMES}
        for name in ('signature', 'contextual'):
            for suffix, fallback in (('strict', False), ('fallback', True)):
                candidates[name + '_' + suffix] = copy_signature(saved['prompt'], saved[name]['text'], fallback=fallback)
        baseline_text = candidates['baseline'][0]
        baseline_status = cache[baseline_text]
        baseline_exact = baseline_text == source['target']
        expected_name = method_name(source['target'])
        result = {'id': key}
        for name, (text, reason) in candidates.items():
            status, called = compile_or_reuse(text, cache, framework)
            calls += called
            exact = text == source['target']
            summary = totals[name]
            summary['exact_text'] += exact
            summary['compile_passes'] += status == 'passed'
            summary['name_exact'] += expected_name is not None and method_name(text) == expected_name
            summary['new_exact_text'] += not baseline_exact and exact
            summary['lost_exact_text'] += baseline_exact and not exact
            summary['new_compiles'] += baseline_status != 'passed' and status == 'passed'
            summary['lost_compiles'] += baseline_status == 'passed' and status != 'passed'
            summary['changed'] += text != baseline_text
            result[name] = {'text': text, 'selection': reason, 'compile_status': status,
                            'exact_text': exact,
                            'character_similarity': difflib.SequenceMatcher(None, source['target'], text, autojunk=False).ratio()}
        results.append(result)
        if index % 8 == 0 or index == len(original):
            print(f'Processed {index}/{len(original)}; new compiler calls={calls}', flush=True)
    for name in totals:
        totals[name]['mean_character_similarity'] = sum(r[name]['character_similarity'] for r in results) / len(results)
    return totals, results, calls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--pointer-results', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--framework', default='net8.0')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists():
        raise SystemExit('Output exists; choose a new path')
    source = Path(args.input)
    pointer = Path(args.pointer_results)
    report = json.loads(pointer.with_suffix('.report.json').read_text(encoding='utf-8'))
    if report.get('input_sha256') != sha256_file(source):
        raise SystemExit('Saved pointer results use a different stress input')
    original, prior = load(source), load(pointer)
    if report.get('examples') != len(original):
        raise SystemExit('Saved report has an unexpected number of examples')
    start = time.perf_counter()
    totals, rows, calls = evaluate(original, prior, args.framework)
    for name in ORIGINAL_NAMES:
        for field in ('exact_text', 'compile_passes', 'name_exact'):
            if totals[name][field] != report['results'][name][field]:
                raise SystemExit(f'Saved report does not match rows for {name}/{field}')
    output = {'examples': len(rows), 'results': totals, 'new_compiler_calls': calls,
              'input_sha256': sha256_file(source), 'pointer_results_sha256': sha256_file(pointer),
              'elapsed_seconds': time.perf_counter()-start,
              'scope': 'Known-target codec round trip with a synthetic prompt containing the exact target signature; pointer and signature copy use only this prompt and decoded text.',
              'limitations': 'Not unknown-target generation. Signature was extracted from the target to simulate a supplied-signature task; nearby project context was not provided. Repeated development set. Compilation does not prove behavior. Token accuracy is not reported because text edits can change token alignment.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    out.with_suffix('.report.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print('\n=== Pointer + signature-copy stress ===\n' + json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    main()
