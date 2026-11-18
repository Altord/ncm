"""Classify why saved codec failures receive no AST repair candidates."""
import argparse
import json
from collections import Counter
from pathlib import Path
from evaluation.ast_codec_candidates import ast_candidates
from evaluation.codec_prompt_signature_copy import WRAPPER, parse_method
from evaluation.evaluate_pointer_stress import make_signature_prompt
from training.csharp_structure import csharp_parser, _walk
from training.data import sha256_file


def load(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    data = {r['id']: r for r in rows}
    if not rows or len(rows) != len(data): raise ValueError(f'Empty or duplicate IDs: {path}')
    return data


def classify(text, prompt):
    info = parse_method(text)
    raw = (WRAPPER + text + '\n}').encode('utf-8')
    tree = csharp_parser().parse(raw)
    nodes = list(_walk(tree.root_node))
    errors = [n for n in nodes if n.type == 'ERROR' or n.is_missing]
    direct = [n for n in nodes if n.type == 'identifier' and n.parent and n.parent.type == 'ERROR']
    candidates = ast_candidates(prompt, text)
    return {'method_found': info is not None, 'parser_has_error': tree.root_node.has_error,
            'error_nodes': len(errors), 'direct_error_identifiers': len(direct),
            'has_comment': any('comment' in n.type for n in nodes),
            'has_string': any('string' in n.type or n.type == 'character_literal' for n in nodes),
            'candidate_count': len(candidates)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'sample', 'signature-copy', 'output'): p.add_argument('--'+name, required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists(): raise SystemExit('Output exists')
    source, sample, copies = (load(p) for p in (args.input, args.sample, args.signature_copy))
    if not set(sample).issubset(source) or not set(sample).issubset(copies): raise SystemExit('IDs differ')
    rows = []
    for key, saved in sample.items():
        original, copy = source[key], copies[key]
        if saved['baseline'] != original['reconstructed'] or copy['before_exact'] != (original['reconstructed'] == original['target']):
            raise SystemExit(f'Baseline mismatch: {key}')
        if copy['selected'] == original['target']: continue
        try: prompt = make_signature_prompt(original['target'])
        except ValueError: prompt = ''
        info = classify(copy['selected'], prompt)
        rows.append({'id': key, 'signature_copy_selection': copy['selection'], **info})
    totals = Counter({name: 0 for name in ('method_found', 'parser_has_error', 'has_comment', 'has_string',
                                           'direct_error_identifier', 'candidate_available')})
    for row in rows:
        for name in ('method_found', 'parser_has_error', 'has_comment', 'has_string'):
            totals[name] += bool(row[name])
        totals['direct_error_identifier'] += row['direct_error_identifiers'] > 0
        totals['candidate_available'] += row['candidate_count'] > 0
    report = {'sample_size': len(sample), 'failures_after_signature_copy': len(rows), 'counts': dict(totals),
              'input_sha256': sha256_file(args.input), 'sample_sha256': sha256_file(args.sample),
              'signature_sha256': sha256_file(args.signature_copy), 'rows': rows,
              'scope': 'Saved known-target codec reconstruction diagnostic after target-derived synthetic signature copy; no model inference or compiler.',
              'limitations': 'Parser syntax errors do not measure semantic or type correctness. Reused method dataset; no generation evaluation.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f: json.dump(report, f, indent=2); f.write('\n')
    print('\n=== AST failure coverage to share ===')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
    print('First 10 failure classifications:')
    for row in rows[:10]: print(json.dumps(row))
    print(f'Report: {out}')


if __name__ == '__main__': main()
