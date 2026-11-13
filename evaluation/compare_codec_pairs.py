"""Compare saved codec rows pairwise without running either model again."""
import argparse
from collections import Counter
import json
from pathlib import Path


def load(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    by_id = {row['id']: row for row in rows}
    if not rows or len(by_id) != len(rows) or None in by_id:
        raise ValueError(f'{path}: empty rows or missing/duplicate IDs')
    report = json.loads(path.with_suffix('.report.json').read_text(encoding='utf-8'))
    if report.get('examples') != len(rows):
        raise ValueError(f'{path}: report example count does not match rows')
    return by_id, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--show', type=int, default=5)
    args = p.parse_args()
    if args.show < 0:
        p.error('--show must be nonnegative')
    ref, rr = load(args.reference)
    candidate, cr = load(args.candidate)
    for field in ('input_sha256', 'text_key', 'sequence_length', 'code_only', 'code_extensions'):
        if field not in rr or rr[field] != cr.get(field):
            raise SystemExit(f'Reports are not comparable: {field} differs or is missing')
    if ref.keys() != candidate.keys():
        raise SystemExit('Rows have different IDs')
    counts = Counter()
    changes = []
    for key, a in ref.items():
        b = candidate[key]
        if a['target'] != b['target'] or a['valid_token_count'] != b['valid_token_count']:
            raise SystemExit(f'Target text or valid token count differs for {key}')
        old_exact, new_exact = bool(a['token_exact_match']), bool(b['token_exact_match'])
        counts['exact_both'] += old_exact and new_exact
        counts['exact_only_reference'] += old_exact and not new_exact
        counts['exact_only_candidate'] += new_exact and not old_exact
        counts['exact_neither'] += not old_exact and not new_exact
        delta = int(b['correct_tokens']) - int(a['correct_tokens'])
        counts['candidate_more_correct_tokens'] += delta > 0
        counts['reference_more_correct_tokens'] += delta < 0
        counts['same_correct_tokens'] += delta == 0
        if delta or old_exact != new_exact:
            changes.append((key, delta, old_exact, new_exact))
        old_compile, new_compile = a.get('reconstruction_compile'), b.get('reconstruction_compile')
        if old_compile is not None and new_compile is not None:
            old, new = old_compile.get('status'), new_compile.get('status')
            if old not in ('passed', 'failed', 'empty') or new not in ('passed', 'failed', 'empty'):
                raise ValueError(f'Compiler result unavailable or timed out for {key}')
            counts['compile_both'] += old == new == 'passed'
            counts['compile_only_reference'] += old == 'passed' and new != 'passed'
            counts['compile_only_candidate'] += new == 'passed' and old != 'passed'
            counts['compile_neither'] += old != 'passed' and new != 'passed'
    print(f'Examples: {len(ref)}; input SHA-256: {rr["input_sha256"]}')
    print(f"Exact: both={counts['exact_both']}; only reference={counts['exact_only_reference']}; "
          f"only candidate={counts['exact_only_candidate']}; neither={counts['exact_neither']}")
    print(f"Token counts: candidate better={counts['candidate_more_correct_tokens']}; "
          f"reference better={counts['reference_more_correct_tokens']}; same={counts['same_correct_tokens']}")
    if rr.get('compile_requested') and cr.get('compile_requested'):
        print(f"Compilation: both={counts['compile_both']}; only reference={counts['compile_only_reference']}; "
              f"only candidate={counts['compile_only_candidate']}; neither={counts['compile_neither']}")
    print(f'Examples with changed token count or exactness (up to {args.show}):')
    for key, delta, old, new in sorted(changes, key=lambda item: -abs(item[1]))[:args.show]:
        print(f'  {key}: correct-token delta={delta:+d}; exact {old} -> {new}')
    print('Saved paired codec reconstructions; no training or compilation performed. '
          'Token comparisons assume the same tokenizer; report metadata alone does not verify it.')


if __name__ == '__main__':
    main()
