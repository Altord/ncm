"""Bounded parameter-substitution diagnostic using a supplied signature.

This is an intentionally simple heuristic, not C# scope or semantic analysis.
"""
import argparse
import difflib
import itertools
import json
import re
import time
from pathlib import Path
from evaluation.codec_prompt_signature_copy import PROMPT_PREFIX, parse_method
from evaluation.evaluate_pointer_stress import make_signature_prompt, compile_or_reuse
from training.data import sha256_file

WORDS = {'if', 'else', 'return', 'true', 'false', 'null', 'new', 'out', 'ref', 'in', 'int', 'float',
         'bool', 'double', 'string', 'var', 'Math', 'MathF', 'static', 'this', 'base'}


def generate_candidates(prompt, text, limit):
    """Use prompt signature and reconstructed body only; never target body."""
    if not prompt.startswith(PROMPT_PREFIX):
        return [], False
    signature = prompt[len(PROMPT_PREFIX):].strip()
    if parse_method(signature + ' {}', require_clean=True) is None:
        return [], False
    match = re.search(r'\(([^()]*)\)\s*$', signature)
    if not match:
        return [], False
    params = []
    for part in match.group(1).split(','):
        param = re.fullmatch(r'\s*(?:(?:ref|out|in|params)\s+)?[A-Za-z_]\w*(?:\[\])?\s+([A-Za-z_]\w*)\s*', part)
        if not param:
            return [], False
        params.append(param.group(1))
    if not params:
        return [], False
    # Heuristic fallback for malformed headers. Braces in strings/comments can be ambiguous.
    offsets = [i for i in (text.find('=>'), text.find('{')) if i >= 0]
    if not offsets:
        return [], False
    body = text[min(offsets):]
    tokens = list(re.finditer(r'[A-Za-z_]\w*|\.\.\.|::|\?', body))
    spans = []
    for m in tokens:
        word = m.group()
        before, after = body[:m.start()].rstrip(), body[m.end():].lstrip()
        if word == '...':
            if spans and not body[spans[-1][1]:m.start()].strip():
                spans[-1] = (spans[-1][0], m.end(), spans[-1][2])
            continue
        if word == '?':
            if not after.startswith((')', ',', ';')):
                continue
        elif word != '::':
            if word in WORDS or word in params or before.endswith('.') or after.startswith('('):
                continue
        # Adjacent unknown identifier fragments often represent one broken name.
        if spans and not body[spans[-1][1]:m.start()].strip() and word not in ('?', '::'):
            start, _, key = spans[-1]
            spans[-1] = (start, m.end(), key + word)
        else:
            spans.append((m.start(), m.end(), word))
    if not spans or len(spans) > 4:
        return [], False
    keys = list(dict.fromkeys(span[2] for span in spans))
    choices = [sorted(params, key=lambda name: (-difflib.SequenceMatcher(None, key.lower(), name.lower()).ratio(), name)) for key in keys]
    total = 1
    for choice in choices:
        total *= len(choice)
    candidates = []
    for values in itertools.islice(itertools.product(*choices), limit):
        mapping = dict(zip(keys, values))
        result = body
        for start, end, key in reversed(spans):
            result = result[:start] + mapping[key] + result[end:]
        candidate = signature + ' ' + result
        if candidate != text and candidate not in candidates:
            candidates.append(candidate)
    return candidates, total <= limit


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    result = {r['id']: r for r in rows}
    if not rows or len(rows) != len(result):
        raise ValueError(f'Empty or duplicated IDs: {path}')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'previous', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--max-attempts', type=int, default=12)
    p.add_argument('--framework', default='net8.0')
    args = p.parse_args()
    if args.max_attempts < 1: p.error('max attempts must be positive')
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output exists')
    source, previous = load(args.input), load(args.previous)
    if set(source) != set(previous): raise SystemExit('IDs differ')
    rows, calls = [], 0
    start = time.perf_counter()
    for key, original in source.items():
        old = previous[key]['selected']
        if old['compile_status'] not in ('passed', 'failed', 'empty'):
            raise SystemExit(f'Compile status unavailable: {key}')
        passing, exhausted = [], True
        if old['compile_status'] != 'passed':
            prompt = make_signature_prompt(original['target'])
            # Use the old selected body, plus saved head candidates; no gold body in search.
            texts = [old['text']] + [a['text'] for a in previous[key]['attempts']]
            pool, seen = [], {old['text']}
            for text in texts:
                candidates, complete = generate_candidates(prompt, text, args.max_attempts)
                exhausted &= complete
                for candidate in candidates:
                    if candidate not in seen:
                        seen.add(candidate)
                        pool.append(candidate)
            if len(pool) > args.max_attempts:
                exhausted = False
            for candidate in pool[:args.max_attempts]:
                status, called = compile_or_reuse(candidate, {}, args.framework)
                calls += called
                if status == 'passed': passing.append(candidate)
        # Keep the old result unless a completed bounded search has a single compile pass.
        chosen = passing[0] if exhausted and len(passing) == 1 else old['text']
        status = 'passed' if exhausted and len(passing) == 1 else old['compile_status']
        target = original['target']
        rows.append({'id': key, 'previous': old, 'selected': chosen, 'compile_status': status,
                     'selected_exact': chosen == target, 'exhausted': exhausted,
                     'passing': [{'text': t, 'exact_for_offline_scoring': t == target} for t in passing]})
        print(f'Processed {len(rows)}/{len(source)}; new compiler calls={calls}', flush=True)
    result = {'examples': len(rows), 'previous_compiles': sum(r['previous']['compile_status']=='passed' for r in rows),
              'selected_compiles': sum(r['compile_status']=='passed' for r in rows),
              'previous_exact': sum(r['previous']['text']==source[r['id']]['target'] for r in rows),
              'selected_exact': sum(r['selected_exact'] for r in rows),
              'ambiguous': sum(len(r['passing'])>1 for r in rows), 'incomplete': sum(not r['exhausted'] for r in rows),
              'compiler_calls': calls, 'seconds': time.perf_counter()-start, 'input_sha256': sha256_file(args.input),
              'scope': 'Experimental known-target codec round trip; synthetic signature supplied from target; body used offline for scoring only.',
              'limitations': 'Heuristic is not C# scope or semantic analysis. Unique within this bounded search is not proof of correctness. Reused stress set; no generator test.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    out.with_suffix('.report.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('\n=== Parameter search results to share ===\n'+json.dumps(result, indent=2))
    for row in rows:
        if row['passing']:
            print(f"\n{row['id']}: exhausted={row['exhausted']}; passing={len(row['passing'])}")
            for candidate in row['passing']:
                print(f"  exact={candidate['exact_for_offline_scoring']}: {candidate['text']}")


if __name__ == '__main__': main()
