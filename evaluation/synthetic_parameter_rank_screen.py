"""Screen pretrained ranking on synthetic same-type parameter substitutions."""
import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.codec_prompt_signature_copy import parse_method, WRAPPER
from evaluation.evaluate_pointer_stress import make_signature_prompt
from evaluation.rank_codec_candidates_lm import candidate_score
from training.csharp_structure import csharp_parser, _walk
from training.data import sha256_file


def build_case(row):
    text = row['target']
    info = parse_method(text, require_clean=True)
    if info is None: return None
    raw = text.encode('utf-8')
    signature = raw[info['start']:info['body']].decode('utf-8').strip()
    match = re.search(r'\(([^()]*)\)\s*$', signature)
    if not match: return None
    params = []
    for part in match.group(1).split(','):
        m = re.fullmatch(r'\s*([A-Za-z_]\w*(?:\[\])?)\s+([A-Za-z_]\w*)\s*', part)
        if not m: return None
        params.append((m.group(1), m.group(2)))
    source = (WRAPPER + text + '\n}').encode('utf-8')
    tree = csharp_parser().parse(source)
    offset = len(WRAPPER.encode('utf-8'))
    for node in _walk(tree.root_node):
        if node.type != 'identifier' or node.start_byte-offset < info['body']: continue
        name = source[node.start_byte:node.end_byte].decode('utf-8')
        if node.parent and node.parent.type in ('member_access_expression', 'member_binding_expression') and node == node.parent.child_by_field_name('name'):
            continue
        types = [t for t, n in params if n == name]
        if len(types) != 1: continue
        choices = [n for t, n in params if t == types[0] and n != name]
        if not choices: continue
        start, end = node.start_byte-offset, node.end_byte-offset
        candidates = [text] + [(raw[:start] + other.encode('utf-8') + raw[end:]).decode('utf-8') for other in choices]
        return list(dict.fromkeys(candidates))
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='deepseek-ai/deepseek-coder-1.3b-base')
    p.add_argument('--max-examples', type=int, default=80)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output exists')
    rows = [json.loads(s) for s in Path(args.input).read_text(encoding='utf-8').splitlines() if s.strip()]
    if not rows or any(not r.get('id') or not r.get('source') for r in rows) or len({r['id'] for r in rows}) != len(rows):
        raise SystemExit('Missing or duplicate IDs/source paths')
    # Group by file and keep one method per source to reduce correlated observations.
    rows.sort(key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
    used, cases = set(), []
    for row in rows:
        if row['source'] in used: continue
        candidates = build_case(row)
        if candidates and len(candidates) >= 2:
            random.Random(int(hashlib.sha256(row['id'].encode()).hexdigest(), 16)).shuffle(candidates)
            cases.append((row, candidates))
            used.add(row['source'])
        if len(cases) >= args.max_examples: break
    if len(cases) < 2: raise SystemExit('Fewer than two eligible source files')
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32)
    model.to(device).eval()
    results, start = [], time.perf_counter()
    for row, candidates in cases:
        prompt = make_signature_prompt(row['target'])
        scores = [candidate_score(model, tok, prompt, candidate, device) for candidate in candidates]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        chosen = candidates[order[0]]
        results.append({'id': row['id'], 'source': row['source'], 'candidate_count': len(candidates),
                        'first_correct': candidates[0] == row['target'], 'ranker_correct': chosen == row['target'],
                        'gap': scores[order[0]]-scores[order[1]], 'chosen': chosen})
        if len(results) % 10 == 0 or len(results) == len(cases):
            print(f'Ranked {len(results)}/{len(cases)} synthetic cases', flush=True)
    thresholds = (0.0, 0.02, 0.05, 0.1)
    summary = []
    for threshold in thresholds:
        selected = [r for r in results if r['gap'] >= threshold]
        summary.append({'minimum_gap': threshold, 'accepted': len(selected),
                        'correct': sum(r['ranker_correct'] for r in selected),
                        'accuracy': sum(r['ranker_correct'] for r in selected)/len(selected) if selected else None})
    report = {'examples': len(results), 'unique_source_files': len(used), 'input_sha256': sha256_file(args.input),
              'model': args.model, 'first_candidate_correct': sum(r['first_correct'] for r in results),
              'ranker_correct': sum(r['ranker_correct'] for r in results), 'gap_screen': summary,
              'seconds': time.perf_counter()-start,
              'scope': 'Synthetic one-token parameter substitution: original target is explicitly included in candidate set; no codec reconstruction or compile filtering.',
              'limitations': 'Validation corpus was used in earlier development, is not a fresh independent test; source-path grouping does not rule out duplicates. Same-type substitutions can be behaviorally equivalent. Thresholds are descriptive, not validated deployment cutoffs. Uses a separate LLM.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in results: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('\n=== Synthetic parameter ranking results to share ===\n'+json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
