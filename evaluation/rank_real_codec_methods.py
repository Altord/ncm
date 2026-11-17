"""Rank parameter-search candidates from saved real codec reconstructions."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.evaluate_pointer_stress import make_signature_prompt
from evaluation.rank_codec_candidates_lm import candidate_score
from evaluation.signature_parameter_search import generate_candidates
from training.data import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='deepseek-ai/deepseek-coder-1.3b-base')
    p.add_argument('--max-examples', type=int, default=100)
    p.add_argument('--gap', type=float, default=0.05)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.max_examples < 1 or args.gap < 0: p.error('Invalid sample size or gap')
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output exists')
    rows = [json.loads(s) for s in Path(args.input).read_text(encoding='utf-8').splitlines() if s.strip()]
    if not rows or any(not r.get('id') or not isinstance(r.get('target'), str) or not isinstance(r.get('reconstructed'), str) for r in rows) or len({r['id'] for r in rows}) != len(rows):
        raise SystemExit('Input is empty or lacks unique IDs, target, or reconstructed text')
    # Fixed hash order; no filtering based on whether the original was reconstructed correctly.
    rows.sort(key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
    rows = rows[:args.max_examples]
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32)
    model.to(device).eval()
    results, start = [], time.perf_counter()
    for row in rows:
        target, before = row['target'], row['reconstructed']
        try:
            prompt = make_signature_prompt(target)
        except ValueError:
            prompt = None
        candidates, exhausted = generate_candidates(prompt, before, 12, conservative=True) if prompt else ([], False)
        # Always include baseline as a choice; deduplicate without consulting target.
        candidates = list(dict.fromkeys([before] + candidates))
        chosen, gap, accepted = before, None, False
        if prompt and len(candidates) >= 2:
            scores = [candidate_score(model, tokenizer, prompt, c, device) for c in candidates]
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            gap = scores[order[0]] - scores[order[1]]
            if gap >= args.gap and order[0] != 0:
                chosen, accepted = candidates[order[0]], True
        results.append({'id': row['id'], 'baseline': before, 'selected': chosen, 'candidate_count': len(candidates),
            'search_exhausted': exhausted, 'gap': gap, 'accepted': accepted,
            'before_exact': before == target, 'after_exact': chosen == target,
            'target_in_candidates_offline': target in candidates})
        if len(results) % 10 == 0 or len(results) == len(rows):
            print(f'Processed {len(results)}/{len(rows)}', flush=True)
    report = {'examples': len(results), 'minimum_gap': args.gap,
        'baseline_exact': sum(r['before_exact'] for r in results), 'selected_exact': sum(r['after_exact'] for r in results),
        'new_exact': sum(not r['before_exact'] and r['after_exact'] for r in results),
        'lost_exact': sum(r['before_exact'] and not r['after_exact'] for r in results),
        'accepted_changes': sum(r['accepted'] for r in results),
        'examples_with_alternatives': sum(r['candidate_count'] > 1 for r in results),
        'target_in_candidates_offline': sum(r['target_in_candidates_offline'] for r in results),
        'seconds': time.perf_counter()-start, 'input_sha256': sha256_file(args.input), 'model': args.model,
        'scope': 'Saved known-target codec reconstructions; target-derived synthetic signature supplied; conservative search skips clean syntax and bodies with comments/literals; target body used for offline scoring only.',
        'limitations': 'Threshold was chosen after seeing stress results. Method validation set has been repeatedly used; sample is not an independent test. No compilation or behavioral tests. Uses a separate pretrained LLM.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('\n=== Real codec method ranking results to share ===\n'+json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
