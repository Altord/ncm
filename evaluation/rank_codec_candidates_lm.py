"""Rank saved compiling codec candidates with a pretrained causal code model."""
import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.evaluate_pointer_stress import make_signature_prompt
from training.data import sha256_file


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    result = {r['id']: r for r in rows}
    if not rows or len(result) != len(rows): raise ValueError(f'Empty or duplicate IDs: {path}')
    return result


def candidate_score(model, tokenizer, prompt, candidate, device):
    prefix = prompt + '\n'
    # Separate encoding avoids allowing a token to cross the prompt/body boundary.
    pre = tokenizer.encode(prefix, add_special_tokens=False)
    continuation = tokenizer.encode(candidate, add_special_tokens=False)
    if not pre or not continuation: raise ValueError('Empty prompt or candidate tokenization')
    ids = torch.tensor([pre + continuation], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids).logits[:, len(pre)-1:-1, :].float()
        labels = ids[:, len(pre):]
        logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return float(logp.mean().cpu())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'candidates', 'output'): p.add_argument('--'+name, required=True)
    p.add_argument('--model', default='deepseek-ai/deepseek-coder-1.3b-base')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output already exists')
    source, data = load(args.input), load(args.candidates)
    if set(source) != set(data): raise SystemExit('Snippet IDs differ')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    print(f'Loading {args.model}; model weights may download on first use.', flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32)
    model.to(device).eval()
    rows, start = [], time.perf_counter()
    for key, row in data.items():
        if row['previous']['compile_status'] == 'passed': continue
        target = source[key]['target']
        prompt = make_signature_prompt(target)
        candidates = list(dict.fromkeys(c['text'] for c in row['passing']))
        scores = [candidate_score(model, tok, prompt, candidate, device) for candidate in candidates]
        # Stable ties preserve the pre-existing candidate order.
        chosen_index = max(range(len(candidates)), key=lambda i: scores[i]) if candidates else None
        chosen = candidates[chosen_index] if chosen_index is not None else row['previous']['text']
        rows.append({'id': key, 'candidates': [{'text': t, 'mean_log_probability': s} for t, s in zip(candidates, scores)],
                     'chosen': chosen, 'chosen_index': chosen_index, 'chosen_exact_offline': chosen == target,
                     'chosen_whitespace_stripped_match_offline': ''.join(chosen.split()) == ''.join(target.split())})
        print(f'Ranked {len(rows)} cases; latest={key}, candidates={len(candidates)}', flush=True)
    report = {'examples_ranked': len(rows), 'exact': sum(r['chosen_exact_offline'] for r in rows),
              'whitespace_stripped_matches': sum(r['chosen_whitespace_stripped_match_offline'] for r in rows),
              'model': args.model, 'input_sha256': sha256_file(args.input), 'candidates_sha256': sha256_file(args.candidates),
              'elapsed_seconds': time.perf_counter()-start,
              'scope': 'Zero-shot pretrained LLM ranker; synthetic target signature supplied; known-target codec; candidate selection uses mean conditional log probability only.',
              'limitations': 'Five repeatedly inspected examples, not an independent test. Whitespace stripping can change meaning inside literals. This uses a separate pretrained LLM and is not NCM-only or NCM generator quality.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('\n=== Pretrained candidate ranker results to share ===\n'+json.dumps(report, indent=2))
    for row in rows:
        print(f"{row['id']}: chosen index={row['chosen_index']}, exact={row['chosen_exact_offline']}, whitespace_match={row['chosen_whitespace_stripped_match_offline']}: {row['chosen']}")


if __name__ == '__main__': main()
