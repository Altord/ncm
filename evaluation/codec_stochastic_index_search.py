"""Search stochastic VQ assignments at the encoder and save one code stream."""
import argparse
import json
import time
from pathlib import Path
import torch
from evaluation.codec_latent_refiner_screen import load_codec
from evaluation.ensemble_codec_methods import load_rows
from evaluation.godot_csharp_codec_eval import visible_text
from training.data import encode_text, fixed_sequence, load_base_tokenizer, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'input', 'baseline', 'output'): p.add_argument('--'+name, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-examples', type=int, default=0); p.add_argument('--samples', type=int, default=4)
    p.add_argument('--temperature', type=float, default=0.05); p.add_argument('--top-k', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(); out = Path(args.output); codes_path = out.with_suffix('.codes.pt')
    if min(args.batch_size, args.samples, args.top_k) < 1 or args.temperature <= 0: p.error('Invalid sampling settings')
    if any(path.exists() for path in (out, out.with_suffix('.report.json'), codes_path)): raise SystemExit('Output exists')
    source, baseline = load_rows(args.input), load_rows(args.baseline)
    meta = json.loads(Path(args.baseline).with_suffix('.report.json').read_text())
    if (source.keys() != baseline.keys() or meta['input_sha256'] != sha256_file(args.input)
        or Path(meta['checkpoint']).resolve() != Path(args.checkpoint).resolve()): raise SystemExit('Baseline provenance differs')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    torch.manual_seed(args.seed)
    cfg, model = load_codec(args.config, args.checkpoint, device)
    tok = load_base_tokenizer(cfg.tokenizer); context = int(cfg.model.context_length)
    eos, pad = int(tok.eos_token_id), int(tok.pad_token_id)
    rows = list(source.values()); rows = rows[:args.max_examples] if args.max_examples else rows
    results, codes = [], []; before_total = after_total = improved = 0
    if device.type == 'cuda': torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            group = rows[start:start+args.batch_size]; packed = []
            for row in group:
                if baseline[row['id']]['target'] != row['target']: raise SystemExit('Target differs')
                ids = encode_text(tok, row['target'], True)
                if len(ids) > context: raise SystemExit('Input truncated')
                packed.append(fixed_sequence(ids, context, pad, eos))
            x = torch.tensor([r[0] for r in packed], device=device, dtype=torch.long)
            mask = torch.tensor([r[1] for r in packed], device=device, dtype=torch.bool)
            z = model.encode_z(x); model.vq.set_stochastic_encode(False)
            base_indices = model.vq.z_to_idx(z)
            selected = base_indices.clone()
            base_pred = model.decode(base_indices).argmax(-1)
            pred = base_pred.clone(); base_errors = ((base_pred != x) & mask).sum(-1); best = base_errors.clone()
            for _ in range(args.samples):
                model.vq.set_stochastic_encode(True, temperature=args.temperature, top_k=args.top_k, top_p=1.)
                proposal = model.vq.z_to_idx(z)
                candidate = model.decode(proposal).argmax(-1)
                errors = ((candidate != x) & mask).sum(-1)
                better = errors < best
                best = torch.where(better, errors, best)
                pred = torch.where(better[:, None], candidate, pred)
                selected = torch.where(better[:, None], proposal, selected)
            model.vq.set_stochastic_encode(False)
            verified = model.decode(selected).argmax(-1)
            if not torch.equal(verified, pred): raise SystemExit('Selected indices did not decode to selected prediction')
            before_total += int(base_errors.sum()); after_total += int(best.sum()); improved += int((best < base_errors).sum())
            for i, row in enumerate(group):
                before = visible_text(tok, base_pred[i].tolist(), eos, pad)
                after = visible_text(tok, pred[i].tolist(), eos, pad)
                if before != baseline[row['id']]['reconstructed']: raise SystemExit('Baseline mismatch')
                results.append({'id': row['id'], 'baseline_errors': int(base_errors[i]), 'selected_errors': int(best[i]),
                                'baseline_exact': before == row['target'], 'selected_exact': after == row['target'], 'selected': after})
                codes.append({'id': row['id'], 'indices': selected[i].cpu(), 'valid_length': int(mask[i].sum())})
            print(f'Processed {min(start+len(group),len(rows))}/{len(rows)}; improved={improved}', flush=True)
    if device.type == 'cuda': torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-started
    summary = {'examples': len(rows), 'baseline_exact': sum(r['baseline_exact'] for r in results),
               'selected_exact': sum(r['selected_exact'] for r in results),
               'gained_exact': sum(not r['baseline_exact'] and r['selected_exact'] for r in results),
               'lost_exact': sum(r['baseline_exact'] and not r['selected_exact'] for r in results),
               'baseline_token_errors': before_total, 'selected_token_errors': after_total,
               'improved_examples': improved, 'samples': args.samples, 'temperature': args.temperature, 'top_k': args.top_k,
               'seed': args.seed, 'seconds': elapsed, 'input_sha256': sha256_file(args.input),
               'codec_sha256': sha256_file(args.checkpoint),
               'scope': 'Known-input encoder selects one discrete VQ code stream by original token error count; selected indices independently decoded and verified.',
               'limitations': 'Requires original tokens at encoding, extra proposal encodes and decodes; no generator, compile, packed size or end-to-end benchmark. Reused datasets.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in results: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    torch.save(codes, codes_path)
    with out.with_suffix('.report.json').open('x', encoding='utf-8') as f: json.dump(summary, f, indent=2); f.write('\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
