"""Bounded encoder-side VQ index search for lossless-input reconstruction."""
import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from training.data import encode_text, fixed_sequence, load_base_tokenizer, sha256_file
from evaluation.ensemble_codec_methods import load_rows
from evaluation.godot_csharp_codec_eval import visible_text


def propose(model, codes, x, mask, budget):
    """Use the final codebook's embedding gradient to propose single-index changes."""
    vq = model.vq
    length = int(vq.levels[-1]); start = sum(int(n) for n in vq.levels[:-1])
    current = codes[:, start:start + length]
    if codes.shape[0] != 1 or current.shape[1] != length: raise ValueError('Unexpected VQ index layout')
    with torch.no_grad():
        base = vq.idx_to_z_q(codes).detach()
        values = vq.embeddings[-1](current).detach()
    values.requires_grad_(True)
    transformed = vq._apply_quant_resi(values, len(vq.levels) - 1)
    logits = model.decoder(base - transformed.detach() + transformed)
    per_token = F.cross_entropy(logits.float().transpose(1, 2), x, reduction='none')
    loss = (per_token * mask).sum() / mask.sum().clamp_min(1)
    gradient = torch.autograd.grad(loss, values)[0][0].float()
    embeddings = vq.embeddings[-1].weight.detach().float()
    scores = gradient @ embeddings.t() - (gradient * values.detach()[0].float()).sum(-1, keepdim=True)
    scores.scatter_(1, current[0].unsqueeze(1), float('inf'))
    count = min(budget, scores.numel() - length)
    choices = scores.flatten().topk(count, largest=False).indices.tolist()
    proposals = []
    for index in choices:
        position, token = divmod(index, embeddings.shape[0])
        trial = codes.clone(); trial[0, start + position] = token
        proposals.append(trial)
    return proposals


def score(logits, x, mask):
    expanded = x.expand(logits.shape[0], -1)
    valid = mask.expand(logits.shape[0], -1)
    errors = ((logits.argmax(-1) != expanded) & valid).sum(-1)
    ce = F.cross_entropy(logits.float().transpose(1, 2), expanded, reduction='none')
    return errors, (ce * valid).sum(-1) / valid.sum(-1).clamp_min(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'input', 'baseline', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--max-examples', type=int, default=64)
    p.add_argument('--budget', type=int, default=4)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.max_examples < 1 or args.budget < 1: p.error('Example and proposal budgets must be positive')
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output exists')
    cfg = OmegaConf.load(args.config); OmegaConf.resolve(cfg)
    original, baseline = load_rows(args.input), load_rows(args.baseline)
    report = json.loads(Path(args.baseline).with_suffix('.report.json').read_text(encoding='utf-8'))
    if (original.keys() != baseline.keys() or report['input_sha256'] != sha256_file(args.input) or
        Path(report['checkpoint']).resolve() != Path(args.checkpoint).resolve()):
        raise SystemExit('Baseline input, IDs or checkpoint do not match')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    tok = load_base_tokenizer(cfg.tokenizer)
    pad, eos, context = int(tok.pad_token_id), int(tok.eos_token_id), int(cfg.model.context_length)
    vcfg = OmegaConf.to_container(cfg.vqvae, resolve=True); vcfg.pop('checkpoint_path', None)
    if vcfg.get('autoregressive_decoder', False): raise SystemExit('Parallel decoder required')
    model = VQVAE(**OmegaConf.create(vcfg))
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    model.to(device).eval().requires_grad_(False)
    rows, output, attempts = list(original.values())[:args.max_examples], [], 0
    start_time = time.perf_counter()
    for i, row in enumerate(rows, 1):
        key, target = row['id'], row['target']
        if baseline[key]['target'] != target: raise SystemExit(f'Target mismatch: {key}')
        ids = encode_text(tok, target, append_eos=True)
        if len(ids) > context: raise SystemExit(f'Truncated input: {key}')
        packed, valid = fixed_sequence(ids, context, pad, eos)
        x = torch.tensor([packed], dtype=torch.long, device=device)
        mask = torch.tensor([valid], dtype=torch.bool, device=device)
        with torch.no_grad():
            codes = model.encode(x)
            logits = model.decode(codes)
            errors, ce = score(logits, x, mask)
            before = visible_text(tok, logits[0].argmax(-1).tolist(), eos, pad)
        if before != baseline[key]['reconstructed']: raise SystemExit(f'Baseline reconstruction differs: {key}')
        selected_codes = codes
        selected = before
        old_errors = int(errors[0]); new_errors = old_errors
        if before != target and old_errors:
            proposals = propose(model, codes, x, mask, args.budget)
            attempts += len(proposals)
            if proposals:
                with torch.no_grad():
                    trial_codes = torch.cat(proposals, dim=0)
                    trial_logits = model.decode(trial_codes)
                    trial_errors, trial_ce = score(trial_logits, x, mask)
                    order = sorted(range(len(proposals)), key=lambda j: (int(trial_errors[j]), float(trial_ce[j])))
                    best = order[0]
                    if (int(trial_errors[best]), float(trial_ce[best])) < (old_errors, float(ce[0])):
                        selected_codes = trial_codes[best:best+1]
                        selected = visible_text(tok, trial_logits[best].argmax(-1).tolist(), eos, pad)
                        new_errors = int(trial_errors[best])
        # This representation is the complete discrete code stream, requiring no target at decode.
        output.append({'id': key, 'baseline_exact': before == target, 'selected_exact': selected == target,
                       'baseline_token_errors': old_errors, 'selected_token_errors': new_errors,
                       'selected': selected, 'codes': selected_codes[0].tolist()})
        if i % 16 == 0 or i == len(rows): print(f'Processed {i}/{len(rows)}; proposals evaluated={attempts}', flush=True)
    summary = {'examples': len(output), 'baseline_exact': sum(r['baseline_exact'] for r in output),
               'selected_exact': sum(r['selected_exact'] for r in output),
               'gained_exact': sum(not r['baseline_exact'] and r['selected_exact'] for r in output),
               'lost_exact': sum(r['baseline_exact'] and not r['selected_exact'] for r in output),
               'baseline_token_errors': sum(r['baseline_token_errors'] for r in output),
               'selected_token_errors': sum(r['selected_token_errors'] for r in output),
               'attempts': attempts, 'budget_per_example': args.budget, 'seconds': time.perf_counter()-start_time,
               'input_sha256': sha256_file(args.input), 'checkpoint_sha256': sha256_file(args.checkpoint),
               'scope': 'Encoder-side discrete VQ index search uses the known original and decoder predictions to choose codes. Decoder needs only selected VQ indices.',
               'limitations': 'Fixed first-N sample from repeatedly used data; no generation, compilation, behavior, packing, or full production timing. Requires additional encoder-side decoder and gradient work; does not reduce index count. The known input is necessary for selection; not applicable to unknown generated targets.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in output: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    with out.with_suffix('.report.json').open('x', encoding='utf-8') as f: json.dump(summary, f, indent=2); f.write('\n')
    print('\n=== Encoder-side index-search screen ===\n'+json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
