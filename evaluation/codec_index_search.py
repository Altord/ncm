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


def propose_all_levels(model, codes, x, mask, budget):
    """Rank individual code substitutions by their decoder-loss gradients."""
    if codes.shape[0] != 1: raise ValueError('Expected one example')
    vq = model.vq
    length = int(vq.levels[-1])
    with torch.no_grad(): base = vq.idx_to_z_q(codes).detach()
    values, transformed, starts = [], [], []
    offset = 0
    for level, count in enumerate(vq.levels):
        starts.append(offset); count = int(count)
        current = codes[:, offset:offset+count]; offset += count
        value = vq.embeddings[level](current).detach().requires_grad_(True)
        full = vq._up_to_full(value, level, length) if level != len(vq.levels)-1 else value
        values.append(value); transformed.append(vq._apply_quant_resi(full, level))
    if offset != codes.shape[1]: raise ValueError('Unexpected VQ index layout')
    latent = base
    for t in transformed: latent = latent + (t - t.detach())
    logits = model.decoder(latent)
    ce = F.cross_entropy(logits.float().transpose(1, 2), x, reduction='none')
    loss = (ce * mask).sum() / mask.sum().clamp_min(1)
    gradients = torch.autograd.grad(loss, values)
    choices = []
    with torch.no_grad():
        for level, (value, gradient, start) in enumerate(zip(values, gradients, starts)):
            emb = vq.embeddings[level].weight.detach().float()
            scores = gradient[0].float() @ emb.t() - (gradient[0].float() * value[0].float()).sum(-1, keepdim=True)
            current = codes[0, start:start+scores.shape[0]]
            scores.scatter_(1, current.unsqueeze(1), float('inf'))
            n = min(budget, scores.numel() - scores.shape[0])
            if n:
                vals, flat = scores.flatten().topk(n, largest=False)
                choices.extend((float(v), level, int(i), emb.shape[0]) for v, i in zip(vals, flat))
    choices.sort(key=lambda t: t[0])
    proposals = []
    for _, level, flat, vocab in choices[:budget]:
        position, token = divmod(flat, vocab)
        trial = codes.clone(); trial[0, starts[level]+position] = token
        proposals.append(trial)
    return proposals


def score(logits, x, mask):
    expanded = x.expand(logits.shape[0], -1)
    valid = mask.expand(logits.shape[0], -1)
    errors = ((logits.argmax(-1) != expanded) & valid).sum(-1)
    ce = F.cross_entropy(logits.float().transpose(1, 2), expanded, reduction='none')
    return errors, (ce * valid).sum(-1) / valid.sum(-1).clamp_min(1)


def is_improvement(new_errors, new_ce, old_errors, old_ce):
    """Prefer fewer token errors, breaking ties using cross-entropy."""
    return (int(new_errors), float(new_ce)) < (int(old_errors), float(old_ce))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'input', 'baseline', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--max-examples', type=int, default=64)
    p.add_argument('--budget', type=int, default=4)
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--initial-search', help='Stochastic-search JSONL with matching .report.json and .codes.pt')
    p.add_argument('--all-levels', action='store_true', help='Rank substitutions across every VQ level')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.max_examples < 1 or args.budget < 1 or args.rounds < 1: p.error('Example, proposal and round budgets must be positive')
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
    initial = initial_codes = None
    if args.initial_search:
        initial_path = Path(args.initial_search)
        initial_report = json.loads(initial_path.with_suffix('.report.json').read_text(encoding='utf-8'))
        if initial_report['input_sha256'] != sha256_file(args.input) or initial_report['codec_sha256'] != sha256_file(args.checkpoint):
            raise SystemExit('Initial search input or checkpoint differs')
        initial = load_rows(initial_path)
        code_rows = torch.load(initial_path.with_suffix('.codes.pt'), map_location='cpu', weights_only=True)
        initial_codes = {r['id']: r for r in code_rows}
        if initial.keys() != initial_codes.keys(): raise SystemExit('Initial search code IDs differ')
    rows, output, attempts, accepted_total = list(original.values())[:args.max_examples], [], 0, 0
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
        current_ce = float(ce[0]); accepted = 0
        if initial is not None:
            if key not in initial: raise SystemExit(f'Initial search missing: {key}')
            selected_codes = initial_codes[key]['indices'].to(device).long().unsqueeze(0)
            if int(initial_codes[key]['valid_length']) != int(mask.sum()): raise SystemExit('Initial valid length differs')
            with torch.no_grad():
                initial_logits = model.decode(selected_codes)
                initial_errors, initial_ce = score(initial_logits, x, mask)
                selected = visible_text(tok, initial_logits[0].argmax(-1).tolist(), eos, pad)
            if selected != initial[key]['selected'] or int(initial_errors[0]) != initial[key]['selected_errors']:
                raise SystemExit(f'Initial search index round trip differs: {key}')
            new_errors, current_ce = int(initial_errors[0]), float(initial_ce[0])
        for _ in range(args.rounds):
            if selected == target or new_errors == 0: break
            proposals = (propose_all_levels if args.all_levels else propose)(model, selected_codes, x, mask, args.budget)
            attempts += len(proposals)
            if not proposals: break
            with torch.no_grad():
                trial_codes = torch.cat(proposals, dim=0)
                trial_logits = model.decode(trial_codes)
                trial_errors, trial_ce = score(trial_logits, x, mask)
                order = sorted(range(len(proposals)), key=lambda j: (int(trial_errors[j]), float(trial_ce[j])))
                best = order[0]
                if not is_improvement(trial_errors[best], trial_ce[best], new_errors, current_ce): break
                selected_codes = trial_codes[best:best+1]
                selected = visible_text(tok, trial_logits[best].argmax(-1).tolist(), eos, pad)
                new_errors, current_ce = int(trial_errors[best]), float(trial_ce[best])
                accepted += 1
        accepted_total += accepted
        # This representation is the complete discrete code stream, requiring no target at decode.
        output.append({'id': key, 'baseline_exact': before == target, 'selected_exact': selected == target,
                       'baseline_token_errors': old_errors, 'selected_token_errors': new_errors,
                       'selected': selected, 'accepted_changes': accepted, 'codes': selected_codes[0].tolist()})
        if i % 16 == 0 or i == len(rows): print(f'Processed {i}/{len(rows)}; proposals evaluated={attempts}', flush=True)
    summary = {'examples': len(output), 'baseline_exact': sum(r['baseline_exact'] for r in output),
               'selected_exact': sum(r['selected_exact'] for r in output),
               'gained_exact': sum(not r['baseline_exact'] and r['selected_exact'] for r in output),
               'lost_exact': sum(r['baseline_exact'] and not r['selected_exact'] for r in output),
               'baseline_token_errors': sum(r['baseline_token_errors'] for r in output),
               'selected_token_errors': sum(r['selected_token_errors'] for r in output),
               'attempts': attempts, 'budget_per_round': args.budget, 'max_rounds': args.rounds,
               'accepted_changes': accepted_total, 'seconds': time.perf_counter()-start_time,
               'initial_search_sha256': sha256_file(args.initial_search) if args.initial_search else None,
               'all_levels': args.all_levels,
               'input_sha256': sha256_file(args.input), 'checkpoint_sha256': sha256_file(args.checkpoint),
               'codec_sha256': sha256_file(args.checkpoint),
               'scope': 'Encoder-side discrete VQ index search uses the known original and decoder predictions to choose codes. Decoder needs only selected VQ indices.',
               'limitations': 'Fixed first-N sample from repeatedly used data; no generation, compilation, behavior, packing, or full production timing. Requires additional encoder-side decoder and gradient work; does not reduce index count. The known input is necessary for selection; not applicable to unknown generated targets.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in output: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    with out.with_suffix('.report.json').open('x', encoding='utf-8') as f: json.dump(summary, f, indent=2); f.write('\n')
    print('\n=== Encoder-side index-search screen ===\n'+json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
