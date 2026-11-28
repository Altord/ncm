"""Average two frozen codecs' token logits on known-input method reconstruction."""
import argparse
import json
import time
from pathlib import Path
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from training.data import encode_text, fixed_sequence, load_base_tokenizer, sha256_file
from evaluation.godot_csharp_codec_eval import visible_text


def combine(a, b, c=None, weight_a=0.5):
    if a.shape != b.shape: raise ValueError('Codec logits differ in shape')
    if c is not None and a.shape != c.shape: raise ValueError('Codec logits differ in shape')
    if not 0 <= weight_a <= 1: raise ValueError('weight_a must be between zero and one')
    if c is not None:
        if weight_a != 0.5: raise ValueError('Weighted three-codec combination is unsupported')
        return (a.float() + b.float() + c.float()).argmax(-1)
    return (2 * weight_a * a.float() + 2 * (1 - weight_a) * b.float()).argmax(-1)


def load_rows(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    data = {r['id']: r for r in rows}
    if not rows or len(data) != len(rows): raise ValueError(f'Empty or duplicate IDs: {path}')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config-a', 'checkpoint-a', 'saved-a', 'config-b', 'checkpoint-b', 'saved-b', 'input', 'output'):
        p.add_argument('--'+name, required=True)
    for name in ('config-c', 'checkpoint-c', 'saved-c'):
        p.add_argument('--'+name)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--weight-a', type=float, default=0.5)
    args = p.parse_args()
    if any((args.config_c, args.checkpoint_c, args.saved_c)) and not all((args.config_c, args.checkpoint_c, args.saved_c)):
        p.error('config-c, checkpoint-c and saved-c must be provided together')
    if args.batch_size < 1: p.error('batch size must be positive')
    if not 0 <= args.weight_a <= 1 or (args.config_c and args.weight_a != 0.5): p.error('Invalid ensemble weight')
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output exists')
    cfg_a, cfg_b = OmegaConf.load(args.config_a), OmegaConf.load(args.config_b)
    OmegaConf.resolve(cfg_a); OmegaConf.resolve(cfg_b)
    cfg_c = OmegaConf.load(args.config_c) if args.config_c else None
    if cfg_c is not None:
        OmegaConf.resolve(cfg_c)
        if (OmegaConf.to_container(cfg_a.tokenizer, resolve=True) != OmegaConf.to_container(cfg_c.tokenizer, resolve=True) or
            int(cfg_a.model.context_length) != int(cfg_c.model.context_length) or
            int(cfg_a.vqvae.in_vocab_size) != int(cfg_c.vqvae.in_vocab_size)):
            raise SystemExit('Third codec tokenizer configuration, context or vocabulary differs')
    if (OmegaConf.to_container(cfg_a.tokenizer, resolve=True) != OmegaConf.to_container(cfg_b.tokenizer, resolve=True) or
        int(cfg_a.model.context_length) != int(cfg_b.model.context_length) or
        int(cfg_a.vqvae.in_vocab_size) != int(cfg_b.vqvae.in_vocab_size)):
        raise SystemExit('Codec tokenizer configuration, context or vocabulary differs')
    original, saved_a, saved_b = (load_rows(x) for x in (args.input, args.saved_a, args.saved_b))
    if set(original) != set(saved_a) or set(original) != set(saved_b): raise SystemExit('IDs differ')
    saved_c = load_rows(args.saved_c) if args.saved_c else None
    if saved_c is not None and set(original) != set(saved_c): raise SystemExit('Third codec IDs differ')
    sources = [(args.saved_a, args.checkpoint_a), (args.saved_b, args.checkpoint_b)]
    if saved_c is not None: sources.append((args.saved_c, args.checkpoint_c))
    for saved_path, checkpoint in sources:
        report = json.loads(Path(saved_path).with_suffix('.report.json').read_text(encoding='utf-8'))
        if report['input_sha256'] != sha256_file(args.input) or Path(report['checkpoint']).resolve() != Path(checkpoint).resolve():
            raise SystemExit(f'Saved codec report does not match input or checkpoint: {saved_path}')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    def model(cfg, path):
        config = OmegaConf.to_container(cfg.vqvae, resolve=True)
        config.pop('checkpoint_path', None)
        if config.get('autoregressive_decoder', False): raise ValueError('Parallel codec decoder required')
        result = VQVAE(**OmegaConf.create(config))
        result.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'], strict=True)
        return result.to(device).eval().requires_grad_(False)
    a, b = model(cfg_a, args.checkpoint_a), model(cfg_b, args.checkpoint_b)
    c = model(cfg_c, args.checkpoint_c) if cfg_c is not None else None
    tok = load_base_tokenizer(cfg_a.tokenizer)
    context, pad, eos = int(cfg_a.model.context_length), int(tok.pad_token_id), int(tok.eos_token_id)
    rows = list(original.values())
    output, correct, valid_total = [], 0, 0
    start = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(rows), args.batch_size):
            group = rows[begin:begin+args.batch_size]
            packed = []
            for row in group:
                key, target = row['id'], row['target']
                if saved_a[key]['target'] != target or saved_b[key]['target'] != target or (saved_c is not None and saved_c[key]['target'] != target):
                    raise SystemExit(f'Target mismatch: {key}')
                ids = encode_text(tok, target, True)
                if len(ids) > context: raise SystemExit(f'Truncated target: {key}')
                packed.append(fixed_sequence(ids, context, pad, eos))
            x = torch.tensor([r[0] for r in packed], device=device, dtype=torch.long)
            valid = torch.tensor([r[1] for r in packed], device=device, dtype=torch.bool)
            # Each codec independently encodes the known target; no reference labels choose outputs.
            logits_a = a.decode(a.encode(x)).float()
            logits_b = b.decode(b.encode(x)).float()
            pred_a, pred_b = logits_a.argmax(-1), logits_b.argmax(-1)
            logits_c = c.decode(c.encode(x)).float() if c is not None else None
            combined = combine(logits_a, logits_b, logits_c, args.weight_a)
            correct += int(((combined == x) & valid).sum()); valid_total += int(valid.sum())
            for i, row in enumerate(group):
                key, target = row['id'], row['target']
                before_a = visible_text(tok, pred_a[i].tolist(), eos, pad)
                before_b = visible_text(tok, pred_b[i].tolist(), eos, pad)
                if before_a != saved_a[key]['reconstructed'] or before_b != saved_b[key]['reconstructed']:
                    raise SystemExit(f'Recomputed codec output differs from saved result: {key}')
                if logits_c is not None and visible_text(tok, logits_c[i].argmax(-1).tolist(), eos, pad) != saved_c[key]['reconstructed']:
                    raise SystemExit(f'Recomputed third codec output differs from saved result: {key}')
                selected = visible_text(tok, combined[i].tolist(), eos, pad)
                output.append({'id': key, 'baseline_a_exact': before_a == target, 'baseline_b_exact': before_b == target,
                               'ensemble_exact': selected == target, 'selected': selected})
            print(f'Processed {min(begin+len(group),len(rows))}/{len(rows)}', flush=True)
    report = {'examples': len(output), 'token_accuracy': correct/valid_total,
              'baseline_a_exact': sum(r['baseline_a_exact'] for r in output),
              'baseline_b_exact': sum(r['baseline_b_exact'] for r in output),
              'ensemble_exact': sum(r['ensemble_exact'] for r in output),
              'new_exact_vs_a': sum(not r['baseline_a_exact'] and r['ensemble_exact'] for r in output),
              'lost_exact_vs_a': sum(r['baseline_a_exact'] and not r['ensemble_exact'] for r in output),
              'input_sha256': sha256_file(args.input), 'checkpoint_a_sha256': sha256_file(args.checkpoint_a),
              'checkpoint_b_sha256': sha256_file(args.checkpoint_b), 'elapsed_seconds': time.perf_counter()-start,
              'weight_a': args.weight_a if c is None else None,
              'scope': 'Weighted average of logits from two separately encoded known-target frozen codecs; targets score only, not choose predictions.',
              'limitations': 'Two codecs require separate latent streams and computation; not a generator test. Reused dataset, no compilation or behavior test. Matching tokenizer configuration does not verify tokenizer file identity.'}
    if c is not None:
        report['checkpoint_c_sha256'] = sha256_file(args.checkpoint_c)
        report['scope'] = 'Equal-weight average of three separately encoded known-target frozen codecs; targets score only, not choose predictions.'
        report['limitations'] = 'Three codecs require separate latent streams and computation; not a generator test. Reused dataset, no compilation or behavior test. Matching tokenizer configuration does not verify tokenizer file identity.'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in output: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f'\n=== {"Three" if c is not None else "Two"}-codec ensemble results to share ===\n'+json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
