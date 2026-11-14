"""Calibrate a conservative copy policy on pointer-training rows, evaluate on validation.

Thresholds are selected without validation labels. This is still a known-target
VQ codec diagnostic; the target was encoded into the latent representation.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import sys
import time

import torch
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.vqvae import VQVAE
from models.prompt_copy_head import PromptCopyHead
from training.data import load_base_tokenizer, sha256_file
from training.train_prompt_copy_screen import read_records, batch, frozen_codec
from evaluation.godot_csharp_codec_eval import visible_text
from evaluation.compare_codec_method_names import method_name

MARGINS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MASSES = (0.01, 0.03, 0.05, 0.1, 0.2, 0.4)


def apply_policy(base, copy, margin, mass, policy):
    if policy is None:
        return base.clone()
    # No target-derived validity mask here: output length is unknown at inference.
    choose = (margin <= policy['max_margin']) & (mass >= policy['min_attention_mass'])
    return torch.where(choose, copy, base)


def choose_policy(batches, min_fixes=3):
    """Select on calibration targets only; do nothing unless training improves."""
    def cat(key):
        return torch.cat([b[key].reshape(-1) for b in batches])
    target, valid, base, copy, margin, mass = (cat(k) for k in ('target', 'valid', 'base', 'copy', 'margin', 'mass'))
    winner = None
    best_key = (0, 0)
    for limit in MARGINS:
        for minimum in MASSES:
            policy = {'max_margin': limit, 'min_attention_mass': minimum}
            selected = apply_policy(base, copy, margin, mass, policy)
            fixed = int(((base != target) & (selected == target) & valid).sum())
            regressed = int(((base == target) & (selected != target) & valid).sum())
            changed = int(((selected != base) & valid).sum())
            net = fixed - regressed
            rank = (net, -changed)
            if fixed >= min_fixes and rank > best_key:
                winner = {**policy, 'calibration_fixed': fixed, 'calibration_regressed': regressed,
                          'calibration_changed': changed, 'calibration_net_correct': net}
                best_key = rank
    return winner


@torch.inference_mode()
def collect(codec, head, rows, device, batch_size, label):
    batches = []
    for begin in range(0, len(rows), batch_size):
        group = rows[begin:begin + batch_size]
        target, valid, prompt, pmask = batch(group, device)
        features, logits, embeddings = frozen_codec(codec, target, prompt)
        attention, gate = head.distribution(features, embeddings, pmask, logits, prompt)
        top = logits.float().topk(2, dim=-1)
        base = logits.argmax(dim=-1)
        margin = top.values[..., 0] - top.values[..., 1]
        # Sum attention for duplicate prompt token IDs before choosing a token.
        copy_mass = torch.zeros_like(logits, dtype=torch.float32)
        copy_mass.scatter_add_(-1, prompt[:, None, :].expand(-1, target.size(1), -1), attention)
        mass, copy = copy_mass.max(-1)
        mix = logits.float().softmax(-1) * (1 - gate.sigmoid().unsqueeze(-1))
        mix.add_(copy_mass * gate.sigmoid().unsqueeze(-1))
        pointer = mix.argmax(-1)
        batches.append({'target': target.cpu(), 'valid': valid.cpu(), 'base': base.cpu(),
                        'copy': copy.cpu(), 'margin': margin.cpu(), 'mass': mass.cpu(),
                        'pointer': pointer.cpu(), 'rows': group})
        if (begin // batch_size + 1) % 16 == 0 or begin + len(group) == len(rows):
            print(f'{label} {min(begin + len(group), len(rows))}/{len(rows)}', flush=True)
    return batches


def evaluate(batches, policy, tokenizer):
    counts = {'valid_tokens': 0, 'base_correct': 0, 'pointer_correct': 0, 'policy_correct': 0,
              'base_exact_text': 0, 'pointer_exact_text': 0, 'policy_exact_text': 0,
              'base_exact_names': 0, 'pointer_exact_names': 0, 'policy_exact_names': 0,
              'scorable_names': 0, 'policy_fixed': 0, 'policy_regressed': 0, 'policy_changed': 0}
    output = []
    for b in batches:
        target, valid, base, ptr = (b[k] for k in ('target', 'valid', 'base', 'pointer'))
        selected = apply_policy(base, b['copy'], b['margin'], b['mass'], policy)
        counts['valid_tokens'] += int(valid.sum())
        for name, guess in (('base', base), ('pointer', ptr), ('policy', selected)):
            counts[name + '_correct'] += int(((guess == target) & valid).sum())
        counts['policy_fixed'] += int(((base != target) & (selected == target) & valid).sum())
        counts['policy_regressed'] += int(((base == target) & (selected != target) & valid).sum())
        counts['policy_changed'] += int(((selected != base) & valid).sum())
        for i, item in enumerate(b['rows']):
            ref = item['row']['target']
            texts = {name: visible_text(tokenizer, guess[i].tolist(), int(tokenizer.eos_token_id), int(tokenizer.pad_token_id))
                     for name, guess in (('base', base), ('pointer', ptr), ('policy', selected))}
            expected = method_name(ref)
            counts['scorable_names'] += expected is not None
            result = {'id': item['row']['id']}
            for name, text in texts.items():
                exact = text == ref
                counts[name + '_exact_text'] += exact
                counts[name + '_exact_names'] += expected is not None and method_name(text) == expected
                result[name] = text
                result[name + '_exact'] = exact
            output.append(result)
    for name in ('base', 'pointer', 'policy'):
        counts[name + '_token_accuracy'] = counts[name + '_correct'] / max(1, counts['valid_tokens'])
    counts['policy_new_exact'] = sum(not r['base_exact'] and r['policy_exact'] for r in output)
    counts['policy_lost_exact'] = sum(r['base_exact'] and not r['policy_exact'] for r in output)
    return counts, output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--codec-config', required=True)
    p.add_argument('--codec-checkpoint', required=True)
    p.add_argument('--head', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--validation', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--sample-size', type=int, default=512)
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--prefix-length', type=int, default=256)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if min(args.sample_size, args.batch_size, args.prefix_length) < 1:
        p.error('sample size, batch size and prefix length must be positive')
    out = Path(args.output)
    if any((out / name).exists() for name in ('report.json', 'results.jsonl')):
        raise SystemExit('Output exists; choose a new path')
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('GPU unavailable')
    cfg = OmegaConf.load(args.codec_config)
    OmegaConf.resolve(cfg)
    tok = load_base_tokenizer(cfg.tokenizer)
    vqcfg = OmegaConf.to_container(cfg.vqvae, resolve=True)
    vqcfg.pop('checkpoint_path', None)
    if vqcfg.get('autoregressive_decoder', False):
        raise SystemExit('Expected parallel codec')
    codec = VQVAE(**OmegaConf.create(vqcfg))
    saved_codec = torch.load(args.codec_checkpoint, map_location='cpu', weights_only=False)
    codec.load_state_dict(saved_codec['model'], strict=True)
    codec.to(device).eval().requires_grad_(False)
    saved_head = torch.load(args.head, map_location='cpu', weights_only=False)
    if (saved_head.get('codec_sha256') != sha256_file(args.codec_checkpoint) or
        saved_head.get('train_sha256') != sha256_file(args.train)):
        raise SystemExit('Head was trained with a different codec or training dataset')
    head = PromptCopyHead(int(cfg.vqvae.embed_dim), confidence_aware_gate=bool(saved_head.get('confidence_aware_gate', False)))
    head.load_state_dict(saved_head['model'], strict=True)
    head.to(device).eval().requires_grad_(False)
    train = read_records(args.train, tok, int(cfg.model.context_length), args.prefix_length, int(cfg.vqvae.in_vocab_size))
    validation = read_records(args.validation, tok, int(cfg.model.context_length), args.prefix_length, int(cfg.vqvae.in_vocab_size))
    overlap = {r['row']['source'] for r in train} & {r['row']['source'] for r in validation}
    if overlap:
        raise SystemExit('Training and validation source paths overlap')
    sampled = random.Random(args.seed).sample(train, min(args.sample_size, len(train)))
    start = time.perf_counter()
    print(f'Calibrating on {len(sampled)} sampled pointer-training methods; validating on {len(validation)} separate-source methods.', flush=True)
    calibration = collect(codec, head, sampled, device, args.batch_size, 'Calibration')
    policy = choose_policy(calibration)
    print('Selected on training sample:', json.dumps(policy), flush=True)
    val = collect(codec, head, validation, device, args.batch_size, 'Validation')
    metrics, rows = evaluate(val, policy, tok)
    report = {'calibration_examples': len(sampled), 'validation_examples': len(validation),
              'calibration_seed': args.seed, 'minimum_calibration_fixes': 3,
              'margin_grid': MARGINS, 'attention_mass_grid': MASSES,
              'selected_policy': policy, 'metrics': metrics,
              'train_sha256': sha256_file(args.train), 'validation_sha256': sha256_file(args.validation),
              'codec_sha256': sha256_file(args.codec_checkpoint), 'head_sha256': sha256_file(args.head),
              'elapsed_seconds': time.perf_counter()-start,
              'scope': 'Thresholds chosen on pointer-training data only. Validation source paths are disjoint, but it has been repeatedly inspected. Codec latents encode known targets; this is not generator quality.',
              'limitations': 'The pointer head was trained on the calibration examples, so calibration is in-sample and may overfit. Token accuracy and exact method text do not establish semantic correctness. Context may contain near-duplicate information.'}
    with (out / 'results.jsonl').open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('\n=== Calibrated pointer policy ===\n' + json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
