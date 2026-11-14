"""Train a vocabulary-wide refinement head on a frozen known-target codec."""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.context_refinement_head import ContextRefinementHead, refinement_loss
from training.data import load_base_tokenizer, sha256_file
from training.train_prompt_copy_screen import read_records, batch, frozen_codec
from evaluation.godot_csharp_codec_eval import visible_text
from evaluation.compare_codec_method_names import method_name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('codec-config', 'codec-checkpoint', 'train', 'validation', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--prefix-length', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--error-boost', type=float, default=20.0)
    p.add_argument('--trust-weight', type=float, default=0.05)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.prefix_length) < 1 or args.lr <= 0:
        p.error('Steps, batch size, prefix length, and learning rate must be positive')
    out = Path(args.output)
    if any((out / name).exists() for name in ('refinement_head.pt', 'results.jsonl', 'report.json')):
        raise SystemExit('Existing experiment output; stopping to preserve it')
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise SystemExit('GPU unavailable')
        torch.cuda.reset_peak_memory_stats(device)
    cfg = OmegaConf.load(args.codec_config)
    OmegaConf.resolve(cfg)
    tok = load_base_tokenizer(cfg.tokenizer)
    vqcfg = OmegaConf.to_container(cfg.vqvae, resolve=True)
    vqcfg.pop('checkpoint_path', None)
    if vqcfg.get('autoregressive_decoder', False):
        raise SystemExit('Only parallel codec decoders are supported')
    codec = VQVAE(**OmegaConf.create(vqcfg))
    codec.load_state_dict(torch.load(args.codec_checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    codec.to(device).eval().requires_grad_(False)
    train = read_records(args.train, tok, int(cfg.model.context_length), args.prefix_length, int(cfg.vqvae.in_vocab_size))
    val = read_records(args.validation, tok, int(cfg.model.context_length), args.prefix_length, int(cfg.vqvae.in_vocab_size))
    overlap = {r['row']['source'] for r in train} & {r['row']['source'] for r in val}
    if overlap:
        raise SystemExit(f'Train and validation source paths overlap: {sorted(overlap)[:5]}')
    head = ContextRefinementHead(int(cfg.vqvae.embed_dim), max_prompt_length=args.prefix_length).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr)
    start = time.perf_counter()
    print(f'Frozen codec; {len(train)} training and {len(val)} validation methods; head parameters: {sum(p.numel() for p in head.parameters()):,}', flush=True)
    for step in range(1, args.steps + 1):
        x, valid, prompt, pmask = batch([train[rng.randrange(len(train))] for _ in range(args.batch_size)], device)
        features, logits, embeddings = frozen_codec(codec, x, prompt)
        refined = head(features, logits, embeddings, pmask, codec.decoder.output_layer.weight)
        loss, parts = refinement_loss(refined, logits, x, valid, args.error_boost, args.trust_weight)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optim.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            elapsed = time.perf_counter() - start
            print(json.dumps({'step': step, 'loss': float(loss.detach()), **{k: float(v) for k, v in parts.items()},
                              'steps_per_second': step / elapsed, 'estimated_seconds_remaining': (args.steps-step)*elapsed/step}), flush=True)
    training_seconds = time.perf_counter() - start
    torch.save({'model': head.state_dict(), 'steps': args.steps, 'seed': args.seed,
                'prefix_length': args.prefix_length, 'codec_sha256': sha256_file(args.codec_checkpoint)}, out / 'refinement_head.pt')
    head.eval()
    results = []
    valid_count = before_correct = after_correct = fixed = introduced = fixed_absent = 0
    with torch.inference_mode():
        for begin in range(0, len(val), args.batch_size):
            group = val[begin:begin+args.batch_size]
            x, valid, prompt, pmask = batch(group, device)
            features, logits, embeddings = frozen_codec(codec, x, prompt)
            before = logits.argmax(-1)
            after = head(features, logits, embeddings, pmask, codec.decoder.output_layer.weight).argmax(-1)
            wrong = (before != x) & valid
            now_wrong = (after != x) & valid
            present = ((prompt[:, None, :] == x[:, :, None]) & pmask[:, None, :]).any(-1)
            valid_count += int(valid.sum())
            before_correct += int(((before == x) & valid).sum())
            after_correct += int(((after == x) & valid).sum())
            fixed += int((wrong & ~now_wrong).sum())
            fixed_absent += int((wrong & ~now_wrong & ~present).sum())
            introduced += int((~wrong & now_wrong & valid).sum())
            for i, item in enumerate(group):
                target = item['row']['target']
                first = visible_text(tok, before[i].tolist(), int(tok.eos_token_id), int(tok.pad_token_id))
                second = visible_text(tok, after[i].tolist(), int(tok.eos_token_id), int(tok.pad_token_id))
                name = method_name(target)
                results.append({'id': item['row']['id'], 'before': first, 'after': second,
                    'before_exact_text': first == target, 'after_exact_text': second == target,
                    'name_scorable': name is not None, 'before_name_exact': name is not None and method_name(first) == name,
                    'after_name_exact': name is not None and method_name(second) == name})
            print(f'Validation {min(begin+len(group),len(val))}/{len(val)}', flush=True)
    with (out / 'results.jsonl').open('x', encoding='utf-8') as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = {'train_examples': len(train), 'validation_examples': len(val), 'train_steps': args.steps,
        'train_sha256': sha256_file(args.train), 'validation_sha256': sha256_file(args.validation),
        'codec_sha256': sha256_file(args.codec_checkpoint), 'seed': args.seed, 'error_boost': args.error_boost,
        'trust_weight': args.trust_weight, 'head_parameters': sum(p.numel() for p in head.parameters()),
        'before_token_accuracy': before_correct/valid_count, 'after_token_accuracy': after_correct/valid_count,
        'before_exact_text': sum(r['before_exact_text'] for r in results), 'after_exact_text': sum(r['after_exact_text'] for r in results),
        'new_exact_text': sum(not r['before_exact_text'] and r['after_exact_text'] for r in results),
        'lost_exact_text': sum(r['before_exact_text'] and not r['after_exact_text'] for r in results),
        'before_exact_names': sum(r['before_name_exact'] for r in results), 'after_exact_names': sum(r['after_name_exact'] for r in results),
        'scorable_names': sum(r['name_scorable'] for r in results), 'baseline_errors_fixed': fixed,
        'fixed_target_absent_from_prompt': fixed_absent, 'new_token_errors': introduced,
        'training_seconds': training_seconds, 'elapsed_seconds': time.perf_counter()-start,
        'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else None,
        'scope': 'Frozen known-target codec round trip; prompt contains a synthetic target signature and target-removed project context. Not generation.',
        'limitations': 'Repeatedly used validation; path separation does not rule out near duplicates. No compilation or behavioral evaluation.'}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('\n=== Vocabulary-wide refinement screen ===\n'+json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
