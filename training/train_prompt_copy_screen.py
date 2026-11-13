"""Train a small prompt-copy head on a frozen known-input codec.

This diagnostic encodes the target into VQ latents; it is not a generator test.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import time

import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.prompt_copy_head import PromptCopyHead
from training.data import load_base_tokenizer, encode_text, fixed_sequence, sha256_file
from evaluation.godot_csharp_codec_eval import visible_text
from evaluation.compare_codec_method_names import method_name


def read_records(path, tokenizer, context, prefix, vocab):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    seen = set()
    output = []
    pad, eos = int(tokenizer.pad_token_id), int(tokenizer.eos_token_id)
    for row in rows:
        if not row.get('id') or row['id'] in seen or not row.get('source'):
            raise ValueError(f'Missing/duplicate ID or source in {path}')
        seen.add(row['id'])
        if not isinstance(row.get('prompt'), str) or not isinstance(row.get('target'), str):
            raise ValueError('Rows must contain prompt and target strings')
        prompt = encode_text(tokenizer, row['prompt'], False)
        target = encode_text(tokenizer, row['target'], True)
        if not prompt or len(prompt) > prefix or len(target) > context:
            raise ValueError(f'Truncated or empty input for {row["id"]}; refusing silent truncation')
        if min(prompt + target) < 0 or max(prompt + target) >= vocab:
            raise ValueError('Token outside codec vocabulary')
        x, mask = fixed_sequence(target, context, pad, eos)
        prompt_pad = prompt + [pad] * (prefix - len(prompt))
        output.append({'row': row, 'x': x, 'mask': mask, 'prompt': prompt_pad,
                       'prompt_mask': [1]*len(prompt) + [0]*(prefix-len(prompt))})
    if not output:
        raise ValueError(f'Empty dataset: {path}')
    return output


def batch(rows, device):
    def tensor(key, dtype):
        return torch.tensor([r[key] for r in rows], dtype=dtype, device=device)
    return (tensor('x', torch.long), tensor('mask', torch.bool),
            tensor('prompt', torch.long), tensor('prompt_mask', torch.bool))


@torch.no_grad()
def frozen_codec(codec, x, prompt):
    codes = codec.encode(x)
    z = codec.vq.idx_to_z_q(codes)
    features = codec.decoder.forward_features(z)
    logits = codec.decoder.output_layer(features)
    prompt_embeddings = codec.decoder.output_layer.weight[prompt]
    return features.detach(), logits.detach(), prompt_embeddings.detach()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--codec-config', required=True)
    p.add_argument('--codec-checkpoint', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--validation', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--prefix-length', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=3e-4)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.prefix_length) < 1 or args.lr <= 0:
        p.error('steps, batch size, prefix length and lr must be positive')
    out = Path(args.output)
    if any((out / name).exists() for name in ('copy_head.pt', 'results.jsonl', 'report.json')):
        raise SystemExit('Existing experiment output; stopping to preserve it')
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('GPU unavailable')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    cfg = OmegaConf.load(args.codec_config)
    OmegaConf.resolve(cfg)
    tok = load_base_tokenizer(cfg.tokenizer)
    vqcfg = OmegaConf.to_container(cfg.vqvae, resolve=True)
    vqcfg.pop('checkpoint_path', None)
    if vqcfg.get('autoregressive_decoder', False):
        raise SystemExit('This screen supports only a parallel codec decoder')
    codec = VQVAE(**OmegaConf.create(vqcfg))
    ckpt = torch.load(args.codec_checkpoint, map_location='cpu', weights_only=False)
    codec.load_state_dict(ckpt['model'], strict=True)
    codec.to(device).eval().requires_grad_(False)
    if not 0 <= int(tok.pad_token_id) < int(cfg.vqvae.in_vocab_size):
        raise SystemExit('Tokenizer pad token is outside the codec vocabulary')
    train = read_records(args.train, tok, int(cfg.model.context_length), args.prefix_length, int(cfg.vqvae.in_vocab_size))
    val = read_records(args.validation, tok, int(cfg.model.context_length), args.prefix_length, int(cfg.vqvae.in_vocab_size))
    overlap = {r['row']['source'] for r in train} & {r['row']['source'] for r in val}
    if overlap:
        raise SystemExit(f'Train/validation source paths overlap: {sorted(overlap)[:5]}')
    head = PromptCopyHead(int(cfg.vqvae.embed_dim)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    start = time.perf_counter()
    print(f'Training on {len(train)} methods; validating on {len(val)} source-disjoint methods; '
          f'codec frozen; copy-head parameters={sum(p.numel() for p in head.parameters()):,}.', flush=True)
    for step in range(1, args.steps + 1):
        chosen = [train[rng.randrange(len(train))] for _ in range(args.batch_size)]
        x, valid, prompt, pmask = batch(chosen, device)
        features, logits, embeddings = frozen_codec(codec, x, prompt)
        loss = head.nll(features, logits, embeddings, prompt, pmask, x, valid)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            elapsed = time.perf_counter() - start
            print(json.dumps({'step': step, 'loss': float(loss.detach()),
                              'steps_per_second': step / max(elapsed, 1e-9),
                              'elapsed_seconds': elapsed,
                              'estimated_training_seconds_remaining': (args.steps-step) * elapsed / step}), flush=True)
    training_seconds = time.perf_counter() - start
    torch.save({'model': head.state_dict(), 'steps': args.steps, 'seed': args.seed,
                'codec_sha256': sha256_file(args.codec_checkpoint),
                'train_sha256': sha256_file(args.train)}, out / 'copy_head.pt')
    head.eval()
    result = []
    total_correct = [0, 0]
    total_valid = 0
    gates = 0.0
    with torch.inference_mode():
        for begin in range(0, len(val), args.batch_size):
            group = val[begin:begin + args.batch_size]
            x, valid, prompt, pmask = batch(group, device)
            features, logits, embeddings = frozen_codec(codec, x, prompt)
            original = logits.argmax(-1)
            copied, gate = head.predict(features, logits, embeddings, prompt, pmask)
            total_correct[0] += int(((original == x) & valid).sum())
            total_correct[1] += int(((copied == x) & valid).sum())
            total_valid += int(valid.sum())
            gates += float((gate * valid).sum())
            for i, item in enumerate(group):
                reference = item['row']['target']
                before = visible_text(tok, original[i].tolist(), int(tok.eos_token_id), int(tok.pad_token_id))
                after = visible_text(tok, copied[i].tolist(), int(tok.eos_token_id), int(tok.pad_token_id))
                expected_name = method_name(reference)
                result.append({'id': item['row']['id'], 'before': before, 'after': after,
                               'before_exact_text': before == reference, 'after_exact_text': after == reference,
                               'name_scorable': expected_name is not None,
                               'before_name_exact': expected_name is not None and method_name(before) == expected_name,
                               'after_name_exact': expected_name is not None and method_name(after) == expected_name})
            print(f'Validation {min(begin + len(group), len(val))}/{len(val)}', flush=True)
    with (out / 'results.jsonl').open('x', encoding='utf-8') as f:
        for row in result:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = {'train_examples': len(train), 'validation_examples': len(val),
              'train_sha256': sha256_file(args.train), 'validation_sha256': sha256_file(args.validation),
              'codec_sha256': sha256_file(args.codec_checkpoint), 'train_steps': args.steps,
              'copy_head_parameters': sum(p.numel() for p in head.parameters()),
              'batch_size': args.batch_size, 'prefix_length': args.prefix_length, 'seed': args.seed,
              'before_token_accuracy': total_correct[0] / total_valid, 'after_token_accuracy': total_correct[1] / total_valid,
              'before_exact_text': sum(r['before_exact_text'] for r in result),
              'after_exact_text': sum(r['after_exact_text'] for r in result),
              'new_exact_text': sum(not r['before_exact_text'] and r['after_exact_text'] for r in result),
              'lost_exact_text': sum(r['before_exact_text'] and not r['after_exact_text'] for r in result),
              'before_exact_names': sum(r['before_name_exact'] for r in result),
              'after_exact_names': sum(r['after_name_exact'] for r in result),
              'scorable_names': sum(r['name_scorable'] for r in result),
              'mean_copy_gate': gates / total_valid,
              'training_seconds': training_seconds,
              'peak_gpu_allocated_gib': (torch.cuda.max_memory_allocated(device) / 2**30) if device.type == 'cuda' else None,
              'elapsed_seconds_including_validation': time.perf_counter()-start,
              'scope': 'Learned prompt-token pointer on frozen known-target VQ codec; target encoded into VQ latents. Not a generator test. Prompt signatures are synthetic and contain target signature.',
              'limitations': 'Source-path separation does not exclude copied/near-duplicate code; validation has been repeatedly used. No compilation, behavioral checks, full generation speed, or independent test.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('\n=== Learned prompt-copy screen ===\n' + json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
