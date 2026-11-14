"""Evaluate saved pointer heads on the codec stress set with supplied synthetic signatures.

Signatures are extracted from targets to construct this synthetic task's input.
The codec also encodes the known target. This is not a generator evaluation.
"""
from __future__ import annotations
import argparse
import difflib
import json
from pathlib import Path
import sys
import time

import torch
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.vqvae import VQVAE
from models.prompt_copy_head import PromptCopyHead
from training.data import encode_text, fixed_sequence, load_base_tokenizer, sha256_file
from training.train_prompt_copy_screen import batch, frozen_codec
from evaluation.codec_prompt_signature_copy import parse_method, PROMPT_PREFIX
from evaluation.compare_codec_method_names import method_name
from evaluation.godot_csharp_codec_eval import visible_text
from evaluation.godot_csharp_generator_pilot import compile_candidate


def make_signature_prompt(target):
    parsed = parse_method(target, require_clean=True)
    if parsed is None:
        raise ValueError('Target method cannot be parsed unambiguously')
    raw = target.encode('utf-8')
    signature = raw[parsed['start']:parsed['body']].decode('utf-8').strip()
    if not signature:
        raise ValueError('Empty method signature')
    return PROMPT_PREFIX + signature


def make_batch_row(row, tokenizer, target_length, prefix_length, vocab_size):
    prompt = make_signature_prompt(row['target'])
    prompt_ids = encode_text(tokenizer, prompt, False)
    target_ids = encode_text(tokenizer, row['target'], True)
    if not prompt_ids or len(prompt_ids) > prefix_length or len(target_ids) > target_length:
        raise ValueError(f'Synthetic prompt or target is empty/too long: {row["id"]}')
    if min(prompt_ids + target_ids) < 0 or max(prompt_ids + target_ids) >= vocab_size:
        raise ValueError(f'Token outside codec vocabulary: {row["id"]}')
    pad, eos = int(tokenizer.pad_token_id), int(tokenizer.eos_token_id)
    x, valid = fixed_sequence(target_ids, target_length, pad, eos)
    return {'row': row, 'x': x, 'mask': valid,
            'prompt': prompt_ids + [pad] * (prefix_length-len(prompt_ids)),
            'prompt_mask': [1] * len(prompt_ids) + [0] * (prefix_length-len(prompt_ids))}


def load_head(path, width, prefix, codec_hash, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('codec_sha256') != codec_hash:
        raise ValueError(f'Pointer was trained with a different codec: {path}')
    max_length = int(saved.get('context_max_length', prefix))
    if int(saved.get('prompt_context_layers', 0)) and max_length != prefix:
        raise ValueError(f'Incorrect prompt length for pointer: {path}')
    head = PromptCopyHead(width, confidence_aware_gate=bool(saved.get('confidence_aware_gate', False)),
                          context_layers=int(saved.get('prompt_context_layers', 0)),
                          context_heads=int(saved.get('prompt_context_heads', 4)),
                          context_max_length=max_length)
    head.load_state_dict(saved['model'], strict=True)
    return head.to(device).eval().requires_grad_(False)


def compile_or_reuse(text, cache, framework):
    if text in cache:
        return cache[text], False
    compilation = compile_candidate(text, framework) if text.strip() else {'status': 'empty'}
    if compilation['status'] not in ('passed', 'failed', 'empty'):
        raise RuntimeError(f'Compiler unavailable: {compilation}')
    cache[text] = compilation['status']
    return cache[text], bool(text.strip())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--codec', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--signature-head', required=True)
    p.add_argument('--context-head', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=16,
                   help='Match the saved baseline evaluation batch size')
    p.add_argument('--framework', default='net8.0')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('batch size must be positive')
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists():
        raise SystemExit('Output exists; choose a new path')
    baseline_path = Path(args.baseline)
    old_report = json.loads(baseline_path.with_suffix('.report.json').read_text(encoding='utf-8'))
    if (old_report.get('input_sha256') != sha256_file(args.input) or
        Path(old_report.get('checkpoint', '')).resolve() != Path(args.codec).resolve() or
        not old_report.get('compile_requested')):
        raise SystemExit('Saved baseline does not match input, codec or compilation settings')
    original = [json.loads(x) for x in Path(args.input).read_text(encoding='utf-8').splitlines() if x.strip()]
    saved = [json.loads(x) for x in baseline_path.read_text(encoding='utf-8').splitlines() if x.strip()]
    old = {row['id']: row for row in saved}
    if (not original or len(old) != len(saved) or len(old) != len(original) or
        len({r['id'] for r in original}) != len(original) or set(old) != {r['id'] for r in original} or
        old_report.get('examples') != len(original)):
        raise SystemExit('Missing or duplicate stress rows')
    for row in original:
        prior = old[row['id']]
        if (prior['target'] != row['target'] or prior.get('original_compile', {}).get('status') != 'passed' or
            prior.get('reconstruction_compile', {}).get('status') not in ('passed', 'failed', 'empty')):
            raise SystemExit(f'Unusable saved compile record: {row["id"]}')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('GPU unavailable')
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    tok = load_base_tokenizer(cfg.tokenizer)
    vqcfg = OmegaConf.to_container(cfg.vqvae, resolve=True)
    vqcfg.pop('checkpoint_path', None)
    if vqcfg.get('autoregressive_decoder', False):
        raise SystemExit('Expected parallel codec')
    codec = VQVAE(**OmegaConf.create(vqcfg))
    checkpoint = torch.load(args.codec, map_location='cpu', weights_only=False)
    codec.load_state_dict(checkpoint['model'], strict=True)
    codec.to(device).eval().requires_grad_(False)
    codec_hash = sha256_file(args.codec)
    heads = {'signature': load_head(args.signature_head, int(cfg.vqvae.embed_dim), 64, codec_hash, device),
             'contextual': load_head(args.context_head, int(cfg.vqvae.embed_dim), 256, codec_hash, device)}
    lengths = {'signature': 64, 'contextual': 256}
    rows = {name: [make_batch_row(row, tok, int(cfg.model.context_length), length, int(cfg.vqvae.in_vocab_size))
                   for row in original] for name, length in lengths.items()}
    totals = {name: {'correct_tokens': 0, 'exact_tokens': 0, 'exact_text': 0,
                     'compile_passes': 0, 'name_exact': 0, 'fixed_tokens': 0, 'new_token_errors': 0}
              for name in ('baseline', 'signature', 'contextual')}
    valid_tokens = calls = 0
    results = []
    started = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(original), args.batch_size):
            group = rows['signature'][begin:begin+args.batch_size]
            x, valid, prompt, pmask = batch(group, device)
            features, logits, embeddings = frozen_codec(codec, x, prompt)
            predictions = {'baseline': logits.argmax(-1)}
            predictions['signature'], _ = heads['signature'].predict(features, logits, embeddings, prompt, pmask)
            ctx_group = rows['contextual'][begin:begin+args.batch_size]
            _, _, ctx_prompt, ctx_mask = batch(ctx_group, device)
            ctx_embeddings = codec.decoder.output_layer.weight[ctx_prompt]
            predictions['contextual'], _ = heads['contextual'].predict(features, logits, ctx_embeddings, ctx_prompt, ctx_mask)
            for name, pred in predictions.items():
                totals[name]['correct_tokens'] += int(((pred == x) & valid).sum())
                totals[name]['fixed_tokens'] += int(((pred == x) & (predictions['baseline'] != x) & valid).sum())
                totals[name]['new_token_errors'] += int(((pred != x) & (predictions['baseline'] == x) & valid).sum())
            valid_tokens += int(valid.sum())
            for i, item in enumerate(group):
                row = item['row']
                prior = old[row['id']]
                texts = {name: visible_text(tok, pred[i].tolist(), int(tok.eos_token_id), int(tok.pad_token_id))
                         for name, pred in predictions.items()}
                if texts['baseline'] != prior['reconstructed']:
                    raise SystemExit(f'Recomputed codec output differs from saved baseline: {row["id"]}')
                compile_cache = {texts['baseline']: prior['reconstruction_compile']['status']}
                result = {'id': row['id'], 'prompt': make_signature_prompt(row['target'])}
                expected_name = method_name(row['target'])
                for name, text in texts.items():
                    try:
                        status, called = compile_or_reuse(text, compile_cache, args.framework)
                    except RuntimeError as exc:
                        raise SystemExit(f'{row["id"]}: {exc}') from exc
                    calls += called
                    totals[name]['compile_passes'] += status == 'passed'
                    totals[name]['exact_text'] += text == row['target']
                    totals[name]['exact_tokens'] += bool(((predictions[name][i] == x[i]) | ~valid[i]).all())
                    totals[name]['name_exact'] += expected_name is not None and method_name(text) == expected_name
                    result[name] = {'text': text, 'compile_status': status,
                                    'exact_text': text == row['target'],
                                    'character_similarity': difflib.SequenceMatcher(None, row['target'], text, autojunk=False).ratio()}
                results.append(result)
            print(f'Processed {min(begin + len(group), len(original))}/{len(original)}; '
                  f'new compiler calls={calls}', flush=True)
    for name in totals:
        totals[name]['token_accuracy'] = totals[name]['correct_tokens'] / max(1, valid_tokens)
        totals[name]['mean_character_similarity'] = sum(r[name]['character_similarity'] for r in results) / len(results)
        totals[name]['new_exact_text'] = sum(not r['baseline']['exact_text'] and r[name]['exact_text'] for r in results)
        totals[name]['lost_exact_text'] = sum(r['baseline']['exact_text'] and not r[name]['exact_text'] for r in results)
        totals[name]['new_compiles'] = sum(r['baseline']['compile_status'] != 'passed' and r[name]['compile_status'] == 'passed' for r in results)
        totals[name]['lost_compiles'] = sum(r['baseline']['compile_status'] == 'passed' and r[name]['compile_status'] != 'passed' for r in results)
    report = {'examples': len(results), 'valid_tokens': valid_tokens, 'results': totals,
              'new_compiler_calls': calls, 'elapsed_seconds': time.perf_counter() - started,
              'codec_sha256': codec_hash, 'input_sha256': sha256_file(args.input),
              'signature_head_sha256': sha256_file(args.signature_head),
              'context_head_sha256': sha256_file(args.context_head),
              'scope': 'Known-target codec round trip. Synthetic prompt includes the exact method signature extracted from the target; no target body is supplied in the prompt. Contextual head receives no nearby project context in this stress test.',
              'limitations': 'Not code generation from unknown targets. Synthetic signatures may be unavailable in natural-language tasks; context head is evaluated without its usual project context. Compilation does not establish semantic correctness; stress set has been repeatedly inspected.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('\n=== Pointer codec stress evaluation ===\n' + json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
