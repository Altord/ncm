"""Evaluate a calibrated refinement head on synthetic-signature codec stress inputs."""
import argparse
import json
import time
from pathlib import Path
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.context_refinement_head import ContextRefinementHead
from training.data import load_base_tokenizer, sha256_file
from training.train_prompt_copy_screen import batch, frozen_codec
from evaluation.evaluate_pointer_stress import make_batch_row, compile_or_reuse
from evaluation.godot_csharp_codec_eval import visible_text


def gate_predictions(logits, refined, threshold):
    top = logits.float().topk(2, dim=-1).values
    return torch.where(top[..., 0]-top[..., 1] <= threshold, refined.argmax(-1), logits.argmax(-1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'codec', 'head', 'calibration', 'input', 'baseline', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--framework', default='net8.0')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists():
        raise SystemExit('Output already exists')
    calibration = json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    codec_hash, head_hash = sha256_file(args.codec), sha256_file(args.head)
    if calibration['codec_sha256'] != codec_hash or calibration['head_sha256'] != head_hash:
        raise SystemExit('Calibration did not use this codec and refinement head')
    threshold = calibration['selected_margin']
    threshold = float('inf') if threshold == 'all' else float(threshold)
    baseline_path = Path(args.baseline)
    report = json.loads(baseline_path.with_suffix('.report.json').read_text(encoding='utf-8'))
    if report['input_sha256'] != sha256_file(args.input) or Path(report['checkpoint']).resolve() != Path(args.codec).resolve() or not report['compile_requested']:
        raise SystemExit('Saved baseline does not match input or codec')
    originals = [json.loads(line) for line in Path(args.input).read_text(encoding='utf-8').splitlines() if line.strip()]
    saved = [json.loads(line) for line in baseline_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    old = {r['id']: r for r in saved}
    if not originals or len(old) != len(saved) or len({r['id'] for r in originals}) != len(originals) or set(old) != {r['id'] for r in originals}:
        raise SystemExit('Stress IDs are missing or duplicated')
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    tok = load_base_tokenizer(cfg.tokenizer)
    vc = OmegaConf.to_container(cfg.vqvae, resolve=True)
    vc.pop('checkpoint_path', None)
    device = torch.device(args.device)
    codec = VQVAE(**OmegaConf.create(vc))
    codec.load_state_dict(torch.load(args.codec, map_location='cpu', weights_only=False)['model'], strict=True)
    codec.to(device).eval().requires_grad_(False)
    ckpt = torch.load(args.head, map_location='cpu', weights_only=False)
    if ckpt['codec_sha256'] != codec_hash:
        raise SystemExit('Head codec hash mismatch')
    prefix = int(ckpt['prefix_length'])
    head = ContextRefinementHead(int(cfg.vqvae.embed_dim), max_prompt_length=prefix).to(device)
    head.load_state_dict(ckpt['model'], strict=True)
    head.eval()
    rows = [make_batch_row(row, tok, int(cfg.model.context_length), prefix, int(cfg.vqvae.in_vocab_size)) for row in originals]
    results, fixed, introduced, calls, tokens, before_correct, after_correct = [], 0, 0, 0, 0, 0, 0
    start = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(rows), args.batch_size):
            group = rows[begin:begin+args.batch_size]
            x, valid, prompt, pmask = batch(group, device)
            features, logits, embeddings = frozen_codec(codec, x, prompt)
            before = logits.argmax(-1)
            after = gate_predictions(logits, head(features, logits, embeddings, pmask, codec.decoder.output_layer.weight), threshold)
            tokens += int(valid.sum())
            before_correct += int(((before == x) & valid).sum())
            after_correct += int(((after == x) & valid).sum())
            fixed += int(((before != x) & (after == x) & valid).sum())
            introduced += int(((before == x) & (after != x) & valid).sum())
            for i, item in enumerate(group):
                row, prior = item['row'], old[item['row']['id']]
                if prior['target'] != row['target'] or prior['original_compile']['status'] != 'passed':
                    raise SystemExit(f'Baseline target or original compile mismatch: {row["id"]}')
                first = visible_text(tok, before[i].tolist(), int(tok.eos_token_id), int(tok.pad_token_id))
                second = visible_text(tok, after[i].tolist(), int(tok.eos_token_id), int(tok.pad_token_id))
                if first != prior['reconstructed']:
                    raise SystemExit(f'Recomputed baseline differs: {row["id"]}')
                status = prior['reconstruction_compile']['status']
                if status not in ('passed', 'failed', 'empty'):
                    raise SystemExit(f'Baseline compile unavailable: {row["id"]}')
                selected, called = compile_or_reuse(second, {first: status}, args.framework)
                calls += called
                results.append({'id': row['id'], 'baseline': first, 'selected': second,
                    'baseline_exact': first == row['target'], 'selected_exact': second == row['target'],
                    'baseline_compile': status, 'selected_compile': selected})
            print(f'Processed {min(begin+len(group),len(rows))}/{len(rows)}; new compiler calls={calls}', flush=True)
    output = {'examples': len(results), 'margin': calibration['selected_margin'],
        'baseline_token_accuracy': before_correct/tokens, 'selected_token_accuracy': after_correct/tokens,
        'baseline_exact': sum(r['baseline_exact'] for r in results), 'selected_exact': sum(r['selected_exact'] for r in results),
        'baseline_compiles': sum(r['baseline_compile'] == 'passed' for r in results),
        'selected_compiles': sum(r['selected_compile'] == 'passed' for r in results),
        'exact_gained': sum(not r['baseline_exact'] and r['selected_exact'] for r in results),
        'exact_lost': sum(r['baseline_exact'] and not r['selected_exact'] for r in results),
        'fixed_tokens': fixed, 'new_token_errors': introduced, 'new_compiler_calls': calls,
        'elapsed_seconds': time.perf_counter()-start, 'codec_sha256': codec_hash, 'head_sha256': head_hash,
        'scope': 'Known-target codec round trip. The synthetic prompt supplies the signature from the original target; no project context is supplied.',
        'limitations': 'Stress set has been repeatedly inspected. Compilation does not prove semantic correctness. Not generation.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in results: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    out.with_suffix('.report.json').write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
    print('\n=== Gated refinement stress results to share ===\n'+json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    main()
