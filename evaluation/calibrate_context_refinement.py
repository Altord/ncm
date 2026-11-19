"""Screen confidence gating using a deterministic calibration/evaluation split."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.context_refinement_head import ContextRefinementHead
from training.data import load_base_tokenizer, sha256_file
from training.train_prompt_copy_screen import read_records, batch, frozen_codec
from evaluation.godot_csharp_codec_eval import visible_text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('codec-config', 'codec-checkpoint', 'head-checkpoint', 'validation', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f'Output already exists: {output}')
    if args.batch_size < 1:
        p.error('Batch size must be positive')
    cfg = OmegaConf.load(args.codec_config)
    OmegaConf.resolve(cfg)
    tok = load_base_tokenizer(cfg.tokenizer)
    vqcfg = OmegaConf.to_container(cfg.vqvae, resolve=True)
    vqcfg.pop('checkpoint_path', None)
    device = torch.device(args.device)
    codec = VQVAE(**OmegaConf.create(vqcfg))
    codec.load_state_dict(torch.load(args.codec_checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    codec.to(device).eval().requires_grad_(False)
    saved = torch.load(args.head_checkpoint, map_location='cpu', weights_only=False)
    if saved['codec_sha256'] != sha256_file(args.codec_checkpoint):
        raise SystemExit('Head checkpoint codec hash does not match')
    prefix = int(saved['prefix_length'])
    head = ContextRefinementHead(int(cfg.vqvae.embed_dim), max_prompt_length=prefix,
                                 sequence_layers=int(saved.get('sequence_layers', 0)),
                                 max_sequence_length=int(saved.get('max_sequence_length', cfg.model.context_length))).to(device)
    head.load_state_dict(saved['model'], strict=True)
    head.eval()
    rows = read_records(args.validation, tok, int(cfg.model.context_length), prefix, int(cfg.vqvae.in_vocab_size))
    # Split by source file to avoid putting methods from one file on both sides.
    def split(source):
        return 'calibration' if int(hashlib.sha256(source.encode()).hexdigest(), 16) % 2 == 0 else 'evaluation'
    margins = (-1.0, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, float('inf'))
    totals = {s: [{'examples': 0, 'exact': 0, 'correct': 0, 'tokens': 0, 'fixed': 0, 'introduced': 0} for _ in margins] for s in ('calibration', 'evaluation')}
    start = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(rows), args.batch_size):
            group = rows[begin:begin + args.batch_size]
            x, valid, prompt, pmask = batch(group, device)
            features, logits, embeddings = frozen_codec(codec, x, prompt)
            base = logits.argmax(-1)
            top = logits.float().topk(2, dim=-1).values
            margin = top[..., 0] - top[..., 1]
            candidate = head(features, logits, embeddings, pmask, codec.decoder.output_layer.weight).argmax(-1)
            for n, limit in enumerate(margins):
                chosen = torch.where(margin <= limit, candidate, base)
                for i, item in enumerate(group):
                    stats = totals[split(item['row']['source'])][n]
                    mask = valid[i]
                    first, after, target = base[i], chosen[i], x[i]
                    text = visible_text(tok, after.tolist(), int(tok.eos_token_id), int(tok.pad_token_id))
                    stats['examples'] += 1
                    stats['exact'] += text == item['row']['target']
                    stats['correct'] += int(((after == target) & mask).sum())
                    stats['tokens'] += int(mask.sum())
                    stats['fixed'] += int(((first != target) & (after == target) & mask).sum())
                    stats['introduced'] += int(((first == target) & (after != target) & mask).sum())
            print(f'Processed {min(begin+len(group),len(rows))}/{len(rows)}', flush=True)
    if not totals['calibration'][0]['examples'] or not totals['evaluation'][0]['examples']:
        raise SystemExit('Both source-disjoint partitions must contain examples')
    # The first threshold is negative, so it always chooses the baseline.
    best = max(range(len(margins)), key=lambda i: (totals['calibration'][i]['exact'], totals['calibration'][i]['correct'], -i))
    def label(value):
        return 'all' if value == float('inf') else value
    report = {'validation_sha256': sha256_file(args.validation), 'codec_sha256': sha256_file(args.codec_checkpoint),
              'head_sha256': sha256_file(args.head_checkpoint), 'split': 'SHA-256 of source path modulo 2',
              'candidate_margins': [label(m) for m in margins], 'selected_margin': label(margins[best]),
              'calibration': totals['calibration'], 'evaluation': totals['evaluation'], 'selected_index': best,
              'elapsed_seconds': time.perf_counter()-start,
              'scope': 'Known-target frozen codec; synthetic signature and target-removed context. Threshold chosen only using calibration labels; evaluation labels used for offline scoring.',
              'limitations': 'Validation data has been repeatedly used and is not an independent final test; no compilation, semantics, or generation test.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print('\n=== Confidence-gated refinement results to share ===')
    print(f"Selected codec top-two logit margin threshold: {report['selected_margin']}")
    for label_name, index in [('baseline codec', 0), ('selected threshold', best), ('full refinement', len(margins)-1)]:
        r = totals['evaluation'][index]
        print(f"{label_name}: exact={r['exact']}/{r['examples']}; token accuracy={r['correct']/r['tokens']:.4%}; fixed={r['fixed']}; new errors={r['introduced']}")
    print(f'Report: {output}')
    print(report['scope'])


if __name__ == '__main__':
    main()
