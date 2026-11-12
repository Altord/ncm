"""Compare BF16-autocast vs FP32 VQ distance selection on the same encoder outputs."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from training.engine import make_loader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--cache-dir', default=None)
    p.add_argument('--max-examples', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.max_examples < 1 or args.batch_size < 1:
        p.error('example and batch counts must be positive')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('GPU unavailable; no comparison was run')
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    vc = OmegaConf.to_container(cfg.vqvae, resolve=True)
    vc.pop('checkpoint_path', None)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = VQVAE(**OmegaConf.create(vc))
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device).eval().requires_grad_(False)
    loader = make_loader(args.cache_dir or cfg.train.validation_cache_dir,
                         args.batch_size, 0, False, 42)
    quant = model.vq
    if bool(getattr(quant, '_stoch_encode', False)):
        raise ValueError('Stochastic encoding must be disabled for a paired comparison')
    original_distance = quant._distances
    def fp32_distance(x, codebook):
        with torch.autocast(device_type=device.type, enabled=False):
            return original_distance(x.float(), codebook.float())
    levels = list(map(int, quant.levels))
    level_diff = [0] * len(levels)
    examples = tokens = correct_bf = correct_fp = exact_bf = exact_fp = 0
    with torch.inference_mode():
        for batch in loader:
            count = min(args.max_examples - examples, batch['tokens'].shape[0])
            if count <= 0:
                break
            x = batch['tokens'][:count].to(device).long()
            valid = batch['mask'][:count].to(device).bool()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                z = model._encode_pre_quant(x)
                quant._distances = original_distance
                idx_bf = quant.z_to_idx(z)
                try:
                    quant._distances = fp32_distance
                    idx_fp = quant.z_to_idx(z)
                finally:
                    quant._distances = original_distance
            with torch.autocast(device_type=device.type, enabled=False):
                pred_bf = model.decode(idx_bf).argmax(dim=-1)
                pred_fp = model.decode(idx_fp).argmax(dim=-1)
            equal_bf = (pred_bf == x) | ~valid
            equal_fp = (pred_fp == x) | ~valid
            tokens += int(valid.sum())
            correct_bf += int(((pred_bf == x) & valid).sum())
            correct_fp += int(((pred_fp == x) & valid).sum())
            exact_bf += int(equal_bf.all(dim=1).sum())
            exact_fp += int(equal_fp.all(dim=1).sum())
            offset = 0
            for i, size in enumerate(levels):
                level_diff[i] += int((idx_bf[:, offset:offset+size] != idx_fp[:, offset:offset+size]).sum())
                offset += size
            examples += count
            print(f'Compared {examples}/{args.max_examples} examples', flush=True)
            if examples >= args.max_examples:
                break
    if not examples or not tokens:
        raise SystemExit('Validation cache has no usable examples')
    report = {'examples': examples, 'valid_tokens': tokens, 'levels': len(levels),
              'indices_per_example': sum(levels), 'different_indices': sum(level_diff),
              'different_index_fraction': sum(level_diff)/(examples*sum(levels)),
              'different_indices_by_level': level_diff,
              'bf16_distance_token_accuracy': correct_bf/tokens,
              'fp32_distance_token_accuracy': correct_fp/tokens,
              'bf16_distance_exact_chunks': exact_bf, 'fp32_distance_exact_chunks': exact_fp,
              'device': str(device), 'checkpoint': args.checkpoint,
              'note': 'Paired diagnostic on a frozen checkpoint and the same BF16-autocast encoder outputs. '
                      'Only distance calculation changes; other quantizer operations remain in autocast. '
                      'Both decoded in FP32. Later index differences can cascade from earlier changes. '
                      'Not an FP32-trained model or a compile/generation test.'}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('\n=== VQ distance precision comparison ===')
    print(f"Examples: {examples}; different indices: {report['different_indices']}/"
          f"{examples*sum(levels)} ({report['different_index_fraction']:.2%})")
    print(f"BF16 distances: token accuracy {correct_bf/tokens:.2%}; exact {exact_bf}/{examples}")
    print(f"FP32 distances: token accuracy {correct_fp/tokens:.2%}; exact {exact_fp}/{examples}")
    print(report['note'])
    print(f'Report: {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
