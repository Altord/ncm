"""One synthetic generator training batch using a saved frozen codec; saves no model."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer
from training.engine import autocast_context


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--codec-config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--pilot-config', default='config/train/godot-csharp-generator-pilot.yaml')
    p.add_argument('--device', default='cuda')
    p.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('batch size must be positive')
    dev = torch.device(args.device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('GPU unavailable; no benchmark was run')
    cfg = OmegaConf.load(args.codec_config)
    pilot = OmegaConf.load(args.pilot_config)
    OmegaConf.resolve(cfg)
    model_cfg = OmegaConf.create(OmegaConf.to_container(pilot.model, resolve=True))
    vq_cfg = OmegaConf.to_container(cfg.vqvae, resolve=True)
    vq_cfg.pop('checkpoint_path', None)
    levels = list(map(int, vq_cfg['vector_quantizer_config']['levels']['manual_levels']))
    vocabs = list(map(int, vq_cfg['vqvae_vocab_size']))
    if len(levels) < 2 or len(levels) != len(vocabs):
        raise ValueError('Expected at least two manually configured VQ levels')
    model_cfg.vocab_size = vocabs[1:]
    model_cfg.query_level0 = False
    model_cfg.predict_level0 = False
    model_cfg.prefix_pad_token_id = 0
    positions = int(model_cfg.prefix_len) + sum(levels[1:])
    report = dict(status='started', codec_checkpoint=str(Path(args.checkpoint).resolve()),
                  device=str(dev), precision=args.precision, batch_size=args.batch_size,
                  positions=positions, levels=len(levels), dense_mask_mib=positions**2*4/1024**2,
                  generator_width=int(model_cfg.n_embd), generator_layers=int(model_cfg.n_layer),
                  generator_attention_heads=int(model_cfg.n_head),
                  attention_pattern=str(model_cfg.attn_pattern),
                  note='Random synthetic VQ indices and prompt; randomly initialized pilot generator; '
                       'one forward/backward/optimizer step, no checkpoint and no quality measurement. '
                       'Excludes data preparation and codec encoding. Level 0 is not predicted.')
    stage = 'load codec'
    start = time.perf_counter()
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(dev)
        report['gpu'] = torch.cuda.get_device_name(dev)
        report['hip_version'] = torch.version.hip
    def mark(name):
        nonlocal stage
        stage = name
        print(f'[{time.perf_counter()-start:.1f}s] {name}', flush=True)
    try:
        mark(stage)
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        codec = VQVAE(**OmegaConf.create(vq_cfg))
        codec.load_state_dict(ckpt['model'], strict=True)
        codec.to(dev).eval().requires_grad_(False)
        del ckpt
        mark('construct generator and dense mask on CPU')
        model = NextConceptTransformer(vqvae_local=codec, config=model_cfg)
        model.to(dev).train()
        optim = torch.optim.AdamW(model.parameters(), lr=0.0003)
        mark('construct synthetic batch')
        with torch.no_grad():
            codes = torch.cat([torch.randint(v, (args.batch_size, n), device=dev)
                               for n, v in zip(levels, vocabs)], dim=1)
            prefix = torch.randint(int(codec.in_vocab_size),
                                   (args.batch_size, int(model_cfg.prefix_len)), device=dev)
            target_vecs = codec.vq.idx_to_ncm_input(codes)
            inputs = torch.cat([codec.encoder.wte(prefix), target_vecs], dim=1)
            labels = codes[:, levels[0]:]
            prefix_pad_mask = torch.zeros_like(prefix, dtype=torch.bool)
        if dev.type == 'cuda':
            torch.cuda.synchronize(dev)
        step_start = time.perf_counter()
        mark('forward and level losses')
        with autocast_context(dev, args.precision):
            logits = model(inputs, prefix_pad_mask=prefix_pad_mask)
            if logits.shape[1] != labels.shape[1]:
                raise RuntimeError(f'logit/label length mismatch: {logits.shape[1]} vs {labels.shape[1]}')
            losses, offset = [], 0
            for n, v in zip(levels[1:], vocabs[1:]):
                part = logits[:, offset:offset+n, :v].float()
                losses.append(F.cross_entropy(part.reshape(-1, v), labels[:, offset:offset+n].reshape(-1)))
                offset += n
            loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Loss is not finite')
        mark('backward')
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError('Non-finite gradients')
        mark('optimizer step')
        optim.step()
        if dev.type == 'cuda':
            torch.cuda.synchronize(dev)
        report.update(status='passed', loss=float(loss.detach()),
                      step_seconds=time.perf_counter()-step_start,
                      generator_parameters=sum(p.numel() for p in model.parameters()))
    except (RuntimeError, MemoryError) as exc:
        report.update(status='failed', failure_stage=stage, error=f'{type(exc).__name__}: {exc}')
    report['total_seconds'] = time.perf_counter()-start
    if dev.type == 'cuda':
        report['peak_allocated_gib'] = torch.cuda.max_memory_allocated(dev)/1024**3
        report['peak_reserved_gib'] = torch.cuda.max_memory_reserved(dev)/1024**3
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('\n=== Generator one-batch feasibility ===')
    print(json.dumps(report, indent=2), flush=True)
    print(f'Report: {out.resolve()}', flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
