"""Train or evaluate a small frozen-codec latent adapter on known inputs."""
import argparse
import json
import statistics
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.latent_refiner import make_latent_refiner
from training.data import encode_text, fixed_sequence, load_base_tokenizer, sha256_file
from training.engine import make_loader, seed_all
from evaluation.ensemble_codec_methods import load_rows
from evaluation.godot_csharp_codec_eval import visible_text


@torch.no_grad()
def benchmark_decode(model, adapter, q, device, warmup=3, repeats=10):
    """Measure the frozen decoder on fixed latents, with and without adapter."""
    def sync():
        if device.type == 'cuda': torch.cuda.synchronize(device)
    for _ in range(warmup):
        model.decoder(q); model.decoder(adapter(q))
    sync()
    samples = {'baseline': [], 'adapter': []}
    for i in range(repeats):
        for name in (('baseline', 'adapter') if i % 2 == 0 else ('adapter', 'baseline')):
            sync(); start = time.perf_counter()
            model.decoder(q if name == 'baseline' else adapter(q))
            sync(); samples[name].append(time.perf_counter() - start)
    return {'batch_size': int(q.shape[0]), 'repeats': repeats,
            'baseline_median_seconds': statistics.median(samples['baseline']),
            'adapter_median_seconds': statistics.median(samples['adapter']),
            'scope': 'Fixed latent batch, frozen decoder forward, GPU-synchronized when applicable; excludes encoding, unpacking, argmax, text decoding, compilation and model loading.'}


def load_codec(config, checkpoint, device):
    cfg = OmegaConf.load(config); OmegaConf.resolve(cfg)
    vcfg = OmegaConf.to_container(cfg.vqvae, resolve=True); vcfg.pop('checkpoint_path', None)
    if vcfg.get('autoregressive_decoder', False): raise ValueError('Parallel codec decoder required')
    model = VQVAE(**OmegaConf.create(vcfg))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    return cfg, model.to(device).eval().requires_grad_(False)


def valid_metrics(model, adapter, loader, device):
    correct = valid = exact = examples = 0
    adapter.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch['tokens'].to(device).long(); mask = batch['mask'].to(device).bool()
            q = model.vq.idx_to_z_q(model.encode(x))
            pred = model.decoder(adapter(q)).argmax(-1)
            correct += int(((pred == x) & mask).sum()); valid += int(mask.sum())
            exact += int(((pred == x) | ~mask).all(-1).sum()); examples += x.shape[0]
    if not valid or not examples: raise ValueError('Empty validation cache')
    return {'token_accuracy': correct / valid, 'exact': exact, 'examples': examples}


def latent_valid_mask(mask, latent_length):
    """Mark latent positions whose encoder pooling window includes a valid token."""
    if mask.ndim != 2 or latent_length < 1 or mask.shape[1] % latent_length:
        raise ValueError('Token mask cannot be aligned with latent positions')
    return mask.bool().reshape(mask.shape[0], latent_length, -1).any(-1)


def adapter_training_loss(model, adapter, q, z, x, mask, ce_weight, latent_weight, latent_mask='all'):
    refined = adapter(q)
    if latent_mask == 'valid':
        latent_valid = latent_valid_mask(mask, refined.shape[1]).unsqueeze(-1)
        squared = (refined.float() - z.float()).square() * latent_valid
        mse = squared.sum() / (latent_valid.sum().clamp_min(1) * refined.shape[-1])
    elif latent_mask == 'all':
        mse = F.mse_loss(refined.float(), z.float())
    else:
        raise ValueError('Unknown latent mask mode')
    ce = None
    loss = latent_weight * mse
    if ce_weight:
        logits = model.decoder(refined)
        ce = F.cross_entropy(logits.float().transpose(1, 2), x, reduction='none')
        ce = (ce * mask).sum() / mask.sum().clamp_min(1)
        loss = loss + ce_weight * ce
    return loss, ce, mse


def train(args, cfg, model, device):
    out = Path(args.output)
    if out.exists(): raise SystemExit(f'Output exists: {out}')
    if not cfg.train.get('validation_cache_dir'): raise SystemExit('Validation cache is required')
    if not Path(cfg.data.cache_dir, 'manifest.json').is_file(): raise SystemExit('Training cache is missing')
    seed_all(int(cfg.train.seed))
    loader = make_loader(cfg.data.cache_dir, args.batch_size, 0, True, int(cfg.train.seed))
    val = make_loader(cfg.train.validation_cache_dir, args.batch_size, 0, False, int(cfg.train.seed))
    width = int(cfg.vqvae.embed_dim); adapter = make_latent_refiner(width, width * 2, args.architecture, args.heads).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    best = -1.0; best_step = None; start = time.perf_counter(); it = iter(loader)
    out.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        try: batch = next(it)
        except StopIteration: it = iter(loader); batch = next(it)
        x = batch['tokens'].to(device).long(); mask = batch['mask'].to(device).bool()
        if not mask.any(): raise ValueError('Empty valid token batch')
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            z = model.encode_z(x)
            q = model.vq.idx_to_z_q(model.vq.z_to_idx(z))
        # Only adapter parameters receive gradients; the original codec remains frozen.
        loss, ce, mse = adapter_training_loss(model, adapter, q, z, x, mask, args.ce_weight, args.latent_weight, args.latent_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 and step % args.validate_every != 0:
            elapsed = time.perf_counter() - start
            print(json.dumps({'step': step, 'train_ce': float(ce.detach()) if ce is not None else None, 'latent_mse': float(mse.detach()),
                              'valid_latent_fraction': float(latent_valid_mask(mask, q.shape[1]).float().mean()),
                              'examples_per_second_since_start': step * args.batch_size / max(elapsed, 1e-9)}), flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            metrics = valid_metrics(model, adapter, val, device)
            print(json.dumps({'step': step, 'train_ce': float(ce.detach()) if ce is not None else None, 'latent_mse': float(mse.detach()),
                              'valid_latent_fraction': float(latent_valid_mask(mask, q.shape[1]).float().mean()),
                              'validation': metrics, 'seconds': time.perf_counter()-start}), flush=True)
            if metrics['token_accuracy'] > best:
                best, best_step = metrics['token_accuracy'], step
                torch.save({'adapter': adapter.state_dict(), 'width': width, 'hidden': width*2, 'step': step,
                            'architecture': args.architecture, 'heads': args.heads,
                            'codec_sha256': sha256_file(args.checkpoint), 'config_sha256': sha256_file(args.config),
                            'training_cache_sha256': sha256_file(Path(cfg.data.cache_dir)/'manifest.json'),
                            'validation_cache_sha256': sha256_file(Path(cfg.train.validation_cache_dir)/'manifest.json'),
                            'latent_weight': args.latent_weight, 'ce_weight': args.ce_weight,
                            'latent_mask': args.latent_mask, 'validation': metrics}, out)
            adapter.train()
    print(f'Selected validation-best adapter: step={best_step}; accuracy={best:.2%}; file={out}', flush=True)


def evaluate(args, cfg, model, device):
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output exists')
    state = torch.load(args.adapter, map_location='cpu', weights_only=False)
    if state['codec_sha256'] != sha256_file(args.checkpoint) or state['config_sha256'] != sha256_file(args.config):
        raise SystemExit('Adapter codec checkpoint or config differs')
    adapter = make_latent_refiner(state['width'], state['hidden'], state.get('architecture', 'mlp'), state.get('heads', 4)).to(device)
    adapter.load_state_dict(state['adapter'], strict=True); adapter.eval()
    source, baseline = load_rows(args.input), load_rows(args.baseline)
    meta = json.loads(Path(args.baseline).with_suffix('.report.json').read_text(encoding='utf-8'))
    if source.keys() != baseline.keys() or meta['input_sha256'] != sha256_file(args.input) or Path(meta['checkpoint']).resolve() != Path(args.checkpoint).resolve():
        raise SystemExit('Baseline input, IDs or checkpoint differ')
    tok = load_base_tokenizer(cfg.tokenizer); pad, eos = int(tok.pad_token_id), int(tok.eos_token_id)
    context = int(cfg.model.context_length); rows = list(source.values()); output = []
    correct = {'baseline': 0, 'adapter': 0}; total = 0; timing = None
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            group = rows[start:start+args.batch_size]; packed = []
            for row in group:
                if row['target'] != baseline[row['id']]['target']: raise SystemExit(f'Target differs: {row["id"]}')
                ids = encode_text(tok, row['target'], append_eos=True)
                if len(ids) > context: raise SystemExit(f'Truncated input: {row["id"]}')
                packed.append(fixed_sequence(ids, context, pad, eos))
            x = torch.tensor([p[0] for p in packed], device=device, dtype=torch.long)
            mask = torch.tensor([p[1] for p in packed], device=device, dtype=torch.bool)
            q = model.vq.idx_to_z_q(model.encode(x))
            if args.benchmark and timing is None:
                timing = benchmark_decode(model, adapter, q, device)
            predictions = {'baseline': model.decoder(q).argmax(-1), 'adapter': model.decoder(adapter(q)).argmax(-1)}
            total += int(mask.sum())
            for name, pred in predictions.items(): correct[name] += int(((pred == x) & mask).sum())
            for i, row in enumerate(group):
                before = visible_text(tok, predictions['baseline'][i].tolist(), eos, pad)
                after = visible_text(tok, predictions['adapter'][i].tolist(), eos, pad)
                if before != baseline[row['id']]['reconstructed']: raise SystemExit(f'Baseline mismatch: {row["id"]}')
                output.append({'id': row['id'], 'baseline_exact': before == row['target'],
                               'adapter_exact': after == row['target'], 'selected': after})
            print(f'Processed {min(start+len(group),len(rows))}/{len(rows)}', flush=True)
    report = {'examples': len(output), 'valid_tokens': total, 'baseline_token_accuracy': correct['baseline']/total,
              'adapter_token_accuracy': correct['adapter']/total, 'baseline_exact': sum(r['baseline_exact'] for r in output),
              'adapter_exact': sum(r['adapter_exact'] for r in output),
              'gained_exact': sum(not r['baseline_exact'] and r['adapter_exact'] for r in output),
              'lost_exact': sum(r['baseline_exact'] and not r['adapter_exact'] for r in output),
              'input_sha256': sha256_file(args.input), 'codec_sha256': state['codec_sha256'],
              'adapter_sha256': sha256_file(args.adapter), 'adapter_step': state['step'],
              'scope': 'Known-input codec reconstruction. The adapter consumes only decoded VQ latents; continuous encoder vectors are training targets only.',
              'limitations': 'Frozen codec, reused development datasets; no compilation, behavior or generator test. Adapter adds parameters and decode compute; index count unchanged. Validation selected the adapter checkpoint.'}
    if timing is not None: report['decode_benchmark'] = timing
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in output: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    with out.with_suffix('.report.json').open('x', encoding='utf-8') as f: json.dump(report, f, indent=2); f.write('\n')
    print('\n=== Latent adapter comparison ===\n'+json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('train', 'evaluate'), required=True)
    for name in ('config', 'checkpoint', 'output'): p.add_argument('--'+name, required=True)
    p.add_argument('--adapter'); p.add_argument('--input'); p.add_argument('--baseline')
    p.add_argument('--steps', type=int, default=1000); p.add_argument('--validate-every', type=int, default=250)
    p.add_argument('--batch-size', type=int, default=8); p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--latent-weight', type=float, default=0.1); p.add_argument('--device', default='cuda')
    p.add_argument('--ce-weight', type=float, default=1.0)
    p.add_argument('--latent-mask', choices=('all', 'valid'), default='all')
    p.add_argument('--architecture', choices=('mlp', 'attention'), default='mlp'); p.add_argument('--heads', type=int, default=4)
    p.add_argument('--benchmark', action='store_true', help='Benchmark decoder forward on the first evaluation batch')
    args = p.parse_args()
    if (min(args.steps, args.validate_every, args.batch_size, args.heads) < 1 or args.lr <= 0 or
        args.latent_weight < 0 or args.ce_weight < 0 or not (args.latent_weight or args.ce_weight)):
        p.error('Invalid training arguments')
    if args.mode == 'evaluate' and not all((args.adapter, args.input, args.baseline)): p.error('Evaluation needs adapter, input and baseline')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    cfg, model = load_codec(args.config, args.checkpoint, device)
    if args.mode == 'train': train(args, cfg, model, device)
    else: evaluate(args, cfg, model, device)


if __name__ == '__main__': main()
