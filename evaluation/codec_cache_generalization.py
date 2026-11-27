"""Compare frozen codec and adapter on sampled training and validation caches."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from models.latent_refiner import make_latent_refiner
from training.data import MemmapCache, sha256_file
from evaluation.codec_latent_refiner_screen import load_codec


def sample_indices(count, limit, seed):
    if count < 1 or limit < 0: raise ValueError('Invalid sample size')
    if not limit or limit >= count: return list(range(count))
    return torch.randperm(count, generator=torch.Generator().manual_seed(seed))[:limit].tolist()


def measure(model, adapter, loader, device):
    stats = {name: {'correct': 0, 'exact': 0, 'ce_sum': 0.0} for name in ('codec', 'adapter')}
    examples = tokens = 0
    with torch.no_grad():
        for batch in loader:
            x = batch['tokens'].to(device).long(); mask = batch['mask'].to(device).bool()
            q = model.vq.idx_to_z_q(model.encode(x)); examples += x.shape[0]; tokens += int(mask.sum())
            for name in ('codec', 'adapter'):
                logits = model.decoder(q if name == 'codec' else adapter(q))
                pred = logits.argmax(-1); per = F.cross_entropy(logits.float().transpose(1, 2), x, reduction='none')
                stats[name]['correct'] += int(((pred == x) & mask).sum())
                stats[name]['exact'] += int(((pred == x) | ~mask).all(-1).sum())
                stats[name]['ce_sum'] += float((per * mask).sum())
                del logits, per, pred
    if not tokens or not examples: raise ValueError('Empty valid dataset')
    return {'examples': examples, 'valid_tokens': tokens,
            **{name: {'token_accuracy': item['correct']/tokens, 'exact': item['exact'],
                       'mean_token_cross_entropy': item['ce_sum']/tokens} for name, item in stats.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'adapter', 'output'): p.add_argument('--'+name, required=True)
    p.add_argument('--train-limit', type=int, default=512); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=8); p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.batch_size < 1 or args.train_limit < 1: p.error('Batch size and training limit must be positive')
    out = Path(args.output)
    if out.exists(): raise SystemExit('Output exists')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    cfg, model = load_codec(args.config, args.checkpoint, device)
    state = torch.load(args.adapter, map_location='cpu', weights_only=False)
    if state['codec_sha256'] != sha256_file(args.checkpoint) or state['config_sha256'] != sha256_file(args.config):
        raise SystemExit('Adapter checkpoint or configuration differs')
    adapter = make_latent_refiner(state['width'], state['hidden'], state.get('architecture', 'mlp'), state.get('heads', 4)).to(device)
    adapter.load_state_dict(state['adapter'], strict=True); adapter.eval()
    train_cache, val_cache = MemmapCache(cfg.data.cache_dir), MemmapCache(cfg.train.validation_cache_dir)
    if (state['training_cache_sha256'] != sha256_file(Path(cfg.data.cache_dir)/'manifest.json') or
        state['validation_cache_sha256'] != sha256_file(Path(cfg.train.validation_cache_dir)/'manifest.json')):
        raise SystemExit('Cache manifests differ from adapter training')
    indices = sample_indices(len(train_cache), args.train_limit, args.seed)
    train = measure(model, adapter, DataLoader(Subset(train_cache, indices), batch_size=args.batch_size), device)
    print(f'Measured {len(indices)} training sequences', flush=True)
    validation = measure(model, adapter, DataLoader(val_cache, batch_size=args.batch_size), device)
    report = {'training_sample': train, 'validation': validation, 'sample_seed': args.seed,
              'sample_indices_sha256': hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
              'codec_sha256': sha256_file(args.checkpoint), 'adapter_sha256': sha256_file(args.adapter),
              'scope': 'Known-input frozen codec reconstruction on a deterministic training-cache sample and the full cached validation set.',
              'limitations': 'Training sample is descriptive, not independent. Validation was used to select the adapter checkpoint. Token-weighted CE and exact rates depend on sequence length and data mix. No compilation, generation, or fresh holdout tuning.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f: json.dump(report, f, indent=2); f.write('\n')
    print('\n=== Training versus validation cache ===\n'+json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
