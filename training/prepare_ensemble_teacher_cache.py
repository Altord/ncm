"""Cache ensemble teacher top-k distributions for original codec training data."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from training.data import MemmapCache, sha256_file


def topk_ensemble(a, b, k):
    if a.shape != b.shape or not 2 <= k <= a.shape[-1]:
        raise ValueError('Incompatible teacher logits or top-k')
    values, indices = (a.float() + b.float()).topk(k, dim=-1)
    # The equal-logit ensemble uses (a+b)/2. Divide before softmax to preserve its temperature.
    return indices, torch.softmax(values * 0.5, dim=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config-a', 'checkpoint-a', 'config-b', 'checkpoint-b', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--topk', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.batch_size < 1: p.error('batch size must be positive')
    out = Path(args.output)
    if out.exists(): raise SystemExit(f'Cache directory already exists: {out}')
    a, b = OmegaConf.load(args.config_a), OmegaConf.load(args.config_b)
    OmegaConf.resolve(a); OmegaConf.resolve(b)
    if (OmegaConf.to_container(a.tokenizer, resolve=True) != OmegaConf.to_container(b.tokenizer, resolve=True) or
        int(a.model.context_length) != int(b.model.context_length) or
        int(a.vqvae.in_vocab_size) != int(b.vqvae.in_vocab_size) or
        sha256_file(a.data.input) != sha256_file(b.data.input)):
        raise SystemExit('Teacher tokenizer, context, vocabulary or training data differs')
    dataset = MemmapCache(a.data.cache_dir)
    if (dataset.manifest.get('source_sha256') != sha256_file(a.data.input) or
        int(dataset.manifest.get('sequence_length', -1)) != int(a.model.context_length)):
        raise SystemExit('Training cache does not match teacher training data/context')
    if not 2 <= args.topk <= int(a.vqvae.in_vocab_size): raise SystemExit('Invalid top-k')
    if any(name in dataset.manifest['arrays'] for name in ('ensemble_topk_ids', 'ensemble_topk_probs')):
        raise SystemExit('Source cache already contains ensemble arrays')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    def model(cfg, checkpoint):
        vc = OmegaConf.to_container(cfg.vqvae, resolve=True)
        vc.pop('checkpoint_path', None)
        if vc.get('autoregressive_decoder', False): raise ValueError('Parallel teacher decoder required')
        m = VQVAE(**OmegaConf.create(vc))
        m.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
        return m.to(device).eval().requires_grad_(False)
    first, second = model(a, args.checkpoint_a), model(b, args.checkpoint_b)
    out.mkdir(parents=True, exist_ok=False)
    shape = (len(dataset), int(a.model.context_length), args.topk)
    ids = np.memmap(out/'ensemble_ids.bin.tmp', mode='w+', dtype=np.int32, shape=shape)
    probs = np.memmap(out/'ensemble_probs.bin.tmp', mode='w+', dtype=np.float32, shape=shape)
    done = correct = tokens = 0
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0):
            x = batch['tokens'].to(device).long()
            valid = batch['mask'].to(device).bool()
            left = first.decode(first.encode(x))
            right = second.decode(second.encode(x))
            indices, probabilities = topk_ensemble(left, right, args.topk)
            n = x.shape[0]
            ids[done:done+n] = indices.cpu().numpy().astype(np.int32)
            probs[done:done+n] = probabilities.cpu().numpy()
            correct += int(((indices[..., 0] == x) & valid).sum())
            tokens += int(valid.sum())
            done += n
            if done % 128 == 0 or done == len(dataset): print(f'Cached {done}/{len(dataset)} sequences', flush=True)
    ids.flush(); probs.flush(); del ids, probs
    for spec in dataset.manifest['arrays'].values():
        filename = spec['file']
        if Path(filename).name != filename or filename in ('ensemble_ids.bin', 'ensemble_probs.bin'):
            raise ValueError('Unsafe or conflicting source cache filename')
        shutil.copyfile(dataset.directory/filename, out/filename)
    (out/'ensemble_ids.bin.tmp').replace(out/'ensemble_ids.bin')
    (out/'ensemble_probs.bin.tmp').replace(out/'ensemble_probs.bin')
    manifest = dict(dataset.manifest)
    manifest['arrays'] = dict(manifest['arrays'])
    manifest['arrays']['ensemble_topk_ids'] = {'file':'ensemble_ids.bin', 'dtype':'int32', 'shape':[shape[1], shape[2]]}
    manifest['arrays']['ensemble_topk_probs'] = {'file':'ensemble_probs.bin', 'dtype':'float32', 'shape':[shape[1], shape[2]]}
    manifest['ensemble_distillation'] = {'topk': args.topk, 'checkpoint_sha256': [sha256_file(args.checkpoint_a), sha256_file(args.checkpoint_b)],
        'config_sha256': [sha256_file(args.config_a), sha256_file(args.config_b)], 'training_source_sha256': sha256_file(a.data.input),
        'precision': 'fp32 inference', 'teacher_correct_valid_fraction': correct/max(1,tokens)}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(f'Cached ensemble teacher; correct valid-token fraction: {correct/max(1,tokens):.2%}', flush=True)


if __name__ == '__main__': main()
