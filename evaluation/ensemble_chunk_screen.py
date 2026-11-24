"""Test fixed shorter chunks with independently encoded frozen codec ensembles."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from training.data import encode_text, fixed_sequence, load_base_tokenizer, sha256_file
from evaluation.ensemble_codec_methods import load_rows, combine
from evaluation.godot_csharp_codec_eval import visible_text


def partition(ids, size):
    if size < 1: raise ValueError('Chunk size must be positive')
    return [ids[i:i + size] for i in range(0, len(ids), size)] or [[]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config-a', 'checkpoint-a', 'config-b', 'checkpoint-b', 'input', 'baseline', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--chunk-size', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output exists')
    cfgs = [OmegaConf.load(args.config_a), OmegaConf.load(args.config_b)]
    for cfg in cfgs: OmegaConf.resolve(cfg)
    a, b = cfgs
    if (OmegaConf.to_container(a.tokenizer, resolve=True) != OmegaConf.to_container(b.tokenizer, resolve=True) or
        int(a.model.context_length) != int(b.model.context_length) or int(a.vqvae.in_vocab_size) != int(b.vqvae.in_vocab_size)):
        raise SystemExit('Codec tokenizer configuration, context or vocabulary differs')
    context = int(a.model.context_length)
    if not 1 <= args.chunk_size < context or args.batch_size < 1: p.error('Invalid chunk or batch size')
    source, baseline = load_rows(args.input), load_rows(args.baseline)
    meta = json.loads(Path(args.baseline).with_suffix('.report.json').read_text(encoding='utf-8'))
    if (source.keys() != baseline.keys() or meta['input_sha256'] != sha256_file(args.input) or
        meta['checkpoint_a_sha256'] != sha256_file(args.checkpoint_a) or meta['checkpoint_b_sha256'] != sha256_file(args.checkpoint_b)):
        raise SystemExit('Baseline input, IDs or checkpoints differ')
    tok = load_base_tokenizer(a.tokenizer)
    eos, pad = int(tok.eos_token_id), int(tok.pad_token_id)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    models = []
    for cfg, checkpoint in zip(cfgs, (args.checkpoint_a, args.checkpoint_b)):
        vcfg = OmegaConf.to_container(cfg.vqvae, resolve=True)
        vcfg.pop('checkpoint_path', None)
        if vcfg.get('autoregressive_decoder', False): raise SystemExit('Parallel decoder required')
        model = VQVAE(**OmegaConf.create(vcfg))
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
        models.append(model.to(device).eval().requires_grad_(False))
    output, groups = [], defaultdict(lambda: {'examples': 0, 'baseline_exact': 0, 'chunk_exact': 0})
    correct = total = chunks_total = 0
    with torch.inference_mode():
        for index, (key, row) in enumerate(source.items(), 1):
            target = row['target']
            raw = encode_text(tok, target, append_eos=False)
            if len(raw) + 1 > context: raise SystemExit(f'Baseline input would truncate: {key}')
            if visible_text(tok, raw + [eos], eos, pad) != target: raise SystemExit(f'Tokenizer is not text-exact: {key}')
            if bool(baseline[key]['ensemble_exact']) != (baseline[key]['selected'] == target): raise SystemExit(f'Baseline score mismatch: {key}')
            parts = partition(raw, args.chunk_size)
            decoded = []
            for start in range(0, len(parts), args.batch_size):
                subset = parts[start:start + args.batch_size]
                packed = [fixed_sequence(part + [eos], context, pad, eos)[0] for part in subset]
                x = torch.tensor(packed, dtype=torch.long, device=device)
                pred = combine(models[0].decode(models[0].encode(x)), models[1].decode(models[1].encode(x)))
                for i, part in enumerate(subset):
                    guess = pred[i, :len(part)].tolist()
                    correct += sum(x == y for x, y in zip(part, guess))
                    total += len(part)
                    decoded.extend(guess)
            chunks_total += len(parts)
            selected = visible_text(tok, decoded, eos, pad)
            old, new = bool(baseline[key]['ensemble_exact']), selected == target
            output.append({'id': key, 'selected': selected, 'baseline_exact': old, 'chunk_exact': new, 'chunks': len(parts)})
            project = row.get('source', '').split('/')[0] or 'unknown'
            length = len(raw) + 1
            bucket = '1-64' if length <= 64 else '65-128' if length <= 128 else '129-192' if length <= 192 else '193-256'
            for name in ('all', 'project:' + project, 'length:' + bucket):
                g = groups[name]; g['examples'] += 1; g['baseline_exact'] += old; g['chunk_exact'] += new
            if index % 32 == 0 or index == len(source): print(f'Processed {index}/{len(source)}', flush=True)
    report = {'examples': len(output), 'chunk_size_tokens': args.chunk_size, 'chunks': chunks_total,
              'token_accuracy_excluding_eos': correct / total if total else None,
              'baseline_exact': sum(r['baseline_exact'] for r in output), 'chunk_exact': sum(r['chunk_exact'] for r in output),
              'gained_exact': sum(not r['baseline_exact'] and r['chunk_exact'] for r in output),
              'lost_exact': sum(r['baseline_exact'] and not r['chunk_exact'] for r in output),
              'groups': dict(sorted(groups.items())), 'input_sha256': sha256_file(args.input),
              'baseline_sha256': sha256_file(args.baseline),
              'scope': 'Known-target two-codec reconstruction; each fixed chunk is separately encoded by both codecs. Original token count defines chunk boundaries and is required to reassemble.',
              'limitations': 'Reused datasets; token pieces may split syntax or identifiers. No compilation, behavioral or generator test. Extra chunks increase latent streams and compute; no size or speed measurement. Matching tokenizer configuration does not verify tokenizer file identity.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in output: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    with out.with_suffix('.report.json').open('x', encoding='utf-8') as f: json.dump(report, f, indent=2); f.write('\n')
    print('\n=== Short-chunk ensemble comparison ===\n' + json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
