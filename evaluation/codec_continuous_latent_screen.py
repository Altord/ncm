"""Compare frozen discrete reconstruction with a continuous-latent diagnostic."""
import argparse
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from training.data import encode_text, fixed_sequence, load_base_tokenizer, sha256_file
from evaluation.ensemble_codec_methods import load_rows
from evaluation.godot_csharp_codec_eval import visible_text


def evaluate_predictions(prediction, target, mask):
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError('Prediction, target and mask shapes differ')
    return int(((prediction == target) & mask).sum()), int(mask.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'input', 'baseline', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.batch_size < 1: p.error('Batch size must be positive')
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output exists')
    cfg = OmegaConf.load(args.config); OmegaConf.resolve(cfg)
    source, baseline = load_rows(args.input), load_rows(args.baseline)
    meta = json.loads(Path(args.baseline).with_suffix('.report.json').read_text(encoding='utf-8'))
    if (source.keys() != baseline.keys() or meta['input_sha256'] != sha256_file(args.input) or
        Path(meta['checkpoint']).resolve() != Path(args.checkpoint).resolve()):
        raise SystemExit('Baseline input, IDs or checkpoint do not match')
    tok = load_base_tokenizer(cfg.tokenizer)
    pad, eos, context = int(tok.pad_token_id), int(tok.eos_token_id), int(cfg.model.context_length)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    vcfg = OmegaConf.to_container(cfg.vqvae, resolve=True); vcfg.pop('checkpoint_path', None)
    if vcfg.get('autoregressive_decoder', False): raise SystemExit('Parallel decoder required')
    model = VQVAE(**OmegaConf.create(vcfg))
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    model.to(device).eval().requires_grad_(False)
    rows, output = list(source.values()), []
    correct = {'discrete': 0, 'continuous': 0}; valid = 0
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            group = rows[start:start + args.batch_size]; packed = []
            for row in group:
                key = row['id']; target = row['target']
                if baseline[key]['target'] != target: raise SystemExit(f'Target mismatch: {key}')
                ids = encode_text(tok, target, append_eos=True)
                if len(ids) > context: raise SystemExit(f'Truncated input: {key}')
                packed.append(fixed_sequence(ids, context, pad, eos))
            x = torch.tensor([r[0] for r in packed], dtype=torch.long, device=device)
            mask = torch.tensor([r[1] for r in packed], dtype=torch.bool, device=device)
            z = model.encode_z(x)
            indices = model.vq.z_to_idx(z)
            discrete = model.decode(indices).argmax(-1)
            continuous = model.decoder(z).argmax(-1)
            for name, pred in [('discrete', discrete), ('continuous', continuous)]:
                n, denom = evaluate_predictions(pred, x, mask); correct[name] += n
                if name == 'discrete': valid += denom
            for i, row in enumerate(group):
                old = visible_text(tok, discrete[i].tolist(), eos, pad)
                new = visible_text(tok, continuous[i].tolist(), eos, pad)
                if old != baseline[row['id']]['reconstructed']:
                    raise SystemExit(f'Recomputed discrete baseline differs: {row["id"]}')
                output.append({'id': row['id'], 'baseline_exact': old == row['target'],
                               'continuous_exact': new == row['target'], 'continuous': new})
            print(f'Processed {min(start+len(group), len(rows))}/{len(rows)}', flush=True)
    report = {'examples': len(output), 'valid_tokens': valid,
              'discrete_token_accuracy': correct['discrete']/valid,
              'continuous_token_accuracy': correct['continuous']/valid,
              'discrete_exact': sum(r['baseline_exact'] for r in output),
              'continuous_exact': sum(r['continuous_exact'] for r in output),
              'gained_exact': sum(not r['baseline_exact'] and r['continuous_exact'] for r in output),
              'lost_exact': sum(r['baseline_exact'] and not r['continuous_exact'] for r in output),
              'input_sha256': sha256_file(args.input), 'checkpoint_sha256': sha256_file(args.checkpoint),
              'scope': 'Frozen known-target codec; continuous encoder vectors are sent directly to the decoder instead of VQ indices.',
              'limitations': 'Decoder was trained on quantized values, so continuous inputs may be out of distribution; this is not an upper bound or a usable discrete codec. Continuous vectors require storage and are not generator outputs. Reused datasets; no compilation or behavioral test.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in output: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    with out.with_suffix('.report.json').open('x', encoding='utf-8') as f: json.dump(report, f, indent=2); f.write('\n')
    print('\n=== Frozen continuous-latent comparison ===\n'+json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
