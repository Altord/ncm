"""Screen fixed self-reencoding passes of a frozen known-input codec."""
import argparse
import json
from pathlib import Path
import torch
from evaluation.codec_latent_refiner_screen import load_codec
from evaluation.ensemble_codec_methods import load_rows
from evaluation.godot_csharp_codec_eval import visible_text
from training.data import encode_text, fixed_sequence, load_base_tokenizer, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'input', 'baseline', 'output'): p.add_argument('--'+name, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args(); out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output exists')
    source, baseline = load_rows(args.input), load_rows(args.baseline)
    report = json.loads(Path(args.baseline).with_suffix('.report.json').read_text())
    if (source.keys() != baseline.keys() or report['input_sha256'] != sha256_file(args.input)
        or Path(report['checkpoint']).resolve() != Path(args.checkpoint).resolve()):
        raise SystemExit('Baseline provenance differs')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    cfg, model = load_codec(args.config, args.checkpoint, device)
    tok = load_base_tokenizer(cfg.tokenizer); context = int(cfg.model.context_length)
    eos, pad = int(tok.eos_token_id), int(tok.pad_token_id)
    rows = list(source.values()); output = []; totals = [0, 0, 0]; valid = 0
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            group = rows[start:start+args.batch_size]; packed = []
            for row in group:
                if baseline[row['id']]['target'] != row['target']: raise SystemExit('Target differs')
                ids = encode_text(tok, row['target'], True)
                if len(ids) > context: raise SystemExit('Input truncated')
                packed.append(fixed_sequence(ids, context, pad, eos))
            original = torch.tensor([r[0] for r in packed], device=device, dtype=torch.long)
            mask = torch.tensor([r[1] for r in packed], device=device, dtype=torch.bool)
            x = original; predictions = []
            for step in range(3):
                prediction = model.decode(model.encode(x)).argmax(-1)
                predictions.append(prediction)
                totals[step] += int(((prediction == original) & mask).sum())
                # Original valid length is carried as metadata, never the target token IDs.
                x = torch.where(mask, prediction, torch.full_like(prediction, pad))
            valid += int(mask.sum())
            for i, row in enumerate(group):
                texts = [visible_text(tok, prediction[i].tolist(), eos, pad) for prediction in predictions]
                if texts[0] != baseline[row['id']]['reconstructed']: raise SystemExit('Baseline mismatch')
                output.append({'id': row['id'], 'selected': texts, 'exact': [text == row['target'] for text in texts]})
            print(f'Processed {min(start+len(group),len(rows))}/{len(rows)}', flush=True)
    summary = {'examples': len(output), 'valid_tokens': valid, 'token_accuracy': [n/valid for n in totals],
               'exact': [sum(row['exact'][i] for row in output) for i in range(3)],
               'input_sha256': sha256_file(args.input), 'codec_sha256': sha256_file(args.checkpoint),
               'scope': 'Known-input codec; subsequent fixed passes encode their own previous predictions and length metadata.',
               'limitations': 'Each pass adds codec compute. No oracle selection, compile, behavior, generation or fresh test.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in output: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    with out.with_suffix('.report.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2); f.write('\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
