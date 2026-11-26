"""Calibrate an inference-only confidence gate for a frozen latent adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from models.latent_refiner import LatentRefiner
from training.data import encode_text, fixed_sequence, load_base_tokenizer, sha256_file
from evaluation.codec_latent_refiner_screen import load_codec
from evaluation.ensemble_codec_methods import load_rows
from evaluation.godot_csharp_codec_eval import visible_text


def split_source(source):
    if not isinstance(source, str) or not source: raise ValueError('A source path is required for source-grouped calibration')
    return 'calibration' if int.from_bytes(hashlib.sha256(source.encode('utf-8')).digest()[:8], 'big') % 2 == 0 else 'evaluation'


def select(row, threshold):
    return threshold is not None and row['changed'] and row['confidence_delta'] >= threshold


def choose_threshold(rows):
    """Choose only from calibration labels; ties prefer fewer adapter uses."""
    if not rows: raise ValueError('Empty calibration partition')
    candidates = [None] + sorted({r['confidence_delta'] for r in rows if r['changed']})
    def metric(threshold):
        choices = [select(row, threshold) for row in rows]
        exact = sum(row['adapter_exact'] if use else row['baseline_exact'] for row, use in zip(rows, choices))
        return exact, -sum(choices)
    return max(candidates, key=metric)


def summarize(rows, threshold):
    choices = [select(row, threshold) for row in rows]
    baseline = sum(row['baseline_exact'] for row in rows)
    selected = sum(row['adapter_exact'] if use else row['baseline_exact'] for row, use in zip(rows, choices))
    return {'examples': len(rows), 'baseline_exact': baseline, 'selected_exact': selected,
            'adapter_only_exact': sum(row['adapter_exact'] for row in rows), 'adapter_selected': sum(choices),
            'gained_exact': sum(not r['baseline_exact'] and use and r['adapter_exact'] for r, use in zip(rows, choices)),
            'lost_exact': sum(r['baseline_exact'] and use and not r['adapter_exact'] for r, use in zip(rows, choices))}


def infer(model, adapter, cfg, tok, input_path, baseline_path, saved_path, device, batch_size):
    source, baseline, saved = (load_rows(path) for path in (input_path, baseline_path, saved_path))
    if source.keys() != baseline.keys() or source.keys() != saved.keys(): raise ValueError('Input IDs differ')
    base_report = json.loads(Path(baseline_path).with_suffix('.report.json').read_text(encoding='utf-8'))
    saved_report = json.loads(Path(saved_path).with_suffix('.report.json').read_text(encoding='utf-8'))
    if (base_report['input_sha256'] != sha256_file(input_path) or saved_report['input_sha256'] != sha256_file(input_path) or
        sha256_file(base_report['checkpoint']) != saved_report['codec_sha256']): raise ValueError('Saved report input or codec differs')
    pad, eos, context = int(tok.pad_token_id), int(tok.eos_token_id), int(cfg.model.context_length)
    records = list(source.values()); output = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            group = records[start:start+batch_size]; packed = []
            for row in group:
                if row['target'] != baseline[row['id']]['target']: raise ValueError(f'Target differs: {row["id"]}')
                ids = encode_text(tok, row['target'], append_eos=True)
                if len(ids) > context: raise ValueError(f'Truncated input: {row["id"]}')
                packed.append(fixed_sequence(ids, context, pad, eos))
            x = torch.tensor([r[0] for r in packed], device=device, dtype=torch.long)
            mask = torch.tensor([r[1] for r in packed], device=device, dtype=torch.bool)
            q = model.vq.idx_to_z_q(model.encode(x)); predictions = {}; confidence = {}
            for name in ('baseline', 'adapter'):
                logits = model.decoder(q if name == 'baseline' else adapter(q)).float()
                values, indices = logits.max(-1)
                # Mean log probability of the path's own top prediction; no original token IDs enter the score.
                confidence[name] = (((values - torch.logsumexp(logits, dim=-1)) * mask).sum(-1) / mask.sum(-1).clamp_min(1)).cpu().tolist()
                predictions[name] = indices.cpu().tolist()
                del logits, values, indices
            for i, row in enumerate(group):
                key = row['id']; target = row['target']
                before = visible_text(tok, predictions['baseline'][i], eos, pad)
                after = visible_text(tok, predictions['adapter'][i], eos, pad)
                if before != baseline[key]['reconstructed'] or after != saved[key]['selected']:
                    raise ValueError(f'Recomputed prediction differs from saved output: {key}')
                output.append({'id': key, 'source': row.get('source'), 'baseline': before, 'adapter': after,
                    'changed': before != after, 'confidence_delta': confidence['adapter'][i] - confidence['baseline'][i],
                    'baseline_exact': before == target, 'adapter_exact': after == target})
            print(f'Processed {min(start+len(group),len(records))}/{len(records)}: {input_path}', flush=True)
    return output, saved_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config','checkpoint','adapter','methods-input','methods-baseline','methods-adapter',
                 'external-input','external-baseline','external-adapter','output-dir'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--batch-size', type=int, default=8); p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.batch_size < 1: p.error('batch size must be positive')
    out = Path(args.output_dir)
    for name in ('methods.jsonl','external.jsonl','report.json'):
        if (out/name).exists(): raise SystemExit(f'Output exists: {out/name}')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    cfg, model = load_codec(args.config, args.checkpoint, device)
    state = torch.load(args.adapter, map_location='cpu', weights_only=False)
    if state['codec_sha256'] != sha256_file(args.checkpoint) or state['config_sha256'] != sha256_file(args.config):
        raise SystemExit('Adapter codec or config differs')
    adapter = LatentRefiner(state['width'], state['hidden']).to(device)
    adapter.load_state_dict(state['adapter'], strict=True); adapter.eval()
    tok = load_base_tokenizer(cfg.tokenizer)
    methods, method_report = infer(model, adapter, cfg, tok, args.methods_input, args.methods_baseline, args.methods_adapter, device, args.batch_size)
    external, external_report = infer(model, adapter, cfg, tok, args.external_input, args.external_baseline, args.external_adapter, device, args.batch_size)
    for report in (method_report, external_report):
        if report['adapter_sha256'] != sha256_file(args.adapter) or report['codec_sha256'] != sha256_file(args.checkpoint):
            raise SystemExit('Saved adapter or codec hash differs')
    calibration = [r for r in methods if split_source(r['source']) == 'calibration']
    evaluation = [r for r in methods if split_source(r['source']) == 'evaluation']
    if not calibration or not evaluation: raise SystemExit('Empty source-grouped partition')
    threshold = choose_threshold(calibration)
    for row in methods + external: row['use_adapter'] = select(row, threshold)
    report = {'threshold': threshold, 'threshold_meaning': 'Minimum change in mean top-token log probability; null means always use baseline',
              'calibration': summarize(calibration, threshold), 'methods_evaluation': summarize(evaluation, threshold),
              'external_evaluation': summarize(external, threshold), 'methods_input_sha256': sha256_file(args.methods_input),
              'external_input_sha256': sha256_file(args.external_input), 'adapter_sha256': sha256_file(args.adapter),
              'scope': 'Known-input VQ reconstruction; gate uses decoded logits and input length; target labels choose threshold on calibration files and score all sets offline.',
              'limitations': 'Calibration labels are not inference inputs. Methods and external sets have been repeatedly inspected; source-file grouping does not exclude semantic duplicates. No compilation, behavior, generation or end-to-end speed test; gate runs both decoder paths and needs input length metadata.'}
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [('methods', methods), ('external', external)]:
        with (out/(name+'.jsonl')).open('x', encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    with (out/'report.json').open('x', encoding='utf-8') as f: json.dump(report, f, indent=2); f.write('\n')
    print('\n=== Latent adapter confidence gate ===\n'+json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
