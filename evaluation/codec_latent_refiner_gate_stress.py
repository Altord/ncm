"""Apply an already-calibrated latent-adapter gate to codec stress examples."""
import argparse
import json
import math
from pathlib import Path
import torch
from models.latent_refiner import make_latent_refiner
from training.data import load_base_tokenizer, sha256_file
from evaluation.codec_latent_refiner_screen import load_codec
from evaluation.codec_latent_refiner_gate import infer, select, summarize


def apply_gate(rows, threshold):
    output = []
    for row in rows:
        use = select(row, threshold)
        output.append({'id': row['id'], 'selected': row['adapter'] if use else row['baseline'],
                       'use_adapter': use, 'baseline_exact': row['baseline_exact'],
                       'adapter_exact': row['adapter_exact'], 'selected_exact': row['adapter_exact'] if use else row['baseline_exact'],
                       'confidence_delta': row['confidence_delta']})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'adapter', 'input', 'baseline', 'adapter-output', 'gate-report', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    if args.batch_size < 1: p.error('batch size must be positive')
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output exists')
    gate = json.loads(Path(args.gate_report).read_text(encoding='utf-8'))
    threshold = gate['threshold']
    if threshold is not None and (not isinstance(threshold, (int, float)) or not math.isfinite(threshold)):
        raise ValueError('Invalid saved gate threshold')
    if gate['adapter_sha256'] != sha256_file(args.adapter): raise SystemExit('Calibrated adapter hash differs')
    state = torch.load(args.adapter, map_location='cpu', weights_only=False)
    if state['codec_sha256'] != sha256_file(args.checkpoint) or state['config_sha256'] != sha256_file(args.config):
        raise SystemExit('Adapter codec or config differs')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    cfg, model = load_codec(args.config, args.checkpoint, device)
    adapter = make_latent_refiner(state['width'], state['hidden'], state.get('architecture', 'mlp'), state.get('heads', 4)).to(device)
    adapter.load_state_dict(state['adapter'], strict=True); adapter.eval()
    tok = load_base_tokenizer(cfg.tokenizer)
    rows, report = infer(model, adapter, cfg, tok, args.input, args.baseline, args.adapter_output, device, args.batch_size)
    if report['adapter_sha256'] != sha256_file(args.adapter) or report['codec_sha256'] != sha256_file(args.checkpoint):
        raise SystemExit('Saved stress adapter output does not match checkpoint')
    output = apply_gate(rows, threshold)
    summary = {**summarize(rows, threshold), 'threshold': threshold,
               'input_sha256': sha256_file(args.input), 'codec_sha256': sha256_file(args.checkpoint),
               'adapter_sha256': sha256_file(args.adapter), 'gate_report_sha256': sha256_file(args.gate_report),
               'scope': 'Known-target codec reconstruction; gate uses decoder confidence and original-length metadata. Threshold fixed from prior calibration.',
               'limitations': 'Repeatedly used stress set; no new calibration, training, behavioral or generation test. Two decoder passes and length metadata are required.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in output: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    with out.with_suffix('.report.json').open('x', encoding='utf-8') as f: json.dump(summary, f, indent=2); f.write('\n')
    print('\n=== Fixed gate stress comparison ===\n'+json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
