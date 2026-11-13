"""Bounded compile-guided token alternatives for known-input codec diagnostics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from training.data import load_base_tokenizer, encode_text, fixed_sequence
from evaluation.godot_csharp_codec_eval import file_sha256, visible_text
from evaluation.godot_csharp_generator_pilot import compile_candidate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--baseline', required=True, help='Existing compile-evaluation JSONL')
    p.add_argument('--max-attempts', type=int, default=4)
    p.add_argument('--framework', default='net8.0')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.max_attempts < 1:
        p.error('--max-attempts must be positive')
    input_path, baseline = Path(args.input), Path(args.baseline)
    report_path = baseline.with_suffix('.report.json')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('input_sha256') != file_sha256(input_path) or
        Path(report.get('checkpoint', '')).resolve() != Path(args.checkpoint).resolve() or
        not report.get('compile_requested')):
        raise SystemExit('Baseline does not match input, checkpoint, or compilation settings')
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists():
        raise SystemExit('Output exists; choose a new path')
    rows = [json.loads(line) for line in baseline.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != report.get('examples'):
        raise SystemExit('Baseline row count differs from its report')
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    tokenizer = load_base_tokenizer(cfg.tokenizer)
    vqcfg = OmegaConf.to_container(cfg.vqvae, resolve=True)
    vqcfg.pop('checkpoint_path', None)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = VQVAE(**OmegaConf.create(vqcfg))
    model.load_state_dict(ckpt['model'], strict=True)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('GPU unavailable')
    model.to(device).eval().requires_grad_(False)
    eos, pad = int(tokenizer.eos_token_id), int(tokenizer.pad_token_id)
    context = int(cfg.model.context_length)
    output = []
    attempts = recovered = 0
    start = time.perf_counter()
    for index, row in enumerate(rows):
        status = row.get('reconstruction_compile', {}).get('status')
        if row.get('original_compile', {}).get('status') != 'passed' or status not in ('passed', 'failed', 'empty'):
            raise SystemExit(f'Unusable original or reconstruction compile result on row {index}')
        selected = row['reconstructed']
        chosen = 'baseline'
        row_attempts = 0
        if status != 'passed':
            ids = encode_text(tokenizer, row['target'], append_eos=True)
            if len(ids) > context:
                raise SystemExit('Truncated target detected; cannot compare this example')
            padded, _ = fixed_sequence(ids, context, pad, eos)
            x = torch.tensor([padded], dtype=torch.long, device=device)
            with torch.inference_mode():
                logits = model.decode(model.encode(x))[0].float()
                values, candidates = logits.topk(2, dim=-1)
                top = candidates[:, 0].tolist()
                alternative = candidates[:, 1].tolist()
                margin = (values[:, 0] - values[:, 1]).tolist()
            if visible_text(tokenizer, top.copy(), eos, pad) != selected:
                raise SystemExit(f'Baseline reconstruction mismatch at row {index}; stopping')
            # Only the model's predicted EOS determines the search window, not target length.
            search_end = top.index(eos) + 1 if eos in top else len(top)
            tried = {selected}
            for pos in sorted(range(search_end), key=lambda i: margin[i]):
                changed = top.copy()
                changed[pos] = alternative[pos]
                candidate = visible_text(tokenizer, changed, eos, pad)
                if not candidate.strip() or candidate in tried:
                    continue
                tried.add(candidate)
                result = compile_candidate(candidate, args.framework)
                row_attempts += 1
                attempts += 1
                if result['status'] not in ('passed', 'failed', 'empty'):
                    raise SystemExit(f'Compiler unavailable or timed out at row {index}: {result}')
                if result['status'] == 'passed':
                    selected, status, chosen = candidate, 'passed', 'compile_guided_alternative'
                    recovered += 1
                    break
                if row_attempts >= args.max_attempts:
                    break
        output.append({'id': row['id'], 'baseline': row['reconstructed'],
                       'selected': selected, 'selection': chosen, 'compile_status': status,
                       # Used for offline scoring only, never candidate selection.
                       'selected_text_exact': selected == row['target'],
                       'attempts': row_attempts})
        print(f'Processed {index+1}/{len(rows)}; repaired={recovered}; compiler attempts={attempts}', flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in output:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    result = {'examples': len(output), 'baseline_compiles': report['reconstruction_compile_passes'],
              'selected_compiles': sum(r['compile_status'] == 'passed' for r in output),
              'baseline_exact_tokens': report['exact_sequence_matches'],
              'baseline_exact_text': sum(r['reconstructed'] == r['target'] for r in rows),
              'selected_exact_text': sum(r['selected_text_exact'] for r in output),
              'successful_repairs': recovered, 'compiler_attempts': attempts,
              'max_attempts_per_failed_example': args.max_attempts,
              'elapsed_seconds': time.perf_counter()-start,
              'note': 'Codec round trip: original input is encoded into VQ latents. '
                      'Candidates selected only by compiler acceptance; original text used for offline scoring only. '
                      'Compilation is not semantic correctness. Selected exact uses decoded text equality; '
                      'baseline exact uses token-ID equality and is not strictly the same metric. '
                      'Compiler calls are limited to failed baseline snippets; costs exclude baseline compilation.'}
    out.with_suffix('.report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('\n=== Compile-guided codec repair ===')
    print(json.dumps(result, indent=2), flush=True)
    print(f'Report: {out.with_suffix(".report.json").resolve()}', flush=True)


if __name__ == '__main__':
    main()
