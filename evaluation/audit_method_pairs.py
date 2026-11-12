"""Audit a method-pair dataset with the actual tokenizer used by the codec."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from training.data import load_base_tokenizer, encode_text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--codec-config', required=True)
    p.add_argument('--prefix-length', type=int, default=64)
    p.add_argument('--target-length', type=int, default=256)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(args.dataset)
    cfg = OmegaConf.load(args.codec_config)
    tokenizer = load_base_tokenizer(cfg.tokenizer)
    rows = {}
    result = {'splits': {}, 'prefix_length': args.prefix_length,
              'target_length': args.target_length, 'tokenizer': str(cfg.tokenizer.model_id)}
    prompt_targets = defaultdict(set)
    for split in ('train', 'validation'):
        path = root / (split + '.jsonl')
        rows[split] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        prompt_over = target_over = changed_token_counts = 0
        sources = Counter()
        for row in rows[split]:
            prompt_ids = encode_text(tokenizer, row['prompt'], append_eos=False)
            target_ids = encode_text(tokenizer, row['target'], append_eos=True)
            prompt_over += len(prompt_ids) > args.prefix_length
            target_over += len(target_ids) > args.target_length
            changed_token_counts += (row.get('prompt_tokens') != len(prompt_ids) or
                                     row.get('target_tokens_without_eos') != len(tokenizer.encode(row['target'])))
            sources[row['source'].split('/')[0]] += 1
            if split == 'train':
                prompt_targets[row['prompt']].add(row['target'])
        result['splits'][split] = {'examples': len(rows[split]), 'prompt_truncations': prompt_over,
                                   'target_truncations': target_over,
                                   'builder_token_count_mismatches': changed_token_counts,
                                   'sources': dict(sources.most_common())}
    ambiguous = {p: targets for p, targets in prompt_targets.items() if len(targets) > 1}
    result['ambiguous_training_prompts'] = len(ambiguous)
    result['ambiguous_training_rows'] = sum(len(x) for x in ambiguous.values())
    result['ambiguous_examples'] = [{'prompt': p, 'different_targets': len(t)}
                                    for p, t in sorted(ambiguous.items(), key=lambda x: -len(x[1]))[:5]]
    train_sources = {row['source'] for row in rows['train']}
    val_sources = {row['source'] for row in rows['validation']}
    result['source_paths_in_both_splits'] = len(train_sources & val_sources)
    result['note'] = ('Exact prompt ambiguity is not a measure of semantic ambiguity. '
                      'Cross-file similar or duplicate code may still exist; methods may depend on external context.')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    print('Report:', out.resolve(), flush=True)


if __name__ == '__main__':
    main()
