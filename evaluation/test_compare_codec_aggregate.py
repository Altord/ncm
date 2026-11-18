"""Aggregate comparisons reject mismatched inputs and tokenizer configurations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from evaluation.compare_codec_aggregate import compare

report = {'input_sha256': 'a', 'text_key': 'text', 'sequence_length': 256, 'code_only': True,
          'code_extensions': ['.cs'], 'examples': 2, 'valid_tokens': 20,
          'valid_token_accuracy': 0.9, 'exact_sequence_matches': 1, 'mean_character_similarity': 0.9}
cfg = OmegaConf.create({'tokenizer': {'name': 'example'}})
assert compare(report, dict(report), cfg, cfg)['reference']['exact_sequences'] == 1
bad = dict(report, input_sha256='b')
try:
    compare(report, bad, cfg, cfg)
    raise AssertionError('Mismatched input accepted')
except ValueError:
    pass
try:
    compare(report, report, cfg, OmegaConf.create({'tokenizer': {'name': 'different'}}))
    raise AssertionError('Mismatched tokenizer accepted')
except ValueError:
    pass
print('Aggregate codec report checks passed.')
