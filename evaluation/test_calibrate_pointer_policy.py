"""Check calibration uses training labels and declines harmful policies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from evaluation.calibrate_pointer_policy import apply_policy, choose_policy

batch = {'target': torch.tensor([[1, 1, 1, 1, 1]]),
         'valid': torch.tensor([[True, True, True, True, True]]),
         'base': torch.tensor([[0, 0, 0, 1, 1]]),
         'copy': torch.tensor([[1, 1, 1, 0, 0]]),
         'margin': torch.tensor([[.2, .2, .2, 5., 5.]]),
         'mass': torch.tensor([[.5, .5, .5, .5, .5]])}
policy = choose_policy([batch])
assert policy is not None and policy['calibration_fixed'] == 3
assert policy['calibration_regressed'] == 0
selected = apply_policy(batch['base'], batch['copy'], batch['margin'], batch['mass'], policy)
assert torch.equal(selected, batch['target'])
harmful = dict(batch, target=batch['base'])
assert choose_policy([harmful]) is None
assert torch.equal(apply_policy(batch['base'], batch['copy'], batch['margin'], batch['mass'], None), batch['base'])
# No target-derived valid-length mask is passed into inference-time selection.
one = torch.tensor([[1]])
assert apply_policy(one, torch.tensor([[2]]), torch.tensor([[0.1]]), torch.tensor([[0.5]]),
                    {'max_margin': 1, 'min_attention_mass': .1}).item() == 2
print('Pointer policy calibration checks passed.')
