"""Verify stress selection uses only stored confidence and a fixed threshold."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.codec_latent_refiner_gate_stress import apply_gate

rows = [
 {'id':'a', 'baseline':'old', 'adapter':'new', 'changed':True, 'confidence_delta':0.2, 'baseline_exact':False, 'adapter_exact':True},
 {'id':'b', 'baseline':'right', 'adapter':'wrong', 'changed':True, 'confidence_delta':-0.1, 'baseline_exact':True, 'adapter_exact':False},
]
out = apply_gate(rows, 0.0)
assert out[0]['selected'] == 'new' and out[0]['selected_exact']
assert out[1]['selected'] == 'right' and out[1]['selected_exact']
assert not any(r['use_adapter'] for r in apply_gate(rows, None))
# Selection is invariant to scoring labels.
for row in rows: row['baseline_exact'], row['adapter_exact'] = not row['baseline_exact'], not row['adapter_exact']
assert [r['selected'] for r in apply_gate(rows, 0.0)] == ['new', 'right']
print('Fixed latent gate stress checks passed.')
