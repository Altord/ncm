"""Check prompt-only signature copying, including malformed inputs and Unicode."""
import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.codec_prompt_signature_copy import copy_signature
import evaluation.codec_prompt_signature_copy as experiment

prompt = 'Complete this C# method: static int Midpoint(int a, int b)'
decoded = 'static int MidAttribute(int a, int b) => (a + b) / 2;'
result, why = copy_signature(prompt, decoded)
assert why == 'signature_copied'
assert result == 'static int Midpoint(int a, int b) => (a + b) / 2;'
assert copy_signature(prompt, result)[1] == 'unchanged'
assert copy_signature('Write a method', decoded) == (decoded, 'prompt_not_signature')
assert copy_signature('Complete this C# method: broken !@', decoded)[1] == 'invalid_prompt_signature'
assert copy_signature(prompt, 'not a valid method')[1] == 'decoded_method_unavailable'
prompt = 'Complete this C# method: static int café(int value)'
decoded = 'static int caff(int value) { return value + 1; }'
assert copy_signature(prompt, decoded)[0] == 'static int café(int value) { return value + 1; }'
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    row = {'id': 'x', 'prompt': prompt, 'reconstructed': decoded,
           'target': 'static int café(int value) { return value + 1; }'}
    (root / 'in.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
    sys.argv = ['test', '--input', str(root / 'in.jsonl'), '--output', str(root / 'out.jsonl')]
    experiment.main()
    report = json.loads((root / 'out.report.json').read_text())
    assert report['before_exact_text'] == 0 and report['after_exact_text'] == 1
    assert report['before_exact_names'] == 0 and report['after_exact_names'] == 1
print('Prompt signature copy checks passed.')
