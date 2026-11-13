"""Verify source-derived identifier corrections and independent application."""
import sys
import json
import base64
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.codec_identifier_sidecar import encode_sidecar, apply_sidecar
import evaluation.codec_identifier_sidecar as experiment

source = 'static bool IsCooldownReady(float timer) => timer <= 0f;'
decoded = 'static bool IsCooldoldReady(float timer) => timer <= 0f;'
payload, count = encode_sidecar(source, decoded)
assert count > 0 and apply_sidecar(decoded, payload) == source

source = 'static int café(int n) => n + 1;'
decoded = 'static int caff(int n) => n + 2;'
payload, count = encode_sidecar(source, decoded)
restored = apply_sidecar(decoded, payload)
assert count > 0 and 'café' in restored and '+ 2' in restored
assert restored != source  # Numeric errors are not encoded.

payload, count = encode_sidecar(source, source)
assert count == 0 and apply_sidecar(source, payload) == source
try:
    apply_sidecar(source, payload + b'\x00')
except ValueError:
    pass
else:
    raise AssertionError('Trailing payload was accepted')
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    source = 'static int Midpoint(int a, int b) => (a + b) / 2;'
    decoded = 'static int MidAttribute(int a, int b) => (a + b) / 2;'
    row = {'id': 'x', 'target': source, 'reconstructed': decoded,
           'reconstruction_compile': {'status': 'passed'}}
    (root / 'in.jsonl').write_text(json.dumps(row) + '\n')
    experiment.compile_candidate = lambda text, framework: {'status': 'passed'}
    sys.argv = ['test', '--input', str(root / 'in.jsonl'), '--output', str(root / 'out.jsonl'),
                '--compile-changed', '--vq-bytes-per-record', '10']
    experiment.main()
    saved = json.loads((root / 'out.jsonl').read_text())
    report = json.loads((root / 'out.report.json').read_text())
    assert apply_sidecar(decoded, base64.b64decode(saved['sidecar_base64'])) == source
    assert report['after_exact'] == 1 and report['new_compiler_calls'] == 1
    assert report['vq_plus_sidecar_bytes'] == 10 + saved['sidecar_bytes']
print('Identifier sidecar checks passed.')
