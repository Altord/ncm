"""Check adapter report provenance and compiler-result reuse."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.data import sha256_file
import evaluation.compile_codec_ensemble as compiler
import evaluation.compile_codec_latent_refiner as screen

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory); source = root/'source.jsonl'; checkpoint = root/'codec.pt'
    checkpoint.write_bytes(b'checkpoint')
    source.write_text(json.dumps({'id':'one','target':'original'})+'\n')
    (root/'base.jsonl').write_text(json.dumps({'id':'one','target':'original','reconstructed':'old',
        'original_compile':{'status':'passed'},'reconstruction_compile':{'status':'failed'}})+'\n')
    (root/'base.report.json').write_text(json.dumps({'input_sha256':sha256_file(source),'checkpoint':str(checkpoint),'compile_requested':True}))
    (root/'adapter.jsonl').write_text(json.dumps({'id':'one','selected':'new'})+'\n')
    (root/'adapter.report.json').write_text(json.dumps({'input_sha256':sha256_file(source),'codec_sha256':sha256_file(checkpoint)}))
    calls = []
    compiler.compile_candidate = lambda text, framework: calls.append(text) or {'status':'passed'}
    sys.argv = ['test','--input',str(source),'--adapter-output',str(root/'adapter.jsonl'),
                '--baseline',str(root/'base.jsonl'),'--output',str(root/'out.jsonl')]
    with contextlib.redirect_stdout(io.StringIO()): screen.main()
    result = json.loads((root/'out.report.json').read_text())
    assert result['baseline_compiles'] == 0 and result['adapter_compiles'] == 1
    assert result['compile_gains'] == 1 and result['new_compiler_calls'] == 1 and calls == ['new']
print('Latent adapter compile integration checks passed.')
