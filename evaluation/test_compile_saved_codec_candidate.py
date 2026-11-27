"""Exercise compile caching and paired report checks on synthetic saved rows."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.data import sha256_file
import evaluation.compile_codec_ensemble as compiler
import evaluation.compile_saved_codec_candidate as screen

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory); source = root/'input.jsonl'; source.write_text(json.dumps({'id':'one','target':'target'})+'\n')
    (root/'base.jsonl').write_text(json.dumps({'id':'one','target':'target','reconstructed':'old',
      'original_compile':{'status':'passed'},'reconstruction_compile':{'status':'failed'}})+'\n')
    (root/'base.report.json').write_text(json.dumps({'input_sha256':sha256_file(source),'compile_requested':True}))
    (root/'new.jsonl').write_text(json.dumps({'id':'one','target':'target','reconstructed':'new'})+'\n')
    (root/'new.report.json').write_text(json.dumps({'input_sha256':sha256_file(source)}))
    calls = []
    compiler.compile_candidate = lambda text, framework: calls.append(text) or {'status':'passed'}
    sys.argv = ['test','--input',str(source),'--candidate',str(root/'new.jsonl'),
                '--baseline',str(root/'base.jsonl'),'--output',str(root/'out.jsonl')]
    with contextlib.redirect_stdout(io.StringIO()): screen.main()
    report = json.loads((root/'out.report.json').read_text())
    assert report['candidate_compiles'] == report['new_compiler_calls'] == 1 and calls == ['new']
print('Saved codec candidate compilation checks passed.')
