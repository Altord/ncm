"""Exercise dataset extraction, separation and provenance using fake repos."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from training.data import sha256_file
import evaluation.build_fresh_codec_methods as screen


class Tokenizer:
    is_fast = True
    eos_token_id = 1
    @property
    def hf_tokenizer(self): return self
    def encode(self, text, add_special_tokens=False): return [2, 3]


with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    sources = root/'projects'; sources.mkdir()
    for project in ('snake', 'flock'):
        repo = sources/project; repo.mkdir(); (repo/'.git').mkdir()
        methods = '\n'.join(f'static int {project}M{i}(int x) => x + {i};' for i in range(7))
        (repo/'Game.cs').write_text('class Game {\n'+methods+'\n}', encoding='utf-8')
    def write(name, record):
        path = root/name; path.write_text(json.dumps(record)+'\n', encoding='utf-8'); return path
    train = write('train.jsonl', {'text':'class Unrelated {}'})
    val = write('val.jsonl', {'text':'class Different {}'})
    mt = write('method-train.jsonl', {'target':'void Unrelated() {}'})
    mv = write('method-val.jsonl', {'target':'void Different() {}'})
    stress = write('stress.jsonl', {'target':'void Other() {}'})
    extra = write('extra.jsonl', {'target':'static int snakeM0(int x) => x + 0;'})
    cache = root/'cache'; cache.mkdir()
    vc = root/'valcache'; vc.mkdir()
    (cache/'manifest.json').write_text(json.dumps({'source_sha256':sha256_file(train)}))
    (vc/'manifest.json').write_text(json.dumps({'source':str(val), 'source_sha256':sha256_file(val)}))
    cfg = OmegaConf.create({'tokenizer':{}, 'model':{'context_length':256},
        'data':{'input':str(train),'cache_dir':str(cache)}, 'train':{'validation_cache_dir':str(vc)}})
    OmegaConf.save(cfg, root/'a.yaml'); OmegaConf.save(cfg, root/'b.yaml')
    screen.load_base_tokenizer = lambda _: Tokenizer()
    screen.subprocess.check_output = lambda *a, **kw: 'a'*40
    sys.argv = ['test', '--root',str(sources),'--reference-config',str(root/'a.yaml'),
        '--comparison-config',str(root/'b.yaml'),'--method-train',str(mt),'--method-validation',str(mv),
        '--stress',str(stress),'--extra-protected',str(extra),'--output',str(root/'methods.jsonl')]
    with contextlib.redirect_stdout(io.StringIO()): screen.main()
    report = json.loads((root/'methods.manifest.json').read_text())
    assert report['examples'] == 10 and report['dataset_sha256'] == sha256_file(root/'methods.jsonl')
    assert {r['source'].split('/')[0] for r in screen.records(root/'methods.jsonl')} == {'snake','flock'}
    assert all('snakeM0' not in r['target'] for r in screen.records(root/'methods.jsonl'))
    assert str(extra) in report['source_sha256']
print('External method dataset integration checks passed.')
