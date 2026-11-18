"""Exercise mixture source separation and bounded construction on CPU."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
import training.build_codec_method_mixture as mix
from training.data import sha256_file

class Tokenizer:
    eos_token_id = 1
    def encode(self, text): return [2] * len(text)

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    def write(name, rows):
        p = root/name
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        return p
    base = write('base.jsonl', [{'source': f'base{i}.cs', 'text': 'abc'} for i in range(3)])
    val = write('val.jsonl', [{'source': 'val.cs', 'text': 'val'}])
    methods = write('methods.jsonl', [{'source':'held.cs', 'target':'hello'}, {'source':'val.cs', 'target':'world'},
                                      {'source':'train.cs', 'target':'fine'}])
    held = write('held.jsonl', [{'source':'held.cs', 'target':'hold'}])
    for name, path in [('cache', base), ('valcache', val)]:
        (root/name).mkdir()
        (root/name/'manifest.json').write_text(json.dumps({'source':str(path), 'source_sha256':sha256_file(path)}))
    cfg = OmegaConf.create({'data': {'input':str(base), 'cache_dir':str(root/'cache')},
        'train': {'validation_cache_dir':str(root/'valcache')}, 'tokenizer':{}, 'model':{'context_length':32}})
    OmegaConf.save(cfg, root/'config.yaml')
    mix.load_base_tokenizer = lambda cfg: Tokenizer()
    sys.argv = ['test', '--reference-config', str(root/'config.yaml'), '--methods', str(methods),
                '--heldout-methods', str(held), '--output', str(root/'mixture.jsonl')]
    mix.main()
    report = json.loads((root/'mixture.manifest.json').read_text())
    assert report['method_examples'] == 1 and report['exclusions']['source_overlap'] == 2
    assert report['mixture_sha256'] == sha256_file(root/'mixture.jsonl')
print('Codec method-mixture checks passed.')
