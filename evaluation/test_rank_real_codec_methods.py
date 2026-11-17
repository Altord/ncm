"""Check unchanged, improved, and regressed cases are all counted."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation.rank_real_codec_methods as screen

class Model:
    def to(self, device): return self
    def eval(self): return self

class Loader:
    @staticmethod
    def from_pretrained(*args, **kwargs): return Model()

screen.AutoTokenizer = Loader
screen.AutoModelForCausalLM = Loader
screen.make_signature_prompt = lambda text: 'signature'
screen.generate_candidates = lambda prompt, before, limit, **kwargs: (['replacement'], True)
screen.ast_candidates = lambda prompt, before, limit: ['replacement']
screen.candidate_score = lambda model, tokenizer, prompt, candidate, device: 1.0 if candidate == 'replacement' else 0.0
with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    rows = [{'id': '1', 'target': 'replacement', 'reconstructed': 'old'},
            {'id': '2', 'target': 'old', 'reconstructed': 'old'}]
    (root/'input.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    sys.argv = ['test', '--input', str(root/'input.jsonl'), '--output', str(root/'out.jsonl'), '--device', 'cpu']
    screen.main()
    result = json.loads((root/'out.report.json').read_text())
    assert result['examples'] == 2 and result['new_exact'] == 1 and result['lost_exact'] == 1
    assert result['baseline_exact'] == result['selected_exact'] == 1
print('Real codec method ranking checks passed.')
