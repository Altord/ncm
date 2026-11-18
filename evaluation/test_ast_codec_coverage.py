"""Parser coverage categorizes malformed and clean methods."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.diagnose_ast_codec_coverage import classify

prompt = 'Complete this C# method: static int M(int value, int count)'
clean = classify('static int M(int value, int count) => value + count;', prompt)
assert clean['method_found'] and not clean['parser_has_error'] and clean['candidate_count'] == 0
broken = classify('static int M(int value, int count) => value + nope nope;', prompt)
assert broken['parser_has_error'] and broken['direct_error_identifiers'] and broken['candidate_count']
print('AST failure coverage checks passed.')
