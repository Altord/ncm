"""Verify source rows are retained and supplemental text is readable by the codec."""
import io
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.merge_codec_training import merge

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory); original = root/'original.jsonl'; extra = root/'extra.jsonl'
    old = {'text':'old', 'source':'a.cs'}; new = {'target':'new', 'source':'b.cs'}
    original.write_text(json.dumps(old)+'\n'); extra.write_text(json.dumps(new)+'\n')
    output = io.StringIO()
    assert merge(original, extra, 'text', output) == (1, 1)
    rows = [json.loads(s) for s in output.getvalue().splitlines()]
    assert rows[0] == old and rows[1]['text'] == 'new' and rows[1]['source'] == 'b.cs'
    extra.write_text(json.dumps({'target':'bad', 'source':'b.txt'})+'\n')
    try:
        merge(original, extra, 'text', io.StringIO())
        raise AssertionError('Non-C# source accepted')
    except ValueError:
        pass
print('Codec training merge checks passed.')
