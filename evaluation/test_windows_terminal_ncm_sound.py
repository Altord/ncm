"""Check tone format and that Windows Terminal preferences are preserved."""
import json
import sys
import tempfile
import wave
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.install_windows_terminal_ncm_sound import update_settings, write_tone

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    sound, settings = root / 'tone.wav', root / 'settings.json'
    write_tone(sound)
    with wave.open(str(sound), 'rb') as audio:
        assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
        assert audio.getframerate() == 22050 and audio.getnframes() > 1000
    original = {'profiles': {'defaults': {'fontSize': 12},
                             'list': [{'name': 'Other', 'bellStyle': 'none'}]},
                'copyOnSelect': False}
    contents = json.dumps(original).encode()
    settings.write_bytes(contents)
    backup, overrides = update_settings(settings, sound)
    assert backup.read_bytes() == contents and overrides == ['Other']
    updated = json.loads(settings.read_text(encoding='utf-8'))
    assert updated['copyOnSelect'] is False
    assert updated['profiles']['defaults']['fontSize'] == 12
    assert updated['profiles']['defaults']['bellSound'] == str(sound)
    assert updated['profiles']['defaults']['bellStyle'] == 'audible'
    assert update_settings(settings, sound)[0] is None
print('Windows Terminal sound configuration checks passed.')
