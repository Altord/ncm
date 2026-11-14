"""Install a quiet, distinct bell tone for Windows Terminal profiles.

Windows Terminal plays this sound for terminal BEL characters, including those
sent by NCM scripts over SSH. It does not change Windows system notifications.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import time
import wave


def write_tone(path):
    rate = 22050
    samples = []
    for frequency, duration in ((660, .12), (0, .035), (880, .15)):
        length = round(rate * duration)
        fade = max(1, round(rate * .015))
        for i in range(length):
            envelope = min(1.0, i / fade, (length - 1 - i) / fade)
            sample = 0 if frequency == 0 else int(32767 * .075 * envelope * math.sin(2 * math.pi * frequency * i / rate))
            samples.append(sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(struct.pack('<' + 'h' * len(samples), *samples))


def update_settings(settings, sound):
    raw = settings.read_bytes()
    data = json.loads(raw.decode('utf-8-sig'))
    profiles = data.setdefault('profiles', {})
    if not isinstance(profiles, dict):
        raise ValueError('Unexpected Windows Terminal profiles format')
    defaults = profiles.setdefault('defaults', {})
    if not isinstance(defaults, dict):
        raise ValueError('Unexpected Windows Terminal profiles format')
    overrides = [p.get('name', '<unnamed>') for p in profiles.get('list', [])
                 if isinstance(p, dict) and ('bellSound' in p or 'bellStyle' in p)]
    desired_sound = str(sound)
    if defaults.get('bellSound') == desired_sound and defaults.get('bellStyle') == 'audible':
        return None, overrides
    defaults['bellSound'] = desired_sound
    defaults['bellStyle'] = 'audible'
    backup = settings.with_name(settings.name + f'.ncm-backup-{time.time_ns()}')
    shutil.copy2(settings, backup)
    if settings.read_bytes() != raw:
        raise RuntimeError(f'Settings changed concurrently; original backup is {backup}')
    formatted = json.dumps(data, ensure_ascii=False, indent=4) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                     dir=settings.parent, prefix='ncm-terminal-',
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(formatted)
    try:
        if settings.read_bytes() != raw:
            raise RuntimeError(f'Settings changed concurrently; original backup is {backup}')
        os.replace(temporary, settings)
    finally:
        temporary.unlink(missing_ok=True)
    return backup, overrides


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preview', action='store_true', help='Play the generated tone once')
    args = p.parse_args()
    if sys.platform != 'win32':
        raise SystemExit('Run this installer on the Windows computer, not over the Ubuntu shell.')
    local = Path(os.environ['LOCALAPPDATA'])
    settings = local / 'Packages/Microsoft.WindowsTerminal_8wekyb3d8bbwe/LocalState/settings.json'
    if not settings.is_file():
        raise SystemExit(f'Windows Terminal stable settings not found: {settings}')
    sound = local / 'NCM/completion.wav'
    write_tone(sound)
    backup, overrides = update_settings(settings, sound)
    print(f'Custom terminal tone: {sound}')
    print(f'Windows Terminal settings: {settings}')
    print(f'Previous settings backup: {backup}' if backup else 'Settings already configured.')
    if overrides:
        print('Profiles with their own bell overrides may not use the default:', ', '.join(overrides))
    print('This changes terminal bell sounds for profiles without overrides, not Windows system notifications.')
    if args.preview:
        import winsound
        winsound.PlaySound(str(sound), winsound.SND_FILENAME)


if __name__ == '__main__':
    main()
