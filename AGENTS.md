# NCM experiment scripts

When adding a user-run experiment script, emit one terminal bell when the script exits, including on failure. Preserve the original exit status and keep failure messages clear; the bell means the process ended, not that it succeeded. The user's Windows Terminal over SSH uses a distinct NCM tone configured by `scripts/install_windows_terminal_ncm_sound.py`; do not switch it back to the default Windows notification. Terminal settings control whether the bell is audible.
