# Use NCM Sudoku from Hermes

This example exposes the released NCM easy-Sudoku checkpoint as one MCP tool,
`solve_sudoku`. Hermes can call it like any other MCP server. The server returns
the candidate grid and independently checks whether it is a complete valid
solution that preserves the original clues.

## 1. Install the runtime

Use a Python environment with the NCM dependencies and an MCP 1.x SDK:

```powershell
py -3.12 -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-hermes.txt
```

The repository's `requirements.txt` includes PyTorch. On this Windows machine,
NCM will use CPU unless the installed PyTorch build reports a CUDA-compatible
device. The AMD GPU is not selected by this example's standard CUDA detection.

## 2. Download the released Sudoku files

From the repository root:

```powershell
hf download hazyresearch/ncm-checkpoints --include "sudoku/easy/*" --local-dir artifacts
```

This should create `artifacts/sudoku/easy/` with the config, tokenizer, and
generator checkpoint. The server does not load or execute dataset pickle files.

## 3. Start the server manually

```powershell
$env:NCM_DEVICE = "cpu"
python -m examples.hermes_mcp.server
```

The first tool call loads the model; later calls reuse it. Use `NCM_DEVICE=auto`
to select CUDA when PyTorch reports it available. Override the paths with
`NCM_CONFIG` and `NCM_CHECKPOINT_DIR` if your files are elsewhere.

## 4. Add it to Hermes

Add this entry to `%USERPROFILE%\.hermes\config.yaml`, adjusting the paths if
your checkout or virtual environment is elsewhere:

```yaml
mcp_servers:
  ncm_sudoku:
    command: "F:\\For Fun Projects\\NCM-Dev\\.venv\\Scripts\\python.exe"
    args: ["-m", "examples.hermes_mcp.server"]
    cwd: "F:\\For Fun Projects\\NCM-Dev"
    env:
      NCM_DEVICE: "cpu"
      NCM_CHECKPOINT_DIR: "F:\\For Fun Projects\\NCM-Dev\\artifacts\\sudoku\\easy"
```

Restart Hermes or run `/reload-mcp`. The tool will appear as
`mcp_ncm_sudoku_solve_sudoku`. Ask Hermes to solve a Sudoku and have it report
the returned `solved` and `valid_solution` fields.

## Input format

Pass nine rows of nine characters, with digits `1`-`9` for clues and `0`, `.`,
or `_` for blanks. For example:

```text
530070000
600195000
098000060
800060003
400803001
700020006
060000280
000419005
000080079
```

The NCM output is a model candidate, not a guarantee. The tool validates the
grid and checks that it keeps the clues before reporting success.
