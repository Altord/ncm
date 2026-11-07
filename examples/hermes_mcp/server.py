"""Expose the released NCM Sudoku checkpoint as a Hermes MCP tool.

Run from the repository root with:

    python -m examples.hermes_mcp.server
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "eval" / "sudoku-easy.yaml"
DEFAULT_CHECKPOINT_DIR = ROOT / "artifacts" / "sudoku" / "easy"

mcp = FastMCP("NCM Sudoku")
_solver: "SudokuSolver | None" = None


def _normalise_puzzle(puzzle: str) -> tuple[list[str], str]:
    """Return the 9 rows and the fixed-width input format used by NCM."""
    allowed = set("0123456789._|B")
    compact = "".join(ch for ch in puzzle.upper() if not ch.isspace())
    invalid = sorted(set(compact) - allowed)
    if invalid:
        raise ValueError(f"Puzzle contains unsupported characters: {''.join(invalid)}")

    cells = [ch for ch in compact if ch in "0123456789._"]
    if len(cells) != 81:
        raise ValueError(f"Expected 81 cells, received {len(cells)}")

    rows = ["".join(cells[i:i + 9]).replace("0", "_").replace(".", "_")
            for i in range(0, 81, 9)]
    prompt = "B" + "".join(row + "|" for row in rows)
    return rows, prompt


def _has_no_conflicts(rows: list[str]) -> bool:
    """Check Sudoku row, column, and 3x3-box constraints, ignoring blanks."""
    grid = [[ch for ch in row] for row in rows]
    for row in grid:
        digits = [ch for ch in row if ch != "_"]
        if len(digits) != len(set(digits)):
            return False

    for col in range(9):
        digits = [grid[row][col] for row in range(9) if grid[row][col] != "_"]
        if len(digits) != len(set(digits)):
            return False

    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            digits = [grid[row][col]
                      for row in range(box_row, box_row + 3)
                      for col in range(box_col, box_col + 3)
                      if grid[row][col] != "_"]
            if len(digits) != len(set(digits)):
                return False
    return True


def _decoded_rows(decoded: str) -> list[str] | None:
    """Parse a model reconstruction into 9 rows, or return None if malformed."""
    cells = [ch for ch in decoded if ch in "0123456789._"]
    if len(cells) != 81:
        return None
    return ["".join(cells[i:i + 9]).replace("0", "_").replace(".", "_")
            for i in range(0, 81, 9)]


class SudokuSolver:
    """Load an NCM checkpoint once and use it for subsequent tool calls."""

    def __init__(self, config_path: Path, checkpoint_dir: Path, device_name: str):
        import torch
        from omegaconf import OmegaConf

        from models.next_concept import NextConceptTransformer
        from tokenization.vqvae_tokenizer import VQVAETokenizer

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("NCM_DEVICE=cuda was requested, but CUDA is unavailable")

        self.device = torch.device(device_name)
        config = OmegaConf.load(config_path)
        config.tokenizer.vqvae.checkpoint_path = str(
            checkpoint_dir / config.checkpoints.tokenizer
        )

        self.tokenizer = VQVAETokenizer.from_config(config.tokenizer)
        self.tokenizer.vqvae = self.tokenizer.vqvae.to(self.device).eval()

        self.model = NextConceptTransformer(
            vqvae_local=self.tokenizer.vqvae,
            config=config.model,
        ).to(self.device)
        state = torch.load(
            checkpoint_dir / config.checkpoints.generator,
            map_location="cpu",
            weights_only=False,
        )
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        self.prefix_len = int(config.model.prefix_len)
        self.temperature = float(config.generation.temperature)
        self.top_k = int(config.generation.top_k)
        self.top_p = float(config.generation.top_p)

    def solve(self, prompt: str) -> str:
        import torch

        token_ids = self.tokenizer.pre_tokenizer.encode(prompt)
        if len(token_ids) != self.prefix_len:
            raise ValueError(
                f"NCM expects {self.prefix_len} prompt tokens; got {len(token_ids)}"
            )

        prefix_tokens = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device
        )
        with torch.inference_mode():
            concept_ids = self.model.generate(
                None,
                prefix_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            _, z_levels = self.tokenizer.vqvae.vq.idx_to_zhats(
                concept_ids, return_zlevels=True
            )
            token_logits = self.tokenizer.vqvae.decoder(z_levels[-1])
            output_ids = token_logits.argmax(dim=-1)[0].tolist()
        return self.tokenizer.pre_tokenizer.decode(output_ids)


def _get_solver() -> SudokuSolver:
    global _solver
    if _solver is None:
        config_path = Path(os.environ.get("NCM_CONFIG", DEFAULT_CONFIG)).expanduser()
        checkpoint_dir = Path(
            os.environ.get("NCM_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)
        ).expanduser()
        device_name = os.environ.get("NCM_DEVICE", "auto").lower()

        if not config_path.is_file():
            raise FileNotFoundError(f"NCM config not found: {config_path}")
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"NCM checkpoint directory not found: {checkpoint_dir}. "
                "Download the Sudoku checkpoint and set NCM_CHECKPOINT_DIR."
            )
        _solver = SudokuSolver(config_path, checkpoint_dir, device_name)
    return _solver


@mcp.tool()
def solve_sudoku(puzzle: str) -> dict[str, Any]:
    """Solve a 9x9 Sudoku with the released NCM easy-Sudoku model.

    Provide 81 cells as nine rows, or as one 81-character string. Use 1-9
    for clues and 0, dot, or underscore for blanks. The result includes a
    candidate grid plus independent validity and clue-preservation checks.
    """
    rows, prompt = _normalise_puzzle(puzzle)
    if not _has_no_conflicts(rows):
        raise ValueError("The givens conflict in a row, column, or 3x3 box")

    decoded = _get_solver().solve(prompt)
    answer_rows = _decoded_rows(decoded)
    if answer_rows is None:
        return {
            "candidate": decoded,
            "valid_solution": False,
            "preserves_clues": False,
            "solved": False,
            "device": str(_get_solver().device),
            "message": "NCM output did not decode to exactly 81 Sudoku cells.",
        }

    preserves_clues = all(
        given == "_" or answer_rows[row][col] == given
        for row, values in enumerate(rows)
        for col, given in enumerate(values)
    )
    complete = all(ch in "123456789" for row in answer_rows for ch in row)
    valid_solution = complete and _has_no_conflicts(answer_rows)
    candidate = "\n".join(answer_rows)
    solver = _get_solver()
    return {
        "candidate": candidate,
        "valid_solution": valid_solution,
        "preserves_clues": preserves_clues,
        "solved": valid_solution and preserves_clues,
        "device": str(solver.device),
        "message": (
            "NCM candidate passed the Sudoku checks."
            if valid_solution and preserves_clues
            else "NCM produced a candidate that failed one or more Sudoku checks."
        ),
    }


if __name__ == "__main__":
    mcp.run()
