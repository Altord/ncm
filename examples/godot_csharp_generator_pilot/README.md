# Godot/C# generator pilot data

This is the next-stage **pipeline smoke set** for the NCM generator. The examples are original, short code tasks created for this repository; they do not copy the synced documentation or demo-project code. They are covered by the repository's MIT license.

## Important scope

The current NCM starter has a 16-token prompt prefix and a 64-token target. `train.jsonl`, `validation.jsonl`, and `test.jsonl` therefore contain tiny C# method-generation tasks only. They exercise prompt conditioning, target encoding, frozen-tokenizer code caches, and generator loss. They are far too small and short to teach complete games or establish coding competence.

`test.jsonl` is held out. Do not feed it into training, tune against it, or merge it into the validation set. The separate `project_eval.jsonl` contains future full-game evaluation briefs with acceptance criteria, but no gold project implementations. Those tasks are for the later long-context, tool-using game-generation system, not this 64-token model.

The JSONL rows include metadata (`id`, `family`, `acceptance`, and optional `cases`) for traceability and later evaluation. `training.prepare_data --stage generator` consumes only `prompt` and `target`; the current generator trainer does not execute acceptance checks.

## Prepare after the Godot/C# tokenizer run finishes

The tokenizer artifact must exist at `artifacts/godot-csharp/tokenizer.pt`. Keep the tokenizer and generator configs' VQ-VAE architecture and codebook levels identical.

```bash
source .venv/bin/activate

python -m training.prepare_data \
  --config config/train/godot-csharp-generator-pilot.yaml \
  --stage generator

python -m training.prepare_data \
  --config config/train/godot-csharp-generator-pilot.yaml \
  --stage generator \
  --input examples/godot_csharp_generator_pilot/validation.jsonl \
  --cache-dir data/cache/godot-csharp/generator-pilot-validation

python -m training.train_generator \
  --config config/train/godot-csharp-generator-pilot.yaml \
  --max-steps 20
```

This 20-step run is only a plumbing check. It should produce finite per-level validation losses and save a generator checkpoint. It will rapidly repeat this tiny training set; do not extend it or call it a useful game-coding model.

## Held-out test handling

The trainer has no test-only evaluation path yet. Keep `test.jsonl` untouched until a separate inference/evaluation harness exists. That harness should generate one candidate per `id`, compile candidates in a clean Godot C# project, execute deterministic cases where practical, and report parse/build/test pass rates. Exact text match is not a suitable correctness metric for code.

The first full-game briefs in `project_eval.jsonl` are intended for later end-to-end tests. Their acceptance checks should be automated where possible (project opens, builds, input works, state transitions pass) and supplemented with a human play review for clarity, feel, accessibility, and fun.
