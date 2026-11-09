# Godot/C# generator pilot data

This is the next-stage **pipeline smoke set** for the NCM generator. The examples are original, short code tasks created for this repository; they do not copy the synced documentation or demo-project code. They are covered by the repository's MIT license.

## Important scope

The current NCM starter has a 16-token prompt prefix and a 64-token target. `train.jsonl`, `validation.jsonl`, and `test.jsonl` therefore contain tiny C# method-generation tasks only. They exercise prompt conditioning, target encoding, frozen-tokenizer code caches, and generator loss. They are far too small and short to teach complete games or establish coding competence.

`test.jsonl` is held out. Do not feed it into training, tune against it, or merge it into the validation set. The separate `project_eval.jsonl` contains future full-game evaluation briefs with acceptance criteria, but no gold project implementations. Those tasks are for the later long-context, tool-using game-generation system, not this 64-token model.

The JSONL rows include metadata (`id`, `family`, `acceptance`, and optional `cases`) for traceability and later evaluation. `training.prepare_data --stage generator` consumes only `prompt` and `target`; the current generator trainer does not execute acceptance checks.

## Prepare after the Godot/C# tokenizer run finishes

The tokenizer artifact must exist at `artifacts/godot-csharp/tokenizer.pt`. Keep the tokenizer and generator configs' VQ-VAE architecture and codebook levels identical.

First inspect tokenizer-only reconstruction on the untouched test targets:

```bash
.venv/bin/python evaluation/godot_csharp_codec_eval.py
```

This reports valid-token accuracy, exact round-trip count, descriptive character similarity, and reconstructed text under `runs/godot-csharp/generator-pilot/test/`. It uses the frozen codec only. Poor reconstruction points to a tokenizer/representation limitation; good reconstruction with poor generated candidates points to the generator/data/conditioning stage. No metric here alone establishes code correctness.

To broaden the codec check to the existing file-held-out C# validation corpus:

```bash
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --input data/mycode/godot-csharp-reference/validation.jsonl \
  --text-key text \
  --code-only \
  --output runs/godot-csharp/codec-validation/csharp.jsonl
```

This filters to `.cs` and `.csproj` sources by default and reports accuracy by extension. It does not update weights or read the training split.

To compile the six original pilot targets and their tokenizer reconstructions as a syntax-preservation check, run the small test split with `--compile`:

```bash
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --compile \
  --output runs/godot-csharp/generator-pilot/test/codec-compiled.jsonl
```

The report counts original-target and reconstruction compile passes separately. Compilation checks syntax/type compatibility with the evaluator's minimal `Vector2` stub; it does not execute the code or test the Godot engine.

## Expanded tokenizer compile suite

`codec_compile_stress.jsonl` contains 48 original, self-contained C# methods covering arithmetic, guards, timers, arrays, strings, and the evaluator's `Vector2` stub. It is separate from generator train/validation/test data. The original methods were compiled together against the same .NET 8 and `Vector2` stub contract; they all compile. Use this larger set to measure whether codec reconstructions compile, not whether generated implementations are semantically correct.

Run it with each tokenizer checkpoint:

```bash
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config /tmp/godot-csharp-strict-64level-40epoch.yaml \
  --checkpoint artifacts/godot-csharp-strict/64level-40epoch/tokenizer.pt \
  --input examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl \
  --text-key target \
  --compile \
  --output runs/godot-csharp-strict/64level-tokenizer-40epoch/compile-stress/results.jsonl
```

Check `original_compile_passes` first; a valid comparison requires all originals to pass. Then compare reconstruction compile passes, exact chunks, and token accuracy across checkpoints. The suite deliberately remains held out and must not be added to tokenizer or generator training.

## Compare code tokenizer candidates

Before changing the base tokenizer, compare token counts, exact text round-trips, context overflow, and how many subword pieces identifier-like strings use:

```bash
.venv/bin/python evaluation/compare_code_tokenizers.py \
  --input examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl \
  --text-key target \
  --output runs/tokenizer-comparison/codec-stress.json
```

The default comparison is GPT-2, CodeBERT, and CodeT5. These are tokenizer diagnostics only: a candidate must be trained with a matching VQ-VAE and then evaluated on the same held-out exact-reconstruction and compile suite before we select it.

## Controlled next tokenizer experiment: code-only data mix

The current mixed corpus has 75% code chunks and 25% documentation chunks. To test whether the prose share is hurting code reconstruction, build a corpus with documentation excluded and use the separate config below. “Code-only” here includes C# source, project files, Godot scenes/resources, and shader files from the existing source manifest; it does not mean C# files alone. The VQ-VAE, seed, optimizer, context length, and quantization levels stay matched to the baseline; only the data mix and run/output paths change.

```bash
.venv/bin/python -m training.build_reference_corpus \
  --root data/reference \
  --output data/mycode/godot-csharp-code-only \
  --sequence-length 64 \
  --max-doc-fraction 0

wc -l data/mycode/godot-csharp-code-only/{train,validation}.jsonl
cat data/mycode/godot-csharp-code-only/manifest.json

.venv/bin/python -m training.prepare_data \
  --config config/train/godot-csharp-code-only-tokenizer.yaml \
  --stage tokenizer

.venv/bin/python -m training.prepare_data \
  --config config/train/godot-csharp-code-only-tokenizer.yaml \
  --stage tokenizer \
  --input data/mycode/godot-csharp-code-only/validation.jsonl \
  --cache-dir data/cache/godot-csharp-code-only/tokenizer-validation
```

Run from scratch; do not resume the 100-step pilot because its learning-rate schedule was sized for 100 steps. Use the same 40-epoch rule, based on this corpus's own row count:

```bash
N=$(wc -l < data/mycode/godot-csharp-code-only/train.jsonl)
EPOCH_STEPS=$(((N + 15) / 16))
MAX_STEPS=$((EPOCH_STEPS * 40))
if (( MAX_STEPS > 330000 )); then MAX_STEPS=330000; fi
echo "rows=$N steps_per_epoch=$EPOCH_STEPS max_steps=$MAX_STEPS"

.venv/bin/python -m training.train_tokenizer \
  --config config/train/godot-csharp-code-only-tokenizer.yaml \
  --max-steps "$MAX_STEPS"
```

After it finishes, compare the code-only validation round trip with the mixed baseline, then compile the six held-out snippets using the code-only artifact:

```bash
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config config/train/godot-csharp-generator-pilot.yaml \
  --checkpoint artifacts/godot-csharp-code-only/tokenizer.pt \
  --input data/mycode/godot-csharp-code-only/validation.jsonl \
  --text-key text --code-only \
  --output runs/godot-csharp-code-only/codec-validation/csharp.jsonl

.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config config/train/godot-csharp-generator-pilot.yaml \
  --checkpoint artifacts/godot-csharp-code-only/tokenizer.pt \
  --compile \
  --output runs/godot-csharp-code-only/pilot-test/codec-compiled.jsonl
```

For a direct validation comparison, run the same first command again with `--checkpoint artifacts/godot-csharp/tokenizer.pt` and a different output path. Both checkpoints use the same VQ-VAE architecture, so the generator-pilot config supplies the matching architecture while `--checkpoint` selects the weights being measured.

This isolates documentation/data-mix effects. If the held-out code metrics barely move, the next ablation should change tokenizer capacity or objective—not simply add more generator steps.

## Audit result and NCM-aligned capacity experiment

The code-only run completed 61,720 updates. Its held-out result (81.83% token accuracy, 2.61% exact 64-token chunks, 71.86% character similarity) is essentially the same as the mixed-data result. The six-snippet compile check remains 0/6 reconstructed snippets. Removing prose documentation was therefore not enough to fix syntax loss.

The principal architecture gap is scale. This project's pilot uses latent width 64, four levels `[1, 4, 16, 64]`, codebooks of 256 entries, two decoder attention heads, no pre-quant normalization, and no learned first-level sampling. The published NCM WMT14 tokenizer uses context length 64 but width 384, eleven levels, 4,096-entry codebooks, six decoder heads, pre-quant GroupNorm and dropout, learned first-level sampling, and 0.5 fine-level dropout. The new `config/train/godot-csharp-ncm-aligned-tokenizer.yaml` adopts those context-64 architecture/quantizer settings, while retaining this project's GPT-2 tokenizer, C# data, and batch size 16. This is a capacity-and-recipe experiment, not an exact reproduction of WMT14. Sources: [NCM paper](https://arxiv.org/html/2610.14162), [upstream WMT14 config](https://github.com/HazyResearch/ncm/blob/main/config/eval/wmt14.yaml).

The previous code-only run's best-by-loss checkpoint gave 80.81% token accuracy and 63 exact chunks, versus 81.83% and 59 chunks for its final artifact. This shows why we should record and choose by code-relevant reconstruction metrics instead of treating cross-entropy as the only quality signal. The trainer now logs both token accuracy and exact-chunk rate. The NCM-aligned config selects the checkpoint with the best exact-chunk rate, breaking ties by token accuracy; `final.pt` remains the resumable last-step checkpoint.

```bash
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config config/train/godot-csharp-generator-pilot.yaml \
  --checkpoint runs/godot-csharp/tokenizer/checkpoints/best.pt \
  --input data/mycode/godot-csharp-code-only/validation.jsonl \
  --text-key text --code-only \
  --output runs/godot-csharp/mixed-best/codec-validation/csharp.jsonl

.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config config/train/godot-csharp-generator-pilot.yaml \
  --checkpoint runs/godot-csharp-code-only/tokenizer/checkpoints/best.pt \
  --input data/mycode/godot-csharp-code-only/validation.jsonl \
  --text-key text --code-only \
  --output runs/godot-csharp-code-only/best/codec-validation/csharp.jsonl

.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config config/train/godot-csharp-generator-pilot.yaml \
  --checkpoint runs/godot-csharp-code-only/tokenizer/checkpoints/best.pt \
  --compile \
  --output runs/godot-csharp-code-only/best/pilot-test/codec-compiled.jsonl
```

If `best.pt` is missing for the mixed run, use its `latest.pt`; the saved run may predate validation-checkpoint metric logging. For the code-only run, `best.pt` was selected by validation loss before this metric change. It is useful for comparison, but future runs will use the configured code metrics.

First smoke-test the larger configuration for 100 steps to check that it fits and produces finite losses. Keep the smoke outputs separate from the full run:

```bash
cp config/train/godot-csharp-ncm-aligned-tokenizer.yaml /tmp/godot-csharp-ncm-aligned-smoke.yaml
sed -i 's#runs/godot-csharp-ncm-aligned/tokenizer#runs/godot-csharp-ncm-aligned/tokenizer-smoke#; s#artifacts/godot-csharp-ncm-aligned#artifacts/godot-csharp-ncm-aligned-smoke#' \
  /tmp/godot-csharp-ncm-aligned-smoke.yaml

.venv/bin/python -m training.train_tokenizer \
  --config /tmp/godot-csharp-ncm-aligned-smoke.yaml \
  --max-steps 100

tail -n 5 runs/godot-csharp-ncm-aligned/tokenizer-smoke/logs/train.jsonl
```

If the smoke run completes without an out-of-memory error and logs finite loss/VRAM values, the NCM-aligned model fits this GPU at the tested batch size. Start a fresh full run using 40 passes over the code-only training split:

```bash
N=$(wc -l < data/mycode/godot-csharp-code-only/train.jsonl)
EPOCH_STEPS=$(((N + 15) / 16))
MAX_STEPS=$((EPOCH_STEPS * 40))
if (( MAX_STEPS > 330000 )); then MAX_STEPS=330000; fi
echo "rows=$N steps_per_epoch=$EPOCH_STEPS max_steps=$MAX_STEPS"

.venv/bin/python -m training.train_tokenizer \
  --config config/train/godot-csharp-ncm-aligned-tokenizer.yaml \
  --max-steps "$MAX_STEPS"
```

Evaluate the aligned run on the same code-only validation split and six compile cases, using its artifact:

```bash
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config config/train/godot-csharp-generator-pilot.yaml \
  --checkpoint artifacts/godot-csharp-ncm-aligned/tokenizer.pt \
  --input data/mycode/godot-csharp-code-only/validation.jsonl \
  --text-key text --code-only \
  --output runs/godot-csharp-ncm-aligned/codec-validation/csharp.jsonl

.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config config/train/godot-csharp-ncm-aligned-tokenizer.yaml \
  --checkpoint artifacts/godot-csharp-ncm-aligned/tokenizer.pt \
  --compile \
  --output runs/godot-csharp-ncm-aligned/pilot-test/codec-compiled.jsonl
```

The aligned run tests whether the much smaller pilot codec was the source of the gap. It also writes `val_token_accuracy` and `val_exact_sequence_match_rate` to validation logs and exports the best checkpoint under the configured exact-chunk criterion. Generator training stays paused until reconstruction improves substantially and the compile check passes.

## Diagnose codec compile failures

After running the aligned codec with `--compile`, inspect the exact changed identifiers, operators, punctuation, and compiler messages:

```bash
.venv/bin/python evaluation/inspect_codec_failures.py \
  --input runs/godot-csharp-ncm-aligned/pilot-test/codec-compiled.jsonl
```

This compares each original pilot target with its reconstruction and prints token-level differences for failed rows. Use those failures to decide the next tokenizer change; do not infer the cause from token accuracy alone.

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

After the 20-step pipeline pilot, generate outputs for the held-out records with:

```bash
.venv/bin/python evaluation/godot_csharp_generator_pilot.py
```

The script writes candidate JSONL and a report under `runs/godot-csharp/generator-pilot/test/`. It reports normalized exact-reference matches as a strict diagnostic, and carries the acceptance cases forward for review. If the .NET SDK is installed, add `--compile` to compile each candidate in a temporary project. That checks syntax/type compatibility with a minimal helper stub; it does **not** execute generated code or prove game-engine compatibility. Exact text match is not a semantic correctness metric. Do not train on `test.jsonl` or tune using it.

The next evaluator upgrade is an isolated C# execution harness that measures deterministic case pass rate without running unreviewed output with unrestricted access to the host. The full-game briefs in `project_eval.jsonl` remain for the later long-context, tool-using game-generation system. Their acceptance checks should be automated where possible (project opens, builds, input works, state transitions pass) and supplemented with a human play review for clarity, feel, accessibility, and fun.

The first full-game briefs in `project_eval.jsonl` are intended for later end-to-end tests. Their acceptance checks should be automated where possible (project opens, builds, input works, state transitions pass) and supplemented with a human play review for clarity, feel, accessibility, and fun.
