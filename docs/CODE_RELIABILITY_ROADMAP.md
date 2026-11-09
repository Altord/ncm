# NCM coding reliability roadmap

> **Plan superseded for the active DeepSeek codec work:** follow [DEEPSEEK_NCM_CODEC_PLAN.md](DEEPSEEK_NCM_CODEC_PLAN.md). This file preserves the earlier GPT-2/tokenizer-comparison experiments as historical context; its old “current state” and next-step recommendations should not override the newer plan.

## Current state

The active task is the frozen VQ-VAE codec, not the code generator. The strict C# corpus has 23,566 training chunks and 2,263 file-held-out validation chunks at 64 GPT-2 tokens each. All metrics below describe codec reconstruction only.

| Codec | Held-out token accuracy | Exact chunks | Character similarity | 6-snippet pilot compiles |
|---|---:|---:|---:|---:|
| 4 levels, 256 codes/level | 81.87% | 2.47% | 71.58% | 0/6 |
| 8 levels, 256 codes/level | 85.81% | 4.11% | 76.83% | 0/6 |
| 16 levels, 256 codes/level | 90.36% | 8.35% | 83.73% | 0/6 |
| 32 levels, 256 codes/level | 93.50% | 16.22% | 89.18% | 0/6 |
| 32 levels, 1,024 codes/level | 93.64% | 16.22% | 89.21% | 0/6 |

Increasing levels improved round-trip fidelity. Increasing the codebook from 256 to 1,024 did not improve exact-chunk rate or pilot compilation. Therefore, finish the already prepared 64-level stress test, but treat it as a capacity limit probe—not as the solution to code generation.

## Work that can proceed alongside the 64-level run

### Larger compile benchmark

`examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl` contains 48 short, self-contained C# methods. Their originals compile together using the same .NET 8 target and minimal Godot `Vector2` stub as the evaluator. The suite is held out from training. Run the 64-level codec over it with `--compile`, then repeat with prior checkpoints. Require the original compile count to equal 48 before interpreting reconstruction compile rate.

This is still a syntax/type smoke benchmark, not a semantic test. It should be expanded with behavior cases and complete Godot project tests after a generated-code stage exists.

### Base-tokenizer comparison

`evaluation/compare_code_tokenizers.py` screens tokenizer-only behavior against Qwen2.5-Coder on the same held-out GPT-2 chunks. The current candidate list covers GPT-2, CodeBERT, CodeT5, StarCoder2, DeepSeek Coder and Coder V2, Qwen3-Coder, Granite Code, CodeGemma, and CodeLlama. This is only a screening test: token count and identifier splits do not establish NCM codec quality. Several candidates may need newer Transformers, HF login, or model access; the script records an individual load failure and continues. Qwen2.5 remains the known baseline. StarCoder2's 49,152-token vocabulary is close to GPT-2's 50,257-token base vocabulary, DeepSeek Coder 1.3B has 32,256 entries, DeepSeek Coder V2 has 102,400, and Qwen3-Coder has 151,936. A matched 32-level StarCoder2 ablation is configured in `config/train/godot-csharp-starcoder2-strict-32level-tokenizer.yaml`. Promote other candidates to NCM training only after the quick screen shows a plausible advantage.

Run the quick screen during the existing 64-level GPU training if desired; it uses tokenizer inference on CPU and does not start training or allocate GPU memory. It downloads tokenizer files from Hugging Face as needed. The input is the same fixed set of held-out text chunks, so the first-pass token counts and string round-trip results are directly comparable:

```bash
.venv/bin/python evaluation/compare_code_tokenizers.py \
  --input data/mycode/godot-csharp-strict/validation.jsonl \
  --text-key text \
  --sequence-length 64 \
  --output runs/tokenizer-comparison/qwen-baseline-expanded.json
```

This diagnostic uses the existing GPT-2 chunk boundaries. Its `over-64` column is a screening signal, not a codec result. Each finalist must later be retokenized and rechunked from the same source files before its NCM codec is trained.

The expanded screening results show Qwen2.5-Coder at 30.6 tokens per GPT-2 validation chunk. Qwen3-Coder produced the identical token IDs on every comparison sample, so it does not need a separate codec run. StarCoder2 and Granite Code both averaged 33.3 tokens and have near-identical identifier metrics, but their token-ID fingerprints differ, so keep both as distinct codec candidates. DeepSeek Coder V2 averaged 37.3 and DeepSeek Coder 1.3B averaged 40.1, so neither beats Qwen on token count. They remain useful as low-vocabulary codec tradeoff candidates, because the VQ-VAE must predict among fewer output token classes. This is a compute/quality hypothesis, not evidence that their codec will be better.

For a fair codec comparison, keep model architecture, 32 quantizer levels, 256 codes per level, optimizer, steps, seed, sequence length, file-level train/validation split, and source files fixed. Rebuild each corpus with the candidate tokenizer: token IDs and chunk boundaries are tokenizer-specific. Compare the resulting codecs on equivalent held-out source files and the same 48-method compile suite. Token accuracy is tokenizer-specific and is not directly comparable across tokenizers; emphasize exact decoded text/chunks, character edit similarity, original-versus-reconstruction compile counts, peak VRAM, throughput, and encoded NCM codes per source character. Do not finalize a tokenizer based on a lower token count alone.

The Qwen corpus must be rebuilt with Qwen tokenization; never reuse GPT-2 token caches. Qwen's much larger vocabulary (151,936 entries for this configured tokenizer) increases VQ-VAE embedding/output compute, so begin with a short pilot and check peak VRAM and throughput before committing to a full run. StarCoder2's published vocabulary is 49,152; the custom pad token adds one entry in this config.

Build a separate file-held-out corpus for that experiment. The GPT-2 validation rows cannot be reused as Qwen rows because the chunk boundaries were made with a different tokenizer:

```bash
.venv/bin/python -m training.build_reference_corpus \
  --root data/reference \
  --output data/mycode/godot-csharp-qwen-strict \
  --tokenizer Qwen/Qwen2.5-Coder-0.5B \
  --sequence-length 64 \
  --max-doc-fraction 0 \
  --include-extensions .cs,.csproj

.venv/bin/python -m training.prepare_data \
  --config config/train/godot-csharp-qwen-strict-32level-tokenizer.yaml \
  --stage tokenizer

.venv/bin/python -m training.prepare_data \
  --config config/train/godot-csharp-qwen-strict-32level-tokenizer.yaml \
  --stage tokenizer \
  --input data/mycode/godot-csharp-qwen-strict/validation.jsonl \
  --cache-dir data/cache/godot-csharp-qwen-strict/tokenizer-validation
```

Prepare the StarCoder2 candidate the same way, keeping a separate corpus and cache:

```bash
.venv/bin/python -m training.build_reference_corpus \
  --root data/reference \
  --output data/mycode/godot-csharp-starcoder2-strict \
  --tokenizer bigcode/starcoder2-3b \
  --sequence-length 64 \
  --max-doc-fraction 0 \
  --include-extensions .cs,.csproj

.venv/bin/python -m training.prepare_data \
  --config config/train/godot-csharp-starcoder2-strict-32level-tokenizer.yaml \
  --stage tokenizer

.venv/bin/python -m training.prepare_data \
  --config config/train/godot-csharp-starcoder2-strict-32level-tokenizer.yaml \
  --stage tokenizer \
  --input data/mycode/godot-csharp-starcoder2-strict/validation.jsonl \
  --cache-dir data/cache/godot-csharp-starcoder2-strict/tokenizer-validation
```

Two additional low-vocabulary codec candidates are configured for the same 32-level/256-code architecture:

```bash
# DeepSeek Coder 1.3B tokenizer
.venv/bin/python -m training.build_reference_corpus \
  --root data/reference \
  --output data/mycode/godot-csharp-deepseek-coder-1.3b-strict \
  --tokenizer deepseek-ai/deepseek-coder-1.3b-base \
  --sequence-length 64 --max-doc-fraction 0 --include-extensions .cs,.csproj
.venv/bin/python -m training.prepare_data \
  --config config/train/godot-csharp-deepseek-coder-1.3b-strict-32level-tokenizer.yaml --stage tokenizer
.venv/bin/python -m training.prepare_data \
  --config config/train/godot-csharp-deepseek-coder-1.3b-strict-32level-tokenizer.yaml \
  --stage tokenizer --input data/mycode/godot-csharp-deepseek-coder-1.3b-strict/validation.jsonl \
  --cache-dir data/cache/godot-csharp-deepseek-coder-1.3b-strict/tokenizer-validation

# DeepSeek Coder V2 tokenizer
.venv/bin/python -m training.build_reference_corpus \
  --root data/reference \
  --output data/mycode/godot-csharp-deepseek-coder-v2-strict \
  --tokenizer deepseek-ai/DeepSeek-Coder-V2-Base \
  --sequence-length 64 --max-doc-fraction 0 --include-extensions .cs,.csproj
.venv/bin/python -m training.prepare_data \
  --config config/train/godot-csharp-deepseek-coder-v2-strict-32level-tokenizer.yaml --stage tokenizer
.venv/bin/python -m training.prepare_data \
  --config config/train/godot-csharp-deepseek-coder-v2-strict-32level-tokenizer.yaml \
  --stage tokenizer --input data/mycode/godot-csharp-deepseek-coder-v2-strict/validation.jsonl \
  --cache-dir data/cache/godot-csharp-deepseek-coder-v2-strict/tokenizer-validation

# IBM Granite Code tokenizer (distinct token IDs from StarCoder2 on the screen)
.venv/bin/python -m training.build_reference_corpus \
  --root data/reference \
  --output data/mycode/godot-csharp-granite-code-strict \
  --tokenizer ibm-granite/granite-8b-code-base-4k \
  --sequence-length 64 --max-doc-fraction 0 --include-extensions .cs,.csproj
.venv/bin/python -m training.prepare_data \
  --config config/train/godot-csharp-granite-code-strict-32level-tokenizer.yaml --stage tokenizer
.venv/bin/python -m training.prepare_data \
  --config config/train/godot-csharp-granite-code-strict-32level-tokenizer.yaml \
  --stage tokenizer --input data/mycode/godot-csharp-granite-code-strict/validation.jsonl \
  --cache-dir data/cache/godot-csharp-granite-code-strict/tokenizer-validation
```

The full codec comparison should train the Qwen2.5 baseline plus StarCoder2, Granite Code, DeepSeek Coder 1.3B, and DeepSeek Coder V2. Qwen3-Coder is excluded because it duplicates Qwen2.5 token IDs on this suite; GPT-2 and CodeBERT tokenize much less efficiently, CodeT5 does not load with the current tokenizer stack, and CodeGemma/CodeLlama require gated HF access. Give each candidate the same number of source-corpus passes. Compute each candidate's step count as `ceil(train_rows / 16) * epochs` because tokenizer-specific chunk counts differ. Run the candidates sequentially after the current GPU job finishes. Evaluate their best checkpoints on each candidate's own held-out chunks and the same compile stress suite. Use decoded text/compile results and cost per source character as the cross-tokenizer outcomes; token accuracy values are only comparable within one tokenizer.

## Decision after 64 levels

1. Compare held-out exact chunks and the 48-method compile rate, not only token accuracy.
2. If 64 levels materially improves exactness and compile rate, decide whether the accuracy justifies the larger representation. A 64-level NCM generator needs a prediction step for every level, so latency is a real cost.
3. If 64 levels improves average accuracy but compile rate remains poor, stop increasing levels. Compare a code-aware base tokenizer and consider an exact residual channel.

## Exact residual-channel hypothesis

The current encode/decode path is lossy: it quantizes continuous latents to discrete codes, and the decoder predicts base-token IDs. One possible hybrid is to serialize corrections for positions where reconstruction differs from the source: a compact position mask plus the original token IDs at those positions. Decoding would first reconstruct from NCM codes and then apply the corrections.

This could make tokenizer round-trip exact, but it does not automatically make generation exact. The generator would also have to predict the correction locations and token IDs. That adds a second target stream and consumes output capacity. Prototype it only after the 64-level and code-tokenizer measurements, and compare correction rate, bytes/codes per source token, generator prediction cost, and exact compile results.

## Generator readiness and project validation

Even a lossless codec does not teach the NCM generator to turn broad instructions into a game. The current generator pilot is only 16 train, 4 validation, and 6 test method examples, with 64-token targets. It is a pipeline smoke set, not meaningful code-generation training data.

Before a serious generator run:

1. Build instruction-to-code/project examples with repository-level train/validation/test splits and explicit licenses/provenance.
2. Increase context and target lengths enough for full classes and project files; keep a matching generator and tokenizer configuration.
3. Evaluate generated candidates in temporary worktrees: format, compile with the intended .NET/Godot version, run unit tests, run headless Godot import/project checks, and capture failures.
4. Use compiler diagnostics and tests for a bounded repair loop. Treat generated code as untrusted: run only in disposable, resource-limited workspaces without credentials or network access.
5. Measure task success, compile rate, test pass rate, repair count, and latency. Token loss alone is not the product metric.

The deterministic compiler/test/repair tools can be separate Hermes tools or agents. They should validate and repair outputs; they do not replace the NCM generator's need for instruction-conditioned training data.
