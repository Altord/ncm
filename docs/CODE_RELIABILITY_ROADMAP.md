# NCM coding reliability roadmap

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

`evaluation/compare_code_tokenizers.py` compares GPT-2, CodeBERT, and CodeT5 tokenizers on the same snippets. It reports token counts, exact text round-trip, 64-token overflow, and pieces per identifier-like span. The diagnostic does not establish which tokenizer produces the best NCM codec. Any alternative must be trained with matching vocabulary and special-token IDs, then compared on the held-out corpus and compile suite.

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
