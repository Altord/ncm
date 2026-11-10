# DeepSeek NCM codec plan: Godot and C#

## Purpose

This document is the working record for the selected code-tokenizer codec and its next experiments. It separates observed results from project decisions and hypotheses. Keep interim findings in the active investigation section. Add a checkpoint only after the problem section has a verified solution; do not label experiments or partial findings as checkpoints.

## Current decision

**Current requested codec run: DeepSeek Coder 1.3B tokenizer + 64 quantizer levels.** The user has specified 128,000 updates, codebook size 512, four decoder blocks, syntax/identifier loss weights, and asked me to choose the context and remaining settings. The resulting recipe is exploratory and has not completed a run; prior results are evidence about earlier components, not proof that this combined setup will work.

The tokenizer is `deepseek-ai/deepseek-coder-1.3b-base` (32,256 base vocabulary entries; one custom pad token makes the NCM input/output vocabulary 32,257). The requested run uses 64 quantizer levels and 512 codes per level. I selected a 128-token context to expose more complete C# methods/classes per example, with a strictly code-only corpus of `.cs` and `.csproj` files. The recipe uses four decoder Transformer blocks and 128,000 optimizer updates.

### Current requested recipe (not yet run)

| Setting | Choice | Reason / interpretation |
|---|---|---|
| Base tokenizer | DeepSeek Coder 1.3B | Existing project choice; code-oriented base tokenization. |
| Context | 128 tokens | Longer code chunks than the previous 64-token codec; useful for methods and larger syntax spans. Requires rebuilding train and validation corpora/caches. |
| Quantizer | 64 stages, resolutions `[1,3,5,...,125,128]` | Sixty-four distinct, increasing resolutions ending at the full context. |
| Codebook | 512 entries per stage | User-requested increase from 256; expands per-stage choices and quantizer compute. |
| Decoder | 4 Transformer blocks | User-requested; prior 4-block test did not fix compilation, but adding depth did not hurt token reconstruction materially. |
| Reconstruction objective | C# syntax pieces weight 3.0; alphanumeric/identifier/literal pieces 1.5; others 1.0 | Syntax-weighting implementation is a heuristic. Its 20-step smoke run confirmed startup only, not quality. |
| VQ loss weight | 1.0 | Keep the established baseline balance; no evidence yet supports retuning it. |
| Training data | Code-only `.cs` and `.csproj`; docs excluded | Tokenizer stage should reconstruct source syntax. Documentation can be used later for generator training or retrieval. |
| Data split | 90/10 by stable source-file hash | Prevents chunks from one file crossing train/validation. |
| Optimizer schedule | AdamW, cosine LR 3e-4 to 3e-5 after 50 warmup updates; batch 16; fp32 | Reuse established recipe and likely fit 16 GB VRAM at this model width/context. |
| Other fixed recipe settings | Seed 42; grad accumulation 1; betas (0.9, 0.95); weight decay 0.01; gradient clip 1.0; dropout 0; EMA decay 0.95; commitment cost 0.25; fine-level dropout 0.25, minimum one stage | Preserve the existing recipe where no experiment has justified a change. |
| Updates | 128,000 | User-requested. Step count is optimizer updates, not epochs; the rebuilt 128-token corpus will have fewer rows than the old 64-token corpus, so this may mean many data passes. |
| Checkpoint selection | Highest unweighted validation token accuracy | Smooth signal at 128 tokens; exact-sequence accuracy is logged but sparse. Compilation remains a separate post-training gate. |
| Save / validation cadence | Save every 5,000 updates; validate every 10,000 | Periodic checkpoints plus enough validation points to select a best artifact without evaluating every update. |

The complete config is `config/train/godot-csharp-deepseek-coder-1.3b-strict-64level-code512-decoder4-ctx128-weighted-128k.yaml`. Because this run changes context, codebook size, decoder, objective, and steps together, its result cannot attribute any gain to a single lever.

#### Data build and preparation commands

Run once after syncing `data/reference` on the Linux PC:

```bash
.venv/bin/python -m training.build_reference_corpus \
  --root data/reference \
  --output data/mycode/godot-csharp-deepseek-coder-1.3b-strict-ctx128 \
  --tokenizer deepseek-ai/deepseek-coder-1.3b-base \
  --sequence-length 128 \
  --max-doc-fraction 0 \
  --include-extensions .cs,.csproj

.venv/bin/python -m training.prepare_data \
  --config config/train/godot-csharp-deepseek-coder-1.3b-strict-64level-code512-decoder4-ctx128-weighted-128k.yaml \
  --stage tokenizer

.venv/bin/python -m training.prepare_data \
  --config config/train/godot-csharp-deepseek-coder-1.3b-strict-64level-code512-decoder4-ctx128-weighted-128k.yaml \
  --stage tokenizer \
  --input data/mycode/godot-csharp-deepseek-coder-1.3b-strict-ctx128/validation.jsonl \
  --cache-dir data/cache/godot-csharp-deepseek-coder-1.3b-strict-ctx128/tokenizer-validation
```

Inspect the generated manifest and counts before starting the 128,000-update run. The builder's file-level split hashes source paths, so train and validation remain disjoint when only the chunk length changes.

**Corpus build result:** 9,103 training chunks from 1,764 source files and 803 validation chunks from 209 held-out files; no documentation chunks. At batch size 16 this is 569 optimizer steps per pass, so 128,000 updates are approximately 225 passes. The Transformers warning about a 244,413-token input was emitted while the corpus builder tokenized one long source file before chunking it; corpus creation and fixed-length cache preparation both completed successfully. Each model example is still truncated/padded to the configured 128-token context.

```bash
N=$(wc -l < data/mycode/godot-csharp-deepseek-coder-1.3b-strict-ctx128/train.jsonl)
EPOCH_STEPS=$(((N + 15) / 16))
printf 'training_rows=%s optimizer_steps_per_epoch=%s estimated_passes=%.1f\n' \
  "$N" "$EPOCH_STEPS" "$(awk -v steps=128000 -v epoch_steps="$EPOCH_STEPS" 'BEGIN { print steps / epoch_steps }')"
```

#### Smoke run, full launch, and live monitor

The context-128 / 512-code config is larger than the prior smoke-tested setup. First run 20 updates using an isolated temporary config whose run/artifact paths end in `smoke`; this checks initialization and VRAM without contaminating the real run. Its accuracy is not meaningful. If it completes, launch the full run. `systemd-inhibit` prevents system sleep, while `runuser` keeps repository files owned by the normal user.

```bash
cp config/train/godot-csharp-deepseek-coder-1.3b-strict-64level-code512-decoder4-ctx128-weighted-128k.yaml \
  /tmp/ncm-deepseek-64level-code512-ctx128-smoke.yaml
sed -i 's#64level-code512-decoder4-ctx128-weighted-128k#64level-code512-decoder4-ctx128-weighted-smoke#g' \
  /tmp/ncm-deepseek-64level-code512-ctx128-smoke.yaml

.venv/bin/python -m training.train_tokenizer \
  --config /tmp/ncm-deepseek-64level-code512-ctx128-smoke.yaml \
  --max-steps 20
```

After the smoke run completes:

```bash
sudo -v
mkdir -p runs/godot-csharp-deepseek-coder-1.3b-strict/64level-code512-decoder4-ctx128-weighted-128k

nohup sudo systemd-inhibit \
  --what=sleep \
  --why="NCM 64-level code512 context128 training" \
  runuser -u "$USER" -- \
  /home/josh/src/ncm/.venv/bin/python -m training.train_tokenizer \
    --config config/train/godot-csharp-deepseek-coder-1.3b-strict-64level-code512-decoder4-ctx128-weighted-128k.yaml \
    --max-steps 128000 \
  > runs/godot-csharp-deepseek-coder-1.3b-strict/64level-code512-decoder4-ctx128-weighted-128k/overnight.log \
  2>&1 < /dev/null &

echo "Launcher PID=$!"
```

In the alternate terminal:

```bash
.venv/bin/python scripts/watch_codec_tokenizer_screen.py \
  --run ds64-code512-decoder4-ctx128=runs/godot-csharp-deepseek-coder-1.3b-strict/64level-code512-decoder4-ctx128-weighted-128k:128000 \
  --interval 5
```

After training, run the held-out reconstruction and compile stress evaluation using the same config/checkpoint:

```bash
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config config/train/godot-csharp-deepseek-coder-1.3b-strict-64level-code512-decoder4-ctx128-weighted-128k.yaml \
  --checkpoint artifacts/godot-csharp-deepseek-coder-1.3b-strict/64level-code512-decoder4-ctx128-weighted-128k/tokenizer.pt \
  --input data/mycode/godot-csharp-deepseek-coder-1.3b-strict-ctx128/validation.jsonl \
  --text-key text --code-only \
  --output runs/godot-csharp-deepseek-coder-1.3b-strict/64level-code512-decoder4-ctx128-weighted-128k/validation/reconstruction.jsonl

.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config config/train/godot-csharp-deepseek-coder-1.3b-strict-64level-code512-decoder4-ctx128-weighted-128k.yaml \
  --checkpoint artifacts/godot-csharp-deepseek-coder-1.3b-strict/64level-code512-decoder4-ctx128-weighted-128k/tokenizer.pt \
  --input examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl \
  --text-key target --compile \
  --output runs/godot-csharp-deepseek-coder-1.3b-strict/64level-code512-decoder4-ctx128-weighted-128k/stress/compiled.jsonl
```

## Facts established so far

| Claim | Evidence | What it establishes |
|---|---|---|
| DeepSeek Coder 1.3B is the selected base tokenizer. | In the five-way short codec screen it led held-out reconstruction: 60.11% token accuracy, 58.11% character similarity, 8/1,530 exact chunks. | It is the tokenizer selected for follow-on work; this does not establish generator quality. |
| More training improved the DeepSeek 8-level codec. | At 3 passes: 60.11% held-out token accuracy, 58.11% character similarity, 8/1,530 exact chunks. At 40 passes: 80.19%, 76.58%, 60/1,530. | Training duration materially changed reconstruction quality at fixed tokenizer and level count. |
| The 8-level, 40-pass codec is still not compile-safe. | On the shared 48-snippet test: originals 48/48 compile; reconstructions 0/48 compile; character similarity 63.04%. | The current codec output cannot yet be accepted as valid code. |
| The stress snippets were not cut off by the 64-token context. | Token-length audit: DeepSeek Coder 1.3B maximum 61 tokens; all 48 snippets fit. | The zero compile result is not explained by context truncation. |
| The NCM model ran on the AMD GPU through ROCm/HIP. | The environment reported PyTorch 2.12.0+rocm7.14.1, HIP 7.14.60850, RX 9060 XT, and runtime MIOpen(HIP) warnings. PyTorch retains `torch.cuda` and `device="cuda"` names for its HIP backend. | The `cuda` device label is normal for this ROCm PyTorch build; it does not mean the run used an NVIDIA GPU. |
| The 64-level GPT-2 codec previously reconstructed more accurately than the earlier GPT-2 codecs. | Prior held-out report: 94.21% token accuracy, 19.13% exact chunks, 90.40% character similarity. A separate 48-snippet compile run reported 8/48 compilable reconstructions. | Higher levels improved fidelity in that GPT-2 experiment. It does not prove the DeepSeek + 64-level result. |

## Selected target and measured reference

- **Tokenizer decision:** DeepSeek Coder 1.3B is selected.
- **Target level count:** 64.
- **Measured reference codec:** DeepSeek Coder 1.3B, 8 levels, 40 passes.
- **Reference metrics:** held-out token accuracy 80.19%; exact chunks 60/1,530 (3.92%); character similarity 76.58%; shared compile set 0/48.
- **Immediate next work:** test decoder depth against the existing 8-level baseline before training the selected 64-level combination.

## Step-by-step plan

### Step 1 — inspect the 8-level compile failures

Use the saved shared-stress output from the Linux machine. This repository checkout does not contain that ignored `runs/` output, so compiler details must be collected on the Linux PC.

```bash
cd ~/src/ncm
.venv/bin/python evaluation/inspect_codec_failures.py \
  --input runs/godot-csharp-deepseek-coder-1.3b-strict/8level-40pass/stress/compiled.jsonl \
  > runs/godot-csharp-deepseek-coder-1.3b-strict/8level-40pass/stress/failures.txt

tail -n 30 runs/godot-csharp-deepseek-coder-1.3b-strict/8level-40pass/stress/failures.txt
```

Record:

- the most common compiler diagnostic codes;
- whether errors cluster around braces, punctuation/operators, identifiers, or missing/truncated content;
- a few representative reference/reconstruction pairs;
- whether the original still compiles 48/48.

**Interim investigation findings (problem remains open):** the report showed originals pass 48/48 and reconstructions pass 0/48. No stress example exceeded the 64-token context. The examples contain substitutions or deletions in declarations, identifiers, keywords, operators, and expressions (for example, `static int` becoming `ant int`, and `count <= 0` becoming `Token values0`). The most frequent compiler diagnostics were CS1519 (131), CS1002 (66), CS1003 (47), CS1001 (45), and CS1525 (30).

**Current interpretation:** this is genuine reconstruction corruption, not input clipping or a failing original/harness. The errors affect keywords, identifiers, and expressions, not only formatting. Decoder-depth testing is complete and did not improve compile rate. The next controlled step is syntax-sensitive reconstruction loss; the selected DeepSeek + 64-level codec remains a later, unverified target.

### Step 2 — controlled decoder-depth test at 8 levels

The ablation config is `config/train/godot-csharp-deepseek-coder-1.3b-strict-8level-decoder4-40pass.yaml`. It holds the selected tokenizer, 8-level quantizer, corpus, context, training schedule, and optimizer fixed against the 8-level/40-pass reference. The only intended model change is `dec_layers: 1` to `dec_layers: 4`. In this implementation `dec_layers` is the total number of decoder Transformer blocks; this adds three residual self-attention/MLP blocks while keeping the 64-dimensional latent width unchanged. The expected run is 44,080 updates for the current 17,630-row training corpus. The quantizer's `manual_levels` values are latent sequence resolutions; they must end at the 64-position context length. A first attempted config mistakenly used `[1,2,3,4,5,6,7,8]`, causing a shape error on the first forward pass. It was corrected to `[1,2,4,8,16,32,48,64]`; no training updates completed in the failed attempt.

After training, record selected checkpoint step, held-out token accuracy, exact-chunk rate, reconstruction CE, codebook utilization, throughput, and peak VRAM. Do not treat validation token accuracy alone as a compile-quality result.

**Decoder-depth result (completed, problem still open):** selected step 44,080. Against the 1-block reference, held-out accuracy changed 80.19% → 80.43%, exact chunks 3.92% → 3.79%, and character similarity 76.58% → 76.86%. On the shared stress set, token accuracy changed 70.96% → 72.27%, character similarity 63.04% → 63.02%, and reconstructed compilation remained 0/48 (originals 48/48). The additional decoder blocks gave a small accuracy change but did not improve the outcome that matters: compilable reconstruction.

### Step 3 — evaluate the decoder-depth test

Use its best saved artifact, held-out file split, and the same 48 stress snippets. Record character similarity, exact chunks, token accuracy, original/reconstructed compile counts, and compiler error classes. The stress-set originals must still compile 48/48.

**Decision gate:** compare against the measured DeepSeek 8-level/40-pass, one-block decoder reference. Four blocks made slight changes to reconstruction metrics but left compile rate at 0/48, so decoder depth alone is not a solution. Failure inspection shows lexical damage across reserved words, identifiers, and expressions (examples include `static` → `as`, a method name changing, and `count <= 0` becoming `countIndex0`). Frequent parser diagnostics remain CS1519, CS1003, CS1001, and CS1002. Keep the problem open; the next experiment tests a syntax-sensitive reconstruction objective with the codec and decoder held fixed.

### Deferred ablation — syntax-sensitive reconstruction loss at 8 levels

`training/reconstruction_loss.py` builds per-token weights from the configured Hugging Face tokenizer: C# keywords and punctuation/operator pieces receive weight 3.0; alphanumeric pieces receive weight 1.5; other pieces retain weight 1.0. `training/train_tokenizer.py` uses this only in the reconstruction cross-entropy used for gradient updates. Validation CE, token accuracy, and exact-chunk rate remain unweighted, so reported metrics stay comparable to the baseline. This lexical categorization is a heuristic, not a syntax parser, and it may misclassify BPE pieces; only the matched compile evaluation can show whether the change helps.

The one-factor config is `config/train/godot-csharp-deepseek-coder-1.3b-strict-8level-decoder4-syntax-weighted-40pass.yaml`. It holds the tokenizer, 8-level schedule, decoder depth (4), corpus, optimizer, and duration fixed against the decoder-4 result. This isolated full run is deferred: the user requested the combined 64-level/512-code/context-128/decoder-4/128k recipe instead. Preserve this config as a later ablation if the combined run improves but the source of improvement remains unclear.

The 20-update smoke run completed and reached validation, confirming the weighted-loss path initializes and trains. Its reported 0% token accuracy is expected at this deliberately tiny step count and is not a quality result.

### Step 4 — train and evaluate the selected 64-level configuration

The previous 64-level/40-pass recipe is superseded for the next run by the current requested recipe table above: 64 quantizer stages, 512 codes per stage, four decoder blocks, context 128, syntax-weighted reconstruction, and 128,000 updates. Build the matching 128-token code-only data and caches before training.

Compare its held-out reconstruction and the same 48-snippet compile metrics against the fixed 8-level baselines. Because the requested run changes several levers at once, treat the outcome as a combined result, not a causal ablation.

**Superseded unrun proposal:** an earlier plan briefly considered 128 residual quantizer stages at context 64, repeating each resolution twice. It was not started; the current recipe below replaces that proposal with the latest user-requested 64-stage, 512-code setup at context 128.

### Step 5 — respond to remaining failures without reopening tokenizer selection

If DeepSeek + 64 levels still fails compilation, use the failure categories from Step 1 to choose one controlled change at a time while keeping the selected tokenizer and 64 levels fixed:

1. **Decoder capacity:** test decoder depth or latent width, changing only one setting per run.
2. **Syntax-aware reconstruction objective:** test whether emphasizing delimiters, operators, and other syntax-critical tokens improves compile rate; compare against ordinary token CE.
3. **Training data and chunk boundaries:** audit the strict C# corpus for duplicates, malformed examples, generated/vendor files, and windows that split useful structures. Keep file-level held-out separation.
4. **Context length:** later test 128/256-token contexts for larger classes and scripts. This addresses complete-program scope; it is not needed to explain the current 48-snippet failures because all 48 inputs fit in 64 tokens.

Keep all run directories and checkpoint reports separate. Use the same validation source split and stress suite. Record interim measurements in the active investigation notes; create a final-solution checkpoint only after a tested change resolves the section.

### Step 6 — only then train the code generator

The codec reconstructs code; it does not learn to turn broad instructions into a program. Before generator training, prepare licensed instruction-to-code examples at method, class, and project scope, and ensure the generator predicts the complete 64-level code representation. Evaluate generated output with formatting, compilation, tests, and Godot headless project checks. Keep documentation/API references for generator learning or retrieval, not as a substitute for codec reconstruction data.

## Final-solution checkpoint template

Use this only when a section is resolved by a verified fix. Do not add checkpoints for experiments, interim results, or hypotheses:

```text
### Checkpoint N — YYYY-MM-DD — short experiment name
Changed:
Kept fixed:
Training rows / validation rows:
Steps / passes / selected checkpoint:
Validation token accuracy / exact chunks / character similarity:
Stress examples / original compile / reconstructed compile:
Top compiler errors:
Throughput / peak VRAM / elapsed time:
Conclusion (fact):
Decision or hypothesis:
Next step:
```

## Final-solution checkpoint history

No codec problem section has a verified final solution yet. The DeepSeek tokenizer and 64-level configuration are selected project choices; the 64-level DeepSeek result and compile reliability are still unverified.

## Codebook-capacity ablation — 64 levels × 4,096 entries

The 128,000-step weighted DeepSeek run reached 95.55% best validation token accuracy and 24/48 compile passes on the 48-snippet stress set. The 100,000-step checkpoint scored 23/48 on the same stress set, with nearly identical token/chunk/similarity metrics. This is evidence that simply training longer was not enough; identifier and syntax token substitutions still cause many failures.

The next experiment changes one model capacity variable: codebook entries per level rise from 512 to 4,096 (an eightfold increase). It keeps the tokenizer, corpus and file split, 64 quantizer levels, 128-token context, four decoder blocks, weighted reconstruction objective, optimizer, batch size, and checkpoint metric unchanged. It is capped at 30,000 steps for an initial capacity/throughput check. The config is `config/train/godot-csharp-deepseek-coder-1.3b-strict-64level-code4096-decoder4-ctx128-weighted-30k.yaml`.

4096 is a power-of-two codebook size, not a number of quantizer levels. This is a plausible test because many codebooks were close to/full utilization, but utilization alone does not prove the codebook is the limiting factor. Each nearest-code lookup becomes more expensive; expect lower throughput, so measure actual steps/second before extrapolating wall time. The code vectors themselves are not the main memory concern; distance computation is the likely cost.

Use the same 128-token validation split and identical 48-snippet stress suite after training. Compare against the 512-entry run's step-30,000 checkpoint (if available) for a matched-update comparison, and also report the best validation-selected checkpoint. The decision metric is reconstructed compile passes, with token accuracy, exact-chunk rate, character similarity, validation reconstruction CE, utilization, VRAM, and throughput as supporting measurements. Do not infer a win from codebook utilization or training CE alone. If 4,096 entries do not improve syntax/identifier preservation or stress compilation, the next controlled change should target chunk/code-span supervision or boundary alignment rather than increasing codebook size again.
