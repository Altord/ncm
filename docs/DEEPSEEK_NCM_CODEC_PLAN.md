# DeepSeek NCM codec plan: Godot and C#

## Purpose

This document is the working record for the selected code-tokenizer codec and its next experiments. It separates observed results from project decisions and hypotheses. Keep interim findings in the active investigation section. Add a checkpoint only after the problem section has a verified solution; do not label experiments or partial findings as checkpoints.

## Current decision

**Selected target for the final codec run: DeepSeek Coder 1.3B tokenizer + 64-level NCM codec.** This is the project direction. The combined configuration has not yet completed a run, so the measured results below are evidence for the components, not proof of the combination.

The tokenizer is `deepseek-ai/deepseek-coder-1.3b-base` (32,256 base vocabulary entries; one custom pad token makes the NCM input/output vocabulary 32,257). The chosen level count is 64, with 256 codes per level. The near-term controlled run keeps the sequence length at 64 and uses 40 passes over the current training corpus.

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

### Step 3a — syntax-sensitive reconstruction loss

`training/reconstruction_loss.py` builds per-token weights from the configured Hugging Face tokenizer: C# keywords and punctuation/operator pieces receive weight 3.0; alphanumeric pieces receive weight 1.5; other pieces retain weight 1.0. `training/train_tokenizer.py` uses this only in the reconstruction cross-entropy used for gradient updates. Validation CE, token accuracy, and exact-chunk rate remain unweighted, so reported metrics stay comparable to the baseline. This lexical categorization is a heuristic, not a syntax parser, and it may misclassify BPE pieces; only the matched compile evaluation can show whether the change helps.

The experiment config is `config/train/godot-csharp-deepseek-coder-1.3b-strict-8level-decoder4-syntax-weighted-40pass.yaml`. It holds the tokenizer, 8-level schedule, decoder depth (4), corpus, optimizer, and training duration fixed against the decoder-4 result. The only intended change is the loss weighting. Compare its held-out reconstruction and compile-stress metrics against the decoder-4 result above. If syntax errors decline but identifiers remain damaged, revise the lexical objective based on diagnostics; if weighted loss does not help compilation, do not promote it to the 64-level run as a proven fix.

### Step 4 — train and evaluate the selected 64-level configuration

The selected final codec target remains DeepSeek Coder 1.3B + 64 levels. Use the decoder depth selected by Step 3. Start from `config/train/godot-csharp-deepseek-coder-1.3b-strict-64level-40pass.yaml`, changing `dec_layers` only if the Step 3 result supports it. Keep the same corpus, 64-token context, optimizer, and 40-pass schedule.

Compare its held-out reconstruction and 48-snippet compile metrics against the fixed 8-level baselines. This tests the level-count change after the decoder-capacity question has been addressed.

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
