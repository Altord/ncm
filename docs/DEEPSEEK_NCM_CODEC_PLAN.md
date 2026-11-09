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

## Checkpoint 0 — selection and baseline recorded

**Status:** recorded; no new training started by this checkpoint.

- **Tokenizer decision:** DeepSeek Coder 1.3B is selected.
- **Target level count:** 64.
- **Measured reference codec:** DeepSeek Coder 1.3B, 8 levels, 40 passes.
- **Reference metrics:** held-out token accuracy 80.19%; exact chunks 60/1,530 (3.92%); character similarity 76.58%; shared compile set 0/48.
- **Immediate next work:** inspect the 48 saved compile failures before training the selected 64-level combination.

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

**Current interpretation:** this is genuine reconstruction corruption, not input clipping or a failing original/harness. The errors are not limited to formatting. This is evidence about the open problem, not a resolved checkpoint. The next controlled step remains the selected DeepSeek + 64-level run; afterward, compare its compiler diagnostics and reconstructed examples against this baseline.

### Step 2 — train the selected DeepSeek + 64-level codec

The controlled 64-level config is `config/train/godot-csharp-deepseek-coder-1.3b-strict-64level-40pass.yaml`. It keeps tokenizer, source corpus, 64-token context, batch size, optimizer, and 40-pass schedule aligned with the DeepSeek 8-level reference. The expected run is 44,080 updates for the current 17,630-row training corpus.

After training, record the selected checkpoint step, validation token accuracy, exact-chunk rate, reconstruction CE, codebook utilization, throughput, and peak VRAM. Do not treat validation token accuracy alone as a compile-quality result.

### Step 3 — evaluate the selected codec on held-out code and the shared compile set

Use its best saved artifact, held-out file split, and the same 48 stress snippets. Record character similarity, exact chunks, token accuracy, original/reconstructed compile counts, and compiler error classes. The stress-set originals must still compile 48/48.

**Decision gate:** compare the 64-level result against the recorded DeepSeek 8-level, 40-pass reference. This answers whether the chosen configuration improves actual code reconstruction and compilation, not just training loss.

### Step 4 — respond to remaining failures without reopening tokenizer selection

If DeepSeek + 64 levels still fails compilation, use the failure categories from Step 1 to choose one controlled change at a time while keeping the selected tokenizer and 64 levels fixed:

1. **Decoder capacity:** test decoder depth or latent width, changing only one setting per run.
2. **Syntax-aware reconstruction objective:** test whether emphasizing delimiters, operators, and other syntax-critical tokens improves compile rate; compare against ordinary token CE.
3. **Training data and chunk boundaries:** audit the strict C# corpus for duplicates, malformed examples, generated/vendor files, and windows that split useful structures. Keep file-level held-out separation.
4. **Context length:** later test 128/256-token contexts for larger classes and scripts. This addresses complete-program scope; it is not needed to explain the current 48-snippet failures because all 48 inputs fit in 64 tokens.

Keep all run directories and checkpoint reports separate. Use the same validation source split and stress suite, and record the change plus result in a new checkpoint section below.

### Step 5 — only then train the code generator

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
