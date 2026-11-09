# NCM Training Runbook

**Purpose:** Explain the published NCM training method, the trainer implemented in this checkout, how to start and manage runs, where outputs go, and how to adapt experiments for coding or another task.

**Repository status checked 2026-11-06:** This checkout includes tokenizer and generator training entry points, JSONL-to-memory-map preparation, validation, resumable checkpoints, and starter configs. These have been reviewed against this checkout's model interfaces but have not yet been run on the Linux ROCm machine. The configs are tiny mechanics baselines and do not establish code quality.

For tensor-level details, read [NCM_DATA_PATH.md](NCM_DATA_PATH.md). For the overall game-generation project, see [../GAME_GENERATION_PLAN.md](../GAME_GENERATION_PLAN.md).

## 1. What training creates

NCM has two separately trained stages:

1. **Tokenizer/codec (VQVAE):** encodes base-token sequences into a hierarchy of discrete code IDs and decodes partial or complete code reconstructions back to base-token distributions.
2. **Generator (NextConceptTransformer):** trained after the tokenizer is frozen; predicts code IDs from coarse to fine using accumulated earlier levels.

A trained task therefore needs a compatible tokenizer checkpoint, generator checkpoint, both configs, and the exact base tokenizer files/revision. The two checkpoints must agree on base vocabulary, codebook sizes, level schedule, latent width, and tokenization.

Training an NCM model alone does not create a Hermes agent, tool caller, image generator, game engine, or verification loop. It creates a text/code generation component. Orchestration, tools, art, build/playtest, and review remain separate parts of the system.

## 2. Implemented commands and current limits

The trainer is ordinary PyTorch code, driven by YAML and invoked from the repository root. It does not change Hermes. The local Linux ROCm setup needs a short runtime check before long runs; this Windows checkout has not attempted a GPU run.

- `training.prepare_data` prepares tokenizer sequences or frozen-tokenizer generator code caches.
- `training.train_tokenizer` trains reconstruction plus quantization loss and writes resumable checkpoints.
- `training.train_generator` trains per-level next-code cross-entropy with optional conditioning corruption and writes evaluator-compatible checkpoints.
- Both trainers support optional held-out validation caches, `--resume`, `--max-steps`, and `--device`.
- The starter config is only a data-flow and hardware pilot: GPT-2 tokenizer IDs, 64-token sequences, four small codebooks, and a small transformer. It is not a competitive code model. Larger configurations require clean validation and measured VRAM first.

First milestone: run briefly, verify save/reload/resume, and overfit a few examples. Do not launch paper-scale budgets yet.

## 3. Published training method

### 3.1 Tokenizer stage

The encoder maps base-token IDs x to continuous vectors z. The multiscale residual quantizer creates code sequences of increasing length. The decoder maps the cumulative quantized vectors to distributions over the original base-token vocabulary.

For a sampled nested-dropout cutoff k, the paper's tokenizer objective is:

$$
L_{tok}=-\frac{1}{L}\sum_i\log p_\psi(x^i\mid\hat z^{(k)})
+\beta\,\mathrm{MSE}(z,\mathrm{sg}[\hat z^{(k)}])
+\mathrm{MSE}(\mathrm{sg}[z],\hat z^{(k)}).
$$

The first term is token reconstruction cross-entropy. The other terms are commitment and quantization MSE terms; sg means stop-gradient. A straight-through estimator passes reconstruction gradients through discrete quantization to the encoder. Nested dropout drops later/finer levels during some training examples so early/coarse levels must be useful on their own.

The quantizer updates codebooks through EMA statistics in training mode. `training.train_tokenizer` calculates reconstruction cross-entropy, combines it with VQ loss, handles padding masks, optimizer/scheduler, validation, logging, and checkpoints. The fine-drop correction in `models/quant.py` applies the VQ penalty at the same sampled cumulative cutoff as reconstruction; EMA codebook statistics still update at every level.

For Godot/C# tokenizer configs, checkpoint selection uses held-out token accuracy (`checkpoint_metric: val_token_accuracy`, mode `max`) rather than the combined loss. This keeps checkpoint choice aligned with codec reconstruction; it does not change the training objective or gradients. Keep exact-chunk accuracy and compile results as additional evaluation gates, because high token accuracy alone can still leave many chunks with at least one error.

`training.build_reference_corpus` accepts `--include-extensions .cs,.csproj` to build a strict C# source corpus. Without this option, `--max-doc-fraction 0` removes prose but still includes Godot text resources such as `.tscn`, `.tres`, `.godot`, and `.gdshader`; the result is code/config-only, not C#-only. Use the same generated validation split for the mixed and strict C# checkpoint evaluations.

`config/train/godot-csharp-strict-8level-tokenizer.yaml` is a one-factor ablation for code reconstruction. It keeps the strict C# corpus and prototype settings fixed while changing the hierarchy from four levels `[1, 4, 16, 64]` to eight `[1, 2, 4, 8, 16, 32, 48, 64]`. Compare its held-out token accuracy, exact-chunk rate, and pilot compile passes with the four-level strict baseline before changing latent width or codebook size.

`config/train/godot-csharp-strict-16level-tokenizer.yaml` continues that capacity ablation from eight to sixteen hierarchy levels while keeping the same strict C# data, 64-dimensional latent width, 256 entries per codebook, and training settings. Its schedule ends at the 64-token context limit. Compare it against the four- and eight-level reports; more levels are useful only if held-out exact chunks and pilot compileability improve, not merely training loss.

`config/train/godot-csharp-strict-32level-1024code-tokenizer.yaml` tests codebook capacity at the 32-level setting. Relative to `godot-csharp-strict-32level-tokenizer.yaml`, it changes only each level's codebook size from 256 to 1024. This is motivated by the training log showing all 256 codes used at 31 of 32 levels. Compare held-out reconstruction and pilot compile results; codebook utilization alone is not an accuracy measure.

`config/train/godot-csharp-strict-32level-tokenizer.yaml` extends the same ablation to 32 levels. It has substantially more code positions per chunk (942 versus 396 at 16 levels), so expect more quantization and generator work. Keep the same 40-epoch exposure and compare held-out reconstruction and pilot compilation; stop increasing levels if code accuracy gains flatten or compilation remains unchanged.

### 3.2 Generator stage

Freeze the tokenizer and encode each example into clean target codes. Construct generator inputs from cumulative reconstructions of earlier levels. To reduce error accumulation at inference, the paper corrupts some conditioning inputs with random IDs from the same codebook, while leaving target labels clean.

The generator predicts every target level in one training forward pass. `training.train_generator` creates cumulative earlier-level inputs and applies weighted code cross-entropy:

$$
L_{NCM}=-\sum_k w_k\sum_{i=1}^{\ell_k}\log p_\theta(r_k^i\mid\tilde z^{(k-1)}),
\qquad
w_k=\frac{\ell_k^{-\alpha}}{\sum_j\ell_j^{1-\alpha}}.
$$

- alpha = 0 weights every code position equally (the ELBO-aligned choice).
- alpha = 1 gives each level equal total weight, increasing the relative influence of shorter coarse levels.
- Values above 1 emphasize coarse levels more; the paper's TinyStories ablation found alpha 1.5 shifted loss to coarse levels without improving final generative perplexity over alpha 0–1.
- Training predicts levels together; inference generates levels sequentially, with positions inside each level generated in parallel.

Conditional tasks use a prefix containing the prompt/source/puzzle. The paper trained XSum and WMT14 generators first on unpaired text, then source-target pairs. Unconditional tasks use a null prefix. This provides a pattern for instruction-to-code training, but this repository has no code dataset or code-specific training config.

### 3.3 Paper and implementation must be compared

- models/vqvae.py returns decoder logits and VQ loss, not reconstruction CE.
- models/next_concept.py returns generator logits, not generator loss.
- models/quant.py implements code assignment, EMA updates, fine-level dropout options, and code-to-input construction.
- Evaluation YAMLs are not training YAMLs; they omit training-only choices such as corruption schedule and loss weighting.
- Evaluation code uses CUDA-specific calls. The Linux Radeon machine needs ROCm/PyTorch validation and possibly backend portability work.

The paper specifies intended behavior; source specifies implemented behavior. A trainer must be checked against both.

## 4. Published experiment settings

These are the authors' reference settings, not a run recommendation for one 16 GB GPU. “Global batch” is across all devices.

### 4.1 Tokenizer

Columns: OWT, LM1B, TinyStories, XSum, WMT14, Sudoku.

| Setting | OWT | LM1B | TinyStories | XSum | WMT14 | Sudoku |
|---|---:|---:|---:|---:|---:|---:|
| Base vocabulary | 50,304 | 30,528 | 8,192 | 50,304 | 151,680 | 12 |
| Target length L | 1,024 | 128 | 128 | 48 | 64 | 89 |
| Levels K | 32 | 12 | 12 | 8 | 11 | 12 |
| Codebook vocabulary | 4,096 | 8,192 | 4,096 | 4,096 | 4,096 | 16 |
| Latent width | 384 | 384 | 384 | 384 | 384 | 256 |
| Encoder blocks | 0 | 0 | 0 | 0 | 0 | 0 |
| Decoder blocks | 1 | 4 | 1 | 1 | 1 | 1 |
| Decoder heads | 6 | 6 | 6 | 6 | 6 | 4 |
| Dropout | .3 | .1 | .3 | .1 | .1 | .3 |
| Updates | 78,432 | 25,957 | 30,000 | 120,001 | 60,000 | 10,000 |
| Global batch | 768 | 2,048 | 4,096 | 4,096 | 4,096 | 1,024 |
| Peak LR → floor | 3e-4 → 3e-5 | 3e-4 → 3e-5 | 5e-4 → 5e-5 | 5e-4 → 5e-5 | 5e-4 → 5e-5 | 5e-4 → 5e-5 |
| Warmup updates | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 500 |
| AdamW beta1/beta2 | .9/.95 | .9/.99 | .9/.95 | .9/.95 | .9/.95 | .9/.99 |
| Weight decay | .01 | .05 | .01 | .01 | .01 | .01 |
| Grad clip norm | 1 | 1 | 1 | 1 | 1 | 1 |
| Codebook EMA decay | .99 | .99 | .95 | .95 | .95 | .99 |
| Nested dropout | .5 | .7/.3* | .5 | .5 | .5 | .5 |

*LM1B uses complete and truncated hierarchies with 0.7/0.3 mixture weighting.

### 4.2 Generator

| Setting | OWT | LM1B | TinyStories | XSum | WMT14 | Sudoku |
|---|---:|---:|---:|---:|---:|---:|
| Parameters | 94.65M | 97.07M | 88.61M | 88.02M | 119.96M | 25.46M |
| Transformer blocks | 12 | 12 | 12 | 12 | 12 | 8 |
| Hidden width | 768 | 768 | 768 | 768 | 768 | 512 |
| Heads | 12 | 12 | 12 | 12 | 12 | 8 |
| Prefix length | 1 | 1 | 1 | 512 | 96 | 91 |
| RoPE base | 1e5 | 1e4 | 1e5 | 1e5 | 1e5 | 1e4 |
| Dropout | .1 | .1 | .1 | .3 | .1 | .1 |
| Updates | 250,000 | 100,000 | 50,000 | 350,000 | 150,000 | 50,000 |
| Global batch | 256 | 512 | 2,048 | 512 | 2,048 | 1,024 |
| Peak LR → floor | 3e-4 → 3e-5 | 3e-4 → 3e-5 | 3e-4 → 3e-5 | 3e-4 → 3e-5 | 3e-4 → 3e-5 | 3e-4 → 3e-5 |
| Warmup updates | 500 | 1,000 | 500 | 500 | 500 | 1,000 |
| AdamW beta1/beta2 | .9/.95 | .9/.95 | .9/.95 | .9/.95 | .9/.95 | .9/.99 |
| Weight decay | .1 | .1 | .1 | .1 | .01 | .01 |
| Grad clip norm | 1 | 1 | 1 | 1 | 1 | 1 |

Paper experiments use seed 42, BF16 mixed precision, and torch.compile. Those choices must be validated on the target backend; they are not guaranteed to work with ROCm unchanged.

### 4.3 Dataset/context

| Dataset | Task | Base tokenizer | Context/prefix | Target length |
|---|---|---|---|---:|
| OpenWebText | Unconditional text | GPT-2 | Null prefix | 1,024 |
| LM1B | Unconditional text | bert-base-uncased | Null prefix | 128 |
| TinyStories | Unconditional text | Custom BPE | Null prefix | 128 |
| XSum | Summarization | GPT-2 | Article, 512 tokens | 48 |
| WMT14 German-English | Translation | Qwen3-0.6B tokenizer | German, 96 tokens | 64 |
| Sudoku | Constraint reasoning | FMLM+ tokenizer | Puzzle prefix | 89 |

The paper trains a separate tokenizer and generator for each dataset. Paper settings are not a code-generation recipe.

## 5. How to prepare and start a short run

### First run: correctness, not quality claims

1. On Linux, install repository requirements and verify the ROCm PyTorch build can see the AMD GPU (`torch.cuda.is_available()` should be true on ROCm PyTorch). The trainer does not install GPU software.
2. Create `data/mycode/train.jsonl` and `data/mycode/validation.jsonl`. Each line must be JSON with `text` (tokenizer learning) and `prompt`/`target` (conditional code generation). Example: `{"text":"Write an add function.\\n\\ndef add(a, b): return a + b","prompt":"Write an add function.","target":"def add(a, b):\\n    return a + b"}`.
3. Prepare tokenizer train and validation caches, then train a brief tokenizer pilot:

       python -m training.prepare_data --config config/train/code-starter-tokenizer.yaml --stage tokenizer
       python -m training.prepare_data --config config/train/code-starter-tokenizer.yaml --stage tokenizer --input data/mycode/validation.jsonl --cache-dir data/cache/code-starter/tokenizer-validation
       python -m training.train_tokenizer --config config/train/code-starter-tokenizer.yaml --max-steps 20

4. The tokenizer artifact is `artifacts/code-starter/tokenizer.pt`; resumable files are in `runs/code-starter/tokenizer/checkpoints/`. Inspect held-out reconstruction before spending more time.
5. Prepare generator train/validation caches with the trained frozen tokenizer, then train a brief generator pilot:

       python -m training.prepare_data --config config/train/code-starter.yaml --stage generator
       python -m training.prepare_data --config config/train/code-starter.yaml --stage generator --input data/mycode/validation.jsonl --cache-dir data/cache/code-starter/generator-validation
       python -m training.train_generator --config config/train/code-starter.yaml --max-steps 20

6. The generator artifact is `artifacts/code-starter/generator.pt`. Validation losses are recorded separately; code quality still needs compilation/unit-test/functional evaluation.
7. Increase budgets only after finite losses, codebook use, sensible validation behavior, loadable outputs, and measurable held-out task gains.

### Normal two-stage order

1. Freeze the dataset format, base tokenizer, context/target lengths, level schedule, and task metric.
2. Train the tokenizer. Track reconstruction, partial-level reconstructions, codebook utilization/dead codes, loss, speed, and peak VRAM.
3. Select and freeze its checkpoint; record a content hash.
4. Encode/cache train and validation sequences with that exact tokenizer.
5. Train the generator against those labels. Track per-level loss, corruption settings, validation outputs, and task metrics.
6. Evaluate with task-appropriate checks. For code, include parsing/compilation, unit tests, and functional task completion; language-model loss alone is insufficient.
7. Package the pair and full manifest as one versioned artifact.

## 6. Commands and outputs

Resume from the latest full state with:

    python -m training.train_generator --config config/train/code-starter.yaml --resume runs/code-starter/generator/checkpoints/latest.pt --max-steps 1000

Run paths are defined in YAML. Outputs are:

    runs/<task>/<stage>/<run-id>/
      config.resolved.yaml
      manifest.json                 # stage, config source, time, run directory
      logs/train.jsonl              # step, losses, LR, examples/sec, VRAM
      logs/validation.jsonl         # held-out losses when configured
      checkpoints/step-00000100.pt  # model + optimizer/scaler + RNG + step
      checkpoints/latest.pt
      checkpoints/best.pt           # lowest validation loss when configured
      checkpoints/final.pt
      samples/                      # decoded samples/previews

Exported eval/deployment bundle:

    artifacts/<task>/<version>/
      config.yaml                    # inference config
      training.config.yaml           # resolved trainer config
      tokenizer.pt
      generator.pt
      manifest.json

Use cache paths keyed by dataset version and tokenizer hash. Keep datasets, caches, logs, and checkpoints out of Git unless intentionally publishing small fixtures. HF tokenizer revisions are currently resolved by `model_id`, not pinned revision. The schedule is reconstructed from saved step and config; optimizer, scaler, and RNG state are checkpointed. The CLI supports config-defined run directories, resume, max steps, and device override. Dataset selection occurs when preparing explicit cache inputs. Distributed training, compilation, and activation checkpointing are not implemented yet.

## 7. Focusing on code and changing the task later

### Code-focused data contract

- Choose a code-appropriate base tokenizer that preserves whitespace, indentation, punctuation, Unicode, and long identifiers. TinyStories BPE is not automatically suitable.
- Use high-quality code data. For prompt-conditioned generation, create prompt/specification → target-code pairs. Other formats such as code completion or source-to-source transformation are different tasks.
- Split by repository/project, not random files, to reduce near-duplicate leakage.
- Specify language markers, file boundaries, EOS/PAD behavior, truncation/packing, and context/target lengths.
- Record dataset provenance, licenses, filters, snapshots, and contamination checks.
- Evaluate with parser/compiler, unit tests, and functional task suites. Record pass rates and regressions.

NCM learns patterns in the chosen training representation. It does not acquire a compiler, editor, shell, asset generator, game engine, or test harness; Hermes and tools provide those.

### Fine-tune versus new model

- If a compatible generator exists and the base tokenizer/representation is unchanged, fine-tune a copy of that generator using a separate run directory and held-out code benchmark.
- If no compatible generator exists, train a generator from scratch on the frozen tokenizer’s codes.
- If tokenization/code structure changes substantially, train a new tokenizer and then a matching generator. Never mix arbitrary tokenizer and generator checkpoints.
- To preserve general behavior, keep the parent checkpoint. Compare a separate specialist generator against continued training on a measured broad+task data mix. Do not overwrite the general model.
- For a later task, create a new task config/run. Reuse a tokenizer only after verifying identical token IDs and acceptable reconstruction on the new domain.

This repo does not document instruction fine-tuning, adapters, preference training, or RL. Establish supervised next-concept training and task evaluation first.

## 8. Efficiency tuning

Profile first: examples/tokens per second, step time, peak VRAM, CPU/RAM use, dataloader wait, checkpoint time, and validation quality. Change one variable at a time and save the resolved config.

| Symptom | First things to try | Caution |
|---|---|---|
| Out of VRAM | Reduce local batch; use gradient accumulation to preserve effective batch; shorten context for smoke tests | Accumulation does not remove a single example's activation memory; shorter context changes the task. |
| GPU idle | Cache token IDs/code IDs; profile preprocessing and data loading; then tune workers | Cache must record exact tokenizer checkpoint hash. |
| Slow steps | Profile; benchmark BF16; try compile only after eager mode works; measure attention cost | Compilation overhead can dominate short runs; paper inference speed is not training speed. |
| Model too large | Reduce layers/width for a pilot | Changes capacity; compare on the same data and metric. |
| Run too long | Use fixed short budgets and validation cadence; select a predeclared metric for stopping | Fewer updates can underfit; compare learning curves, not training loss only. |
| Disk fills | Limit checkpoint cadence and retain latest/best plus a known-good resume point | Verify restore before pruning. |
| ROCm/backend errors | Validate one-batch forward/backward on the actual Linux install; retain an eager fallback | Current evaluator contains CUDA-specific calls. |

For the 16 GB RX 9060 XT, actual batch/model limits must be measured on the Linux PC. Do not treat paper global batch as per-GPU batch. Mixed precision, compilation, activation checkpointing, and multi-GPU should be introduced separately after basic save/resume works.

## 9. Config fields and extensions

The starter uses separate per-stage configs. For a new experiment, copy them into a task-named pair (for example `config/train/mytask/tokenizer.yaml` and `generator.yaml`) and record:

- Task, stage, seed, run ID, output directory.
- Dataset revision, split, preprocessing, tokenizer ID/revision/vocab/merges, EOS/PAD.
- Context/target lengths, packing/truncation, dataloader settings.
- VQ-VAE architecture, levels, codebook sizes, quantizer/EMA, nested dropout, reconstruction/VQ coefficients.
- Generator architecture, prefix layout, attention pattern, heads, corruption schedule, alpha/level weighting.
- Optimizer, scheduler, batch, gradient accumulation, clipping, precision, validation/checkpoint cadence.
- Repo SHA, hardware/backend, library versions, data and tokenizer hashes.

Save a resolved config after overrides so a run can be reproduced exactly.

## 10. Project experiment ladder

1. Check upstream again for official training code and compare it with this implementation.
2. Verify one update/save/resume on the target Linux ROCm setup, then test a tiny overfit.
3. Use a small reference task (TinyStories or Sudoku) to verify the published mechanics. Sudoku verifies structured constraint output; it does not validate code generation.
4. Define a licensed code dataset and functional benchmark before training the code specialist.
5. Train/evaluate on held-out projects; iterate on measured failures.
6. Integrate the trained generator behind Hermes without changing existing profiles; keep build/test tools separate.
7. Add asset generation, playtesting, and visual review around that coding specialist.

One update proves the software runs. Tiny overfit proves it can memorize the format. Neither establishes useful code quality.

## 11. Sources

- [NCM paper (HTML)](https://arxiv.org/html/2610.14162): Sections 3.3–3.4 for objectives/algorithms; Appendix B/C for datasets and reference settings; Appendix F for loss-weighting and corruption ablations.
- [NCM paper (PDF)](https://arxiv.org/pdf/2610.14162)
- [Official HazyResearch/ncm repository](https://github.com/HazyResearch/ncm)
- [Upstream issue #10 asking for training code](https://github.com/HazyResearch/ncm/issues/10): collaborator answered “Yes”; no linked training PR/branch at the check date.
- [Upstream issue #2 requesting training code](https://github.com/HazyResearch/ncm/issues/2)
- [Released checkpoints/configs/datasets collection](https://huggingface.co/collections/hazyresearch/ncm)

## 12. Long-run checklist

- [x] A minimal trainer exists with commands documented in this guide.
- [ ] CLI and tensor shapes are verified on the target Linux/ROCm setup.
- [ ] Data/tokenizer/model compatibility and hashes are recorded.
- [ ] One update runs on the target Linux/ROCm setup.
- [ ] Tiny overfit, validation sampling, save/reload/resume all work.
- [ ] Peak VRAM, step time, throughput, and disk growth are measured.
- [ ] Task benchmark and success threshold are fixed before tuning.
- [ ] Run directory and stop/resume plan are explicit.
