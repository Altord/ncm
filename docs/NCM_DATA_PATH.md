# NCM Data and Generation Path (Repository Reference)

This document traces the path implemented by this checkout of Next-Concept Modeling (NCM), from input text to generated text. It is based on the source code in this fork. The training loops and data preparation now live under `training/`; see [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for the runbook and objective math.

## Executive summary

This implementation has two learned stages:

1. **A tokenizer/codec:** a conventional text tokenizer converts text to integer IDs; a VQ-VAE maps fixed-length spans of those IDs to a hierarchy of discrete codebook indices. The decoder maps accumulated codebook vectors back to distributions over original tokenizer IDs.
2. **A next-concept generator:** a Transformer predicts code indices from coarse levels to finer levels. The generated positions within a level can be predicted together. Once all levels are available, the VQ-VAE decoder converts their accumulated representation back to tokenizer IDs, then the conventional tokenizer converts IDs to text.

“Concept” here refers to learned, discrete multiscale latent codes. The code does not assign them human-readable semantic names.

```mermaid
flowchart LR
  A[Text] --> B[Base tokenizer: BPE / HF / task-specific]
  B --> C[Base token IDs, chunked to fixed context]
  C --> D[VQ-VAE encoder]
  D --> E[Latent sequence z]
  E --> F[Residual multiscale quantizer]
  F --> G[Code indices by level]
  G --> H[Next-concept Transformer]
  H --> I[Predicted code indices, coarse to fine]
  I --> J[Codebook lookup + upsample + sum]
  J --> K[VQ-VAE decoder logits]
  K --> L[Base token IDs]
  L --> M[Base tokenizer decode]
  M --> N[Text]
```

## 1. Vocabulary and terminology

- **Base token ID:** integer assigned to a piece of text by BPE, a Hugging Face tokenizer, or a task-specific tokenizer.
- **Latent vector:** continuous vector emitted by the VQ-VAE encoder at a sequence position.
- **Codebook:** a learned table of vectors. A quantizer replaces a latent vector (or a pooled region of vectors) with the ID of a nearby codebook vector.
- **Level:** one resolution in the hierarchy. Early/coarse levels use fewer positions; later/finer levels use more.
- **Residual:** the part of the latent representation not yet explained by the sum of code vectors chosen at earlier levels.
- **NCM input:** continuous representation derived from already-known coarse codes, used as the Transformer’s input at the next level.
- **NCM target:** the discrete code index that the appropriate output head should predict at a position.

For a batch, use the shapes below:

| Symbol | Shape | Meaning |
|---|---|---|
| `x` | `[B, L]` | Base tokenizer IDs; `B` is batch size and `L` is fixed context length. |
| `z` | `[B, L, C]` | VQ-VAE continuous encoder representation; `C` is embedding width. |
| `levels = [l0, ..., l(K-1)]` | each `lk <= L` | Number of discrete positions at each quantization resolution; commonly the final level has `L` positions. |
| `idx` | `[B, sum(levels)]` | Concatenated discrete code IDs, stored level by level. |
| `ncm_x` | `[B, sum(levels[1:]), C]` when level 0 is the supplied starting level | Continuous, accumulated-coarse-code inputs for predicted levels. |
| `logits` | `[B, sum(levels[1:]), Vmax]` in the ordinary no-level-0-query configuration | Per-position scores for codebook IDs, padded to a common output width. |

`Vmax` is the largest codebook vocabulary among the output heads. For each position, only that level’s actual vocabulary entries are meaningful; `_pad_vocab` fills the rest with the dtype’s minimum finite value.

## 2. Stage A: text to base token IDs

The VQ-VAE does not operate directly on strings. `VQVAETokenizer` first calls its configured `pre_tokenizer`.

### BPE path

`BPETokenizer.encode` in `tokenization/bpe_tokenizer.py`:

1. Splits out configured special-token strings so they are not merged as ordinary text.
2. Applies a regex to split ordinary text into word/number/punctuation/whitespace pieces.
3. Encodes each piece as UTF-8 bytes, initially one byte per symbol.
4. Repeatedly merges the adjacent byte pair with the earliest rank in the configured merge list until no listed pair remains.
5. Maps the resulting byte strings to vocabulary IDs.

The inverse path maps IDs to byte strings, concatenates bytes, and decodes UTF-8.

### Hugging Face path

`HFTokenizer` in `tokenization/hf_tokenizer.py` wraps `transformers.AutoTokenizer`. Its encode/decode methods delegate to that tokenizer. Configuration can set EOS and padding tokens.

### Task-specific path

The registry in `utils/registry.py` also includes task-specific tokenizers (for example, Sudoku). The base tokenizer’s vocabulary and special-token conventions must agree with the VQ-VAE checkpoint/config.

### Fixed-length chunking

`VQVAETokenizer.encode` calls `VQVAETokenizer._adjust_length` in `tokenization/vqvae_tokenizer.py`. It splits IDs into chunks of `vqvae.context_length`; the final short chunk is padded with the base tokenizer’s pad ID. The VQ-VAE therefore receives a batch-shaped integer tensor with fixed width. An empty input is an edge case to handle carefully: `_adjust_length` currently yields no chunks, so callers should not assume empty text follows the ordinary path.

## 3. Stage B: base IDs to a continuous latent sequence

`VQVAE._encode_pre_quant` in `models/vqvae.py` calls `self.encoder(x)`. `Encoder` is implemented in `models/vae.py` and maps integer IDs through learned embeddings and its configured positional/sequence layers to `z` with shape `[B, L, C]`.

An optional normalization and dropout are applied before quantization:

- `pre_quant_norm` can be no normalization, non-affine LayerNorm, or non-affine GroupNorm, depending on configuration.
- `pre_quant_dropout` can drop elements during training if enabled.

The quantizer receives float latent values. It does not see the source string or original programming-language syntax directly; those are represented only through base-token IDs and learned embeddings.

## 4. Stage C: residual multiscale vector quantization

The core implementation is `MultiscaleResidualVectorQuantizer` in `models/quant.py`.

Let the encoder output be `z` and let `r_i` be the residual before level `i`. Initially:

\[
\hat z_{-1} = 0, \qquad r_0 = z.
\]

At level `i`, the quantizer:

1. Reduces the current residual to `l_i` positions using `_down_for_quant` (except at the final level, where the residual is already at full length).
2. Computes the distance from each reduced residual vector to every vector in that level’s codebook.
3. Selects the nearest entry by `argmin` in the deterministic path. The code also has an optional stochastic sampling path.
4. Looks up the selected codebook vectors and upsamples them to the full latent length for non-final levels.
5. Applies the configured residual transform (`quant_resi`, commonly a blend with a convolution).
6. Adds this level’s contribution to the reconstruction and subtracts it from the residual.

In compact form, with `q_i` the transformed, full-length contribution at level `i`:

\[
\hat z_i = \hat z_{i-1} + q_i, \qquad r_{i+1} = z - \hat z_i.
\]

The code indices are concatenated in level order:

\[
\text{idx} = [\text{idx}_0 \Vert \text{idx}_1 \Vert \cdots \Vert \text{idx}_{K-1}],
\]

where `idx_i` has `l_i` entries per example. This is a *single flattened storage tensor*, not one common codebook: each level has its own embeddings and may have its own vocabulary size.

### Distance and code selection

The core `_distances` function compares latent vectors with codebook vectors using squared Euclidean distance. In deterministic encoding, each vector receives the closest codebook ID. This is vector quantization: a continuous value is represented by a discrete symbol plus the shared codebook table.

### Quantizer training behavior present in source

During `MultiscaleResidualVectorQuantizer.forward` in training mode, assignments update per-level exponential moving averages for code counts and vector sums. Optional distributed all-reduce synchronizes those statistics. Dead-code restart, k-means initialization, and stochastic encoding are configuration-dependent features.

The quantizer accumulates a VQ loss at each cumulative level, with a commitment term and a codebook-fit term. Without fine-level dropout it averages these losses; with fine-level dropout it selects the VQ loss at the same sampled cutoff used for reconstruction. The existing implementation updates EMA statistics at every level, even when a cutoff is sampled.

`VQVAE.forward(x)` returns `(decoder_logits, vq_loss, per_level_codebook_utilization, global_codebook_utilization, ppl_value)`. `training.train_tokenizer` calculates the token reconstruction cross-entropy, adds the VQ term, applies padding masks, and performs optimizer updates.

## 5. From known coarse codes to the NCM training input

`VQVAE.get_ncm_inputs(pre_tokenized_ids)` is the explicit bridge between the frozen tokenizer and the NCM generator:

1. `encode` obtains the full level-concatenated code IDs.
2. `vq.idx_to_ncm_input(idx)` reconstructs the accumulated coarse latent at each level, downsamples it to the *next* level’s number of positions, and concatenates those continuous vectors.
3. The method returns that continuous input tensor and `idx[:, levels[0]:]`, i.e. the code IDs after the first/coarsest level.

The first level is excluded from those labels in this API because the normal generator configuration treats it as a supplied seed/prefix code. Alternative configurations can use `query_level0` or `predict_level0`; they change how the first level is produced.

Important distinction: `idx_to_ncm_input` returns **inputs to the generator**, not the integer targets. The target IDs are the corresponding slices of `idx`. At training time the model needs both: context-plus-coarse-code inputs and the correct per-position codebook ID labels.

## 6. Stage D: Next-Concept Transformer

`NextConceptTransformer` in `models/next_concept.py` stores a reference to the frozen VQ-VAE and uses its level sizes, embeddings, and codebooks. The tuple wrapper around `vqvae_local` intentionally keeps the codec out of the generator’s own registered module state; the generator checkpoint is therefore separate from the tokenizer checkpoint.

### Input projection and context

In the ordinary path, `forward(x, ...)` splits the first `prefix_len` vectors from `x` as the prefix representation and treats the remaining concatenated vectors as per-level target inputs. The generator:

- Projects continuous VQ vectors into the Transformer hidden width using `transformer.wte`.
- Adds learned level encodings when configured.
- Adds positional information via RoPE or learned position embeddings.
- Applies the configured attention mask and Transformer blocks.
- Applies a vocabulary output head for each predicted level and concatenates the resulting per-position logits.

There is also an optional context-encoder mode, in which `prefix_ids` are encoded separately. A pretrained T5 encoder can be used by configuration, but TinyStories’ shown config disables that mode.

### Attention and parallelism

The output consists of logits for many positions, but the attention mask controls which positions may condition on one another:

- **`block_diagonal`:** target positions can attend within the same level, but not to other target levels. The prefix is visible to targets. This supports predicting all positions of one level in parallel while generation advances level by level.
- **`block_causal`:** target levels can attend to the same or earlier levels, preserving coarse-to-fine dependency while still allowing parallel prediction within a level.

The model has separate output heads per level unless configured to share them. Heads can optionally be tied to the VQ codebook embeddings. A tied head scores a projected hidden vector by its similarity to codebook vectors (with a learned scale, and optional bias/trainable codebook copy).

`forward` returns logits; `training.train_generator` applies per-level cross-entropy, masks each level to its actual codebook vocabulary, and weights the level means according to alpha. It supplies clean labels and can optionally corrupt conditioning IDs.

## 7. Stage E: coarse-to-fine generation and decoding

`NextConceptTransformer.generate` in `models/next_concept.py` performs iterative level generation:

1. Encode the prefix representation (raw base-token embeddings, VQ-VAE latent prefix, or optional context encoder according to config).
2. Obtain the initial/coarsest code IDs. Depending on configuration, these can be supplied by the caller, predicted with a level-0 query, or predicted by the `predict_level0` path.
3. Look up that level’s codebook vectors, upsample and add them to the accumulated latent, then downsample the accumulated latent to form the next level’s input.
4. Run the Transformer for that next level and sample/argmax its code IDs using temperature, top-k, and top-p settings.
5. Repeat through the final level, then concatenate all predicted/supplied level IDs.

Thus, “parallel” means the multiple positions belonging to a level are emitted by one level computation. The next level still waits for the preceding coarse level. With `K` levels, the coarse-to-fine structure requires roughly one generation pass per generated level (plus any separately generated level 0), not one serial token step per base-token position.

The quantizer’s `idx_to_zhats` / `idx_to_z_q` reconstructs the accumulated latent from the completed codes. The VQ-VAE decoder returns logits over the base-token vocabulary for each original sequence position. Decoding normally takes `argmax` over those logits, then the base tokenizer decodes the resulting IDs into text.

In `scripts/eval_shard.py`, the evaluation path calls `generate`, reconstructs the final latent level, invokes the VQ-VAE decoder, chooses base IDs with `pick_tokens`, removes EOS/PAD as configured, and decodes text. In that evaluator, `pick_tokens` can apply token-frequency and repetition adjustments; this is evaluation-time sampling/post-processing, not evidence of the training objective.

## 8. Concrete config example: TinyStories

`config/eval/tinystories.yaml` demonstrates a concrete released model setup:

- Base tokenizer: repository BPE with configured vocabulary/merges and EOS/PAD tokens.
- VQ-VAE context length: 128 base-token positions; latent width: 384.
- Quantizer levels: `[1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]`.
- The total code count is the sum of those level lengths; it is not the same thing as the 128 base-token positions.
- Generator: 12 Transformer layers, hidden size 768, 12 heads.
- `query_level0: false` and `predict_level0: false`: the evaluator supplies level 0. Its unconditional sampling path uses the configured level-0 unigram distribution when available.
- The generator predicts the later levels, using the configured per-level codebook vocabularies.
- Attention pattern: `block_diagonal`; RoPE and level encoding are enabled.

These values describe this checkpoint’s evaluation configuration, not an established recipe for a code model.

## 9. What is and is not in this checkout

Present:

- Model architecture: `models/next_concept.py`, `models/vqvae.py`, `models/vae.py`, `models/quant.py`.
- Base-tokenizer interfaces and implementations: `tokenization/`.
- Evaluation configs and generation/scoring scripts: `config/eval/`, `evaluate.py`, and `scripts/eval_*.py`.
- Instructions to download released checkpoints and evaluation data: `README.md`.

Still not provided by this checkout:

- Full dataset acquisition, licensing review, cleaning, deduplication, and benchmark curation.
- Exact source preprocessing/data-cleaning details for every paper run.
- A coding-specific quality benchmark, compiler/test evaluation loop, or proven code-generation recipe.
- Distributed training, activation checkpointing, compile integration, and multi-GPU support in the new trainer.

**Important correction from external-source research:** the *math and high-level recipe are not missing from public documentation*. The NCM paper gives the tokenizer objective (Equation 9), nested dropout, the NCM cross-entropy objective and level weights (Equation 10), the conditioning-corruption idea, an algorithm for each training stage, and Appendix C hyperparameter tables. See the sources in Section 11. This fork implements a minimal trainer from those descriptions; it is not the authors' official implementation and does not reproduce every data-pipeline detail.

The trainer is a minimal implementation of the paper objectives, not an official trainer or bit-for-bit reproduction. Conditioning corruption exposes configurable linear start/end rates, defaulting to zero; exact rates for every released run remain unknown. The public GitHub issue asking for training code had a collaborator reply “Yes”; recheck upstream for official code before claiming reproduction.

The README’s evaluator is CUDA-oriented (`.cuda()`, `torch.cuda.device_count()`, `CUDA_VISIBLE_DEVICES`) and recommends a CUDA-capable PyTorch installation. The separate Linux Radeon PC needs a ROCm-compatible installation and a small runtime compatibility check; the README’s command is not proof that this exact checkout runs on that AMD setup unchanged.

## 10. Source map

| Topic | Source |
|---|---|
| Overall two-stage NCM summary and released artifacts | `README.md` |
| BPE and Hugging Face tokenizer wrappers | `tokenization/bpe_tokenizer.py`, `tokenization/hf_tokenizer.py` |
| Chunking, VQ encode/decode, NCM input bridge | `tokenization/vqvae_tokenizer.py`, `models/vqvae.py` |
| Encoder and decoder networks | `models/vae.py` |
| Residual multiscale quantization, code indices, accumulated inputs | `models/quant.py` |
| NCM Transformer setup, masking, output heads, forward and generate | `models/next_concept.py` |
| Example dimensions, level schedule, generation/eval settings | `config/eval/tinystories.yaml` |
| Checkpoint loading, generation, decoding during evaluation | `scripts/eval_shard.py`, `evaluate.py` |

## 11. What the paper gives us (external-source cross-check)

The paper is the primary source for what the authors intended mathematically; the repository is the primary source for what this code actually executes. We should compare both before training.

### Tokenizer training objective

At a randomly selected nested-dropout cutoff `k`, reconstruct the original base-token sequence from the cumulative quantized representation `z_hat^(k)`. The published objective is:

\[
L_tok = -\frac{1}{L}\sum_i \log p_\psi(x^i\mid\hat z^{(k)})
       + \beta\,\mathrm{MSE}(z,\mathrm{sg}[\hat z^{(k)}])
       + \mathrm{MSE}(\mathrm{sg}[z],\hat z^{(k)}).
\]

The first term is token reconstruction cross-entropy. The two VQ terms are commitment and latent quantization MSE terms with stop-gradient (`sg`) placed on opposite sides. A straight-through estimator carries decoder reconstruction gradients across the discrete quantization operation to the encoder. Nested dropout means later/finer levels are sometimes removed, so early levels must support useful partial reconstructions.

### Generator training objective

Freeze the tokenizer, encode a batch into correct level codes `r_1:K`, optionally corrupt some *conditioning inputs* by replacing codes with random entries from the same level’s codebook, and keep the ground-truth labels clean. Build each level’s input from the cumulative reconstruction of the corrupted codes. Run the generator once over all level positions and compute weighted code cross-entropy:

\[
L_NCM = -\sum_k w_k\sum_{i=1}^{\ell_k}\log p_\theta(r_k^i\mid\tilde z^{(k-1)}),
\qquad
w_k=\frac{\ell_k^{-\alpha}}{\sum_j\ell_j^{1-\alpha}}.
\]

`alpha=0` gives equal weight per code position; `alpha=1` gives equal total weight per level. The local trainer arranges the prefix/target tensors according to the active `query_level0` and `predict_level0` configuration. It still needs runtime shape validation on the target backend.

### Published TinyStories reference settings

These are paper-reported settings for the authors’ full experiment, **not a recommended run size for our single 16 GB GPU**:

| Stage | Updates | Global batch | Peak → floor LR | Warmup | AdamW betas | Weight decay | Other |
|---|---:|---:|---:|---:|---|---:|---|
| Tokenizer | 30,000 | 4,096 | 5e-4 → 5e-5 | 1,000 | 0.9 / 0.95 | 0.01 | EMA 0.95; nested dropout 0.5; grad clip 1 |
| NCM generator | 50,000 | 2,048 | 3e-4 → 3e-5 | 500 | 0.9 / 0.95 | 0.1 | grad clip 1 |

The paper reports seed 42, BF16 mixed precision, and `torch.compile` across experiments. Appendix C has the architecture and training tables for each task. A TinyStories code-corruption/loss-weighting ablation separately ran 10,000 updates with batch 1,024; do not mistake that ablation budget for the full TinyStories training recipe.

### Where to look and what each source can answer

1. **Paper, Sections 3.3–3.4:** architecture equations, tokenizer loss, nested dropout, generator loss, corruption concept, and training pseudocode.
2. **Paper, Appendix B/C:** datasets, architecture, updates, global batch sizes, learning rates, optimizer betas, weight decay, gradient clipping, and EMA settings.
3. **Paper, Appendix F:** loss-weighting and code-corruption ablations and their effects.
4. **Upstream GitHub repository and Issues/PRs:** actual implementation, future released training scripts, data preparation, and discrepancies between code/config and paper.
5. **Released checkpoints/configs/datasets:** infer preprocessing formats and verify checkpoint compatibility, but these are not training source code.

### References

- [NCM paper (arXiv HTML)](https://arxiv.org/html/2610.14162)
- [NCM paper PDF](https://arxiv.org/pdf/2610.14162)
- [Official HazyResearch/ncm repository](https://github.com/HazyResearch/ncm)
- [Upstream issue #10: “There isn't any code for the training”](https://github.com/HazyResearch/ncm/issues/10) — collaborator replied “Yes” on Oct. 21, 2026; check issue/PR links for subsequent release.
- [Upstream issue #2: request for training codes](https://github.com/HazyResearch/ncm/issues/2)
- [NCM released checkpoints and datasets](https://huggingface.co/collections/hazyresearch/ncm)

## 12. Questions to settle before a long run

1. Has upstream released a trainer, and how does it differ from this local implementation?
2. What exact corruption probabilities and level-loss weighting (`alpha`) were used for the released generator runs?
3. Which dataset version, filtering, packing, and special-token handling were used for the experiments and would be appropriate for code?
4. Do the local nested-dropout and conditioning-corruption paths match the paper's intended behavior and defaults for each reported run?
5. Can the checkpoint/config/code path run on the Linux machine’s exact ROCm/PyTorch versions, and what are measured peak VRAM and throughput?
6. After those checks, can we reproduce one tiny batch/update numerically and inspect gradients, then compare a short run against a held-out metric?

The paper answers the objective-level math and gives experiment-scale settings. These remaining questions are about exact code/data fidelity and adapting the recipe responsibly before longer training.



