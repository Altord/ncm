# Fewer VQ positions with accurate code reconstruction

Research reviewed 2026-11-12. This is a research and implementation plan, not a verified solution or a final checkpoint. Project measurements are in [the codec experiment history](NCM_CODEC_EXPERIMENT_HISTORY.md).

## The actual NCM bottleneck

The current context-256, 64-level configuration uses 8,194 discrete VQ indices per chunk, including short padded snippets. With 512 choices per index, fixed-width packing takes 9 bits per index before length headers and exact-reconstruction corrections. On the 48 short development snippets, VQ codes and corrections measured 83.30 times the original UTF-8 size. This is an observed storage measurement, not a measurement of NCM generator latency. The generator in `models/next_concept.py` derives its sequence layout from the sum of the configured VQ levels; its actual compute and quality must be measured.

The project has already measured the difficult tradeoff: the 64-level frozen-checkpoint prefix scan lost accuracy rapidly when levels were removed; separately trained 8- and 16-level variants reduced size but were worse at raw compile reconstruction. Those models and data paths are not identical, and frozen-prefix results do not establish how a newly trained compact architecture would perform. The leading 64-level development stress result is 28/48 compilable snippets; it has not demonstrated general code correctness or exact recovery without an oracle correction stream.

## Research and implications

### Compact latent plans plus a conditional code decoder

[Latent Programmer](https://proceedings.mlr.press/v139/hong21a/hong21a.pdf) trains discrete program encodings, predicts latent plans from a specification, then decodes programs conditioned on both the plan and the specification. Its latent sequence is shortened relative to program length; the paper also trains on predicted/noisy latent plans and uses search. It reports improvements on its string-program and natural-language synthesis tasks. Those are not results on our C# dataset, do not imply 99% exact reconstruction, and required substantial training data.

Relevance to NCM: test a short latent representation with a code decoder that can also use information actually available at inference, such as the user's specification, project context, and previously emitted tokens. Our current VQ decoder in `models/vae.py` reconstructs token logits from the VQ representation and is not conditioned on an autoregressive sequence of earlier output tokens or an input specification. Such conditioning may preserve repeated identifiers more effectively, but can introduce sequential generation latency, error propagation and latent bypass; teacher-forced loss is not a substitute for free-running exact-match and behavior tests. Code that is not present in the specification cannot be copied from it.

### Adaptive rate rather than a fixed full pyramid

[Variable Bitrate Residual Vector Quantization](https://arxiv.org/abs/2410.06016) allocates different numbers of quantizers per audio frame and optimizes a rate/fidelity tradeoff. This is evidence for adaptive allocation in audio, not for exact code reconstruction. NCM could investigate variable latent budgets based on chunk length or a learned decision, with explicit length/stop metadata and a decoder and generator trained on those choices. Each emitted code, selector, length and position must be counted; fewer time positions is not necessarily fewer total discrete indices when several codebooks are emitted per position.

NCM currently fixes the quantizer levels and assumes their lengths in `models/quant.py` and `models/next_concept.py`; simply truncating padding or discarding existing indices is not a valid general solution. The current fixed-shape training caches and batching also need changes for truly dynamic per-example layouts. As a lower-risk first step, separately trained fixed short- and long-context models could be benchmarked, but their routing, metadata, padding and separate model costs must be counted. The prior 64-level prefix scan is evidence against assuming the frozen checkpoint can simply drop most levels.

### Improving the quantizer itself

[Restructuring Vector Quantization with the Rotation Trick](https://arxiv.org/abs/2410.06424) reports better reconstruction and utilization in its evaluated VQ-VAE training settings. [Finite Scalar Quantization](https://openreview.net/pdf?id=8ishA3LxN8) proposes a different discrete bottleneck that avoids learned-codebook collapse in its studied settings. These are research candidates, not evidence that either will improve exact C# output. They change training or the representation and require matched code-rate experiments; total discrete symbols and bit rate still matter. NCM's logged utilization counts are cumulative slots ever used, so 100% reported utilization does not establish balanced use, high entropy or meaningful representations.

### Compress indices only after measuring usable representations

[Language Modeling Is Compression](https://arxiv.org/abs/2309.10668) discusses using predictive models with lossless entropy coding. A later CPU-friendly baseline could estimate index frequency distributions using training data only, then actually encode and decode held-out code streams with a deterministic coder. Report all headers, model tables, correction bytes and encode/decode times, and verify exact byte round trips. Adaptive or learned probabilities need identical reproducible coder state at both ends. Entropy-coding preserves the indices but does not reduce the number of VQ positions presented to an unchanged generator or fix a lossy decoder. Current per-chunk zlib results are a necessary known-input baseline, not a generator-quality comparison.

## Recommended sequence

Prototype prepared: `scripts/run_compact_autoregressive_codec.sh` uses a fixed 64-index bottleneck, 4x encoder downsampling, four encoder and decoder blocks, causal decoding conditioned on earlier emitted tokens, AST-aligned data, occurrence weighting, BF16, and no fine dropout. It trains for 5,000 updates before free-running validation and stress compilation. It does not yet implement specification conditioning, variable allocation, a pointer mechanism, or entropy coding. The training validation is teacher-forced and selects the checkpoint on that basis; do not compare that number directly with prior free-running metrics. The separate codec evaluation runs greedily without original target tokens and reports actual reconstruction. It is an extreme rate-reduction screen, not a guaranteed optimum; the 64-index layout changes multiple levers and cannot attribute effects to a single lever. Results are pending.

1. Keep the current 64-level no-drop checkpoint as an accuracy reference and finish the already-running identifier-weight experiment; do not launch another GPU job concurrently.
2. Build an independently testable compact latent + conditional decoder prototype, starting with a single fixed position budget and the same C# source-file split. Track free-running exact tokens, raw compilation and identifier consistency alongside teacher-forced loss. Avoid using original target tokens or their oracle corrections as inference input.
3. Compare with the current 64-level and previously trained 8-/16-level codecs at matched input and update budgets. Report total VQ indices, rate with/without exact sidecar, reconstruction and compile outcomes, VRAM and measured runtime. Use an additional file/template-held-out evaluation set because the 48 development snippets have been repeatedly inspected.
4. If there is a worthwhile accuracy/position tradeoff, investigate length buckets or learned variable allocation, then measure a real index encoder and decoder. Do not claim that theoretical entropy estimates are realized compressed sizes, or that lossless storage results prove generated-code correctness.

Feasibility: these approaches can be implemented in principle. This research does not prove that NCM can simultaneously achieve fewer positions, 99% or 100% token accuracy, all snippets compiling, or faster code generation on the current GPU. If compact candidates cannot meet measured goals at an acceptable budget, retain the established 64-level accuracy reference and continue with the practical staged evaluation plan.
