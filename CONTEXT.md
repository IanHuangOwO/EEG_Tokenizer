# EEG Tokenizer (MeFSQ)

An EEG signal tokenizer: converts multi-channel EEG time series into discrete codes via masked-reconstruction pretraining, using Multi-head Finite Scalar Quantization (MeFSQ).

## Language

**Trial**:
A raw, labeled recording segment as it comes from the source dataset (variable length).
_Avoid_: Segment, recording, epoch (for this meaning)

**Window**:
A fixed-length chunk of signal (`window_length` samples) produced by flattening a subject's trials into one continuous stream and re-cutting it (`assemble_trials=True`). Distinct from a Trial — a Window has no direct 1:1 label relationship to the original trial(s) it was cut from.
_Avoid_: Trial (for this meaning), chunk

## Quantization (MeFSQ)

**Patch**:
One channel's raw `patch_len`-sample time slice within a Window — the smallest unit of signal before embedding, stepped by `patch_stride` (equal to `patch_len` for non-overlapping patches, smaller to overlap consecutive patches — see `IO/preprocessing.py`'s `slice_patches`).

**Token** (_retired_):
Former fused, cross-channel representation of one patch position: all C channels' patch embeddings concatenated into one C*D vector, shared identically by the Router and every Expert. Retired because concatenation bakes channel *position* into a fixed slot, which breaks under cross-dataset channel-count/order variance and zero-padded channels (see `docs/adr/0002-per-expert-channel-attention.md`). Superseded by **Expert View**.

**Expert View**:
The per-Expert, content-based-attention-pooled summary of one patch position's C channel embeddings: each Expert (Routed or Shared pool) has its own learnable query attending over the C per-channel D-dim vectors (keys), producing one D-wide weighted sum specific to that Expert. Replaces the shared, concatenation-based Token — different Experts may weight channels differently for the same patch. Padded/invalid channels (`valid_channels` mask) get their attention score set to −inf before softmax, so channel count/order can vary across datasets without corrupting the pooled result. The Router scores each Expert on that same Expert's own View (not a separate shared input), so the gating decision and the quantized input are always the same vector.
_Avoid_: Token (for this meaning, retired)

**Expert**:
An independent FSQ quantizer unit (down-proj -> quantize -> up-proj -> decode) living in a Routed or Shared pool. Canonical term for what earlier docs called a "VQ head".
_Avoid_: Head, VQ head (retired; "head" now means attention head only, e.g. spatial_heads, PerChannelHeadAttn)

**Routed pool**:
A pool of Experts where a Router top-k gates which Experts fire per patch, weighted-summed over the selected ones. Each Expert is scored against its own Expert View. Buys representation specialization (every Expert still densely computed, unselected ones masked to zero — not a compute-saving sparse dispatch at this scale).

**Shared pool**:
A pool of Experts that are always active for every patch, summed and down-weighted, providing a constant baseline contribution to reconstruction alongside the Routed pool's specialized contribution. Each Expert still has its own Expert View — "always active" means ungated, not that they share one pooled input.

**Code**:
The discrete FSQ output an Expert assigns to its own Expert View (per-head sigmoid quantization + straight-through estimator, `num_discrete` groups per Expert).

**Codebook**:
The space of possible Codes an Expert can assign (size governed by `r` / `vq_head_vocab_size`).

**Pre-VQ feature**:
The continuous, unquantized per-channel vector produced by the encoder before the pooling+quantize step. Broadcast to every Expert independently (unlike the Expert View, where each Expert forms its own channel-pooled vector). Exposed via `encode_pre_vq` for diagnostics only — no longer read by finetune (see Tokenizer bypass, retired).

**Tokenizer bypass** (_retired_):
Former finetune mode that read Pre-VQ features directly, skipping the discrete Code round-trip. Removed: every Expert broadcasts the identical Pre-VQ vector, so the finetune head had no per-Expert signal to differentiate and its attention pooling collapsed to uniform. Finetune now always reads `encode_post_vq` (see below).

**Post-VQ feature**:
Per-Expert, per-channel signal read AFTER quantization: each Expert's own decoded reconstruction, split back out per channel, before the cross-Expert sum. Genuinely Expert-differentiated (unlike Pre-VQ, where every Expert sees the same broadcast vector) since each Expert quantizes to its own Code and decodes with its own weights. Exposed via `encode_post_vq`; this is what finetune reads.

## Sparse tokenization (MeSAE)

A parallel, non-discrete tokenizer approach (`model/MeSAE/`) — goal is explainable per-patch sparse features (ICA-style linear sum of independent source contributions), not a discrete Code vocabulary. The unit is a **Stamp**, not an Expert/Code; see `model/MeSAE/MeSAE.py` and `StampBank` in `MeSAE_modules.py`.

**Stamp**: A learned unit-norm temporal template `D` of length `patch_len`, plus its derived Hilbert quadrature partner `H` (rFFT, every positive-frequency bin rotated -90 degrees, DC/Nyquist zeroed, renormalized). A stamp contributes `a*D + b*H` to a channel, so amplitude is `sqrt(a^2+b^2)` and phase `atan2(b, a)` — the template presents at any arrival phase without its shape morphing, because the partner is derived rather than learned.

**Mixing column**: A stamp's `[C]` vector of signed per-channel gains at one patch position — the ICA-style topography of that stamp's contribution at that instant.

**Group selection**: `StampBank` takes channel-grouped input `[G, C, D]` (G = B*N patch positions) and picks ONE top-k stamp set per patch position, shared by all C channels; each channel then decodes that same set with its own gains. This is what makes a stamp's `[C]` column a mixing vector instead of C unrelated per-channel choices.

**Tokenizer stage**:
MeSAE's first training phase: the encoder (kept shallow/local, see `docs/adr/0003-mesae-two-stage-masked-training.md`) and the StampBank train jointly, unmasked, full reconstruction only. No masked-patch pretext task at this stage — that's deferred to the Masked stage.

**Masked stage**:
MeSAE's second training phase: the Tokenizer-stage StampBank is frozen (weights fixed) and only the backbone trains, on masked input, to reconstruct through it. Mirrors MeFSQ's `freeze_vq_and_decoder()` split but two-stage/sequential rather than joint-warmup-then-freeze — deliberate, to keep the frozen target local rather than already-contextualized (see ADR 0003). `embed_dim` must match the tokenizer stage that produced the checkpoint (the StampBank and patch embedding both consume `z` of that width); `enc_depth`/`pool_after_blocks` may differ freely, and `train_pretrain.py` fails loudly on any other mismatch.

## Model plugin architecture

**Unit**: Umbrella term for whatever a model quantizes per patch position — an Expert (MeFSQ) or a Stamp (MeSAE). Used in shared code (`model/base_checker.py`, `model/base_plotter.py`) that doesn't know which model it's plotting.

Each model (MeFSQ, MeSAE, or a future one) plugs into shared training/viz infrastructure via a `model/<Name>/plugin.py` bundling a `Trainer`/`Checker`/`Plotter` into one `BasePlugin`, registered in `model/factory.py`'s `MODEL_REGISTRY`. See `docs/adr/0004-model-plugin-base-classes.md` and `docs/agents/adding-a-model.md` for the full contract.
