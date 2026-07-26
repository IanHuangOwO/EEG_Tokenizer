# EEG Tokenizer (MeFSQ)

An EEG signal tokenizer: converts multi-channel EEG time series into discrete codes via masked-reconstruction pretraining, using Multi-head Finite Scalar Quantization (MeFSQ).

## Language

**Trial**:
A raw, labeled recording segment as it comes from the source dataset (variable length).
_Avoid_: Segment, recording, epoch (for this meaning)

**Window**:
A fixed-length chunk of signal (`trial_length` samples) produced by flattening a subject's trials into one continuous stream and re-cutting it (`assemble_trials=True`). Distinct from a Trial — a Window has no direct 1:1 label relationship to the original trial(s) it was cut from.
_Avoid_: Trial (for this meaning), chunk

## Quantization (MeFSQ)

**Patch**:
One channel's raw `patch_len`-sample time slice within a Window — the smallest unit of signal before embedding.

**Token**:
The fused, cross-channel representation of one patch position: all C channels' patch embeddings (each dim D) concatenated into one C*D vector. This is the unit the Router and Experts operate on — one shared Code must explain all channels jointly.
_Avoid_: Patch (for this meaning)

**Expert**:
An independent FSQ quantizer unit (down-proj -> quantize -> up-proj -> decode) living in a Routed or Shared pool. Canonical term for what earlier docs called a "VQ head".
_Avoid_: Head, VQ head (retired; "head" now means attention head only, e.g. spatial_heads, PerChannelHeadAttn)

**Routed pool**:
A pool of Experts where a Router top-k gates which Experts fire per Token, weighted-summed over the selected ones. Buys representation specialization (every Expert still densely computed, unselected ones masked to zero — not a compute-saving sparse dispatch at this scale).

**Shared pool**:
A pool of Experts that are always active for every Token, summed and down-weighted, providing a constant baseline contribution to reconstruction alongside the Routed pool's specialized contribution.

**Code**:
The discrete FSQ output an Expert assigns to a Token (per-head sigmoid quantization + straight-through estimator, `num_discrete` groups per Expert).

**Codebook**:
The space of possible Codes an Expert can assign (size governed by `r` / `vq_head_vocab_size`).

**Pre-VQ feature**:
The continuous, unquantized per-channel vector produced by the encoder before the concat+quantize step. Broadcast to every Expert independently (unlike the Token, which concatenates all channels). Exposed via `encode_pre_vq` for diagnostics only — no longer read by finetune (see Tokenizer bypass, retired).

**Tokenizer bypass** (_retired_):
Former finetune mode that read Pre-VQ features directly, skipping the discrete Code round-trip. Removed: every Expert broadcasts the identical Pre-VQ vector, so the finetune head had no per-Expert signal to differentiate and its attention pooling collapsed to uniform. Finetune now always reads `encode_post_vq` (see below).

**Post-VQ feature**:
Per-Expert, per-channel signal read AFTER quantization: each Expert's own decoded reconstruction, split back out per channel, before the cross-Expert sum. Genuinely Expert-differentiated (unlike Pre-VQ, where every Expert sees the same broadcast vector) since each Expert quantizes to its own Code and decodes with its own weights. Exposed via `encode_post_vq`; this is what finetune reads.
