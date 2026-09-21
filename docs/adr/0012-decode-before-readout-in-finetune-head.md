# 0012 — Decode before readout in the finetune head

Status: Accepted (design direction), pending implementation
Date: 2026-09-17
Model under test: `mesae_tokenizer_v5` (`output/pretrain/mesae_tokenizer_v5/tokenizer/best_tokenizer.pth`)
Probe dataset: BCICIV2a, 2592 trials, 9 subjects, 4 classes, chance 0.250
Secondary: EEGMMIdb, 39569 trials, 109 subjects, k=3 subset, chance 0.333

## Context

Finetune linear probing sat near chance (~0.28–0.34) while tokenizer reconstruction was
excellent (`mse_patch` ~0.0275, stitched-trial MSE 0.0167). That combination is not
self-consistent: a reconstruction that good cannot have thrown away the class signal.
Two possibilities — a pipeline/label bug, or a readout failure in the head.

This ADR records the full diagnostic series run to separate those, including four
hypotheses that were **refuted by measurement**.

All probes use per-subject LDA with 5-fold stratified CV (no cross-subject pooling, so
no subject-identity leakage), computed on frozen tokenizer outputs. Scripts live in the
session scratchpad: `diagnose_pipeline.py`, `recon_lda.py`, `stamp_lda.py`,
`temporal_bins.py`, `quadratic_test.py`, `feature_*.py`.

## 1. The data path is sound

`diagnose_pipeline.py`, no model involved:

| check | result |
|---|---|
| label balance | 648/648/648/648, 72 per class per subject |
| cache vs loader label sequence | byte-identical (`True`) |
| per-subject mu+beta log-power LDA | **0.513** mean (0.385–0.674 per subject) |

0.513 is squarely in the published BCICIV2a within-subject range. **No data/label
misalignment.** The earlier 0.27–0.30 numbers came from cross-subject pooling, not from
a broken loader.

Caveat: the C3/C4 grand-average lateralization check was inconclusive — all four classes
showed the same-sign `C3-C4` (−0.21 to −0.33). Grand averages over a full trial window
mask ERD contrast; the LDA result is the stronger evidence and it is positive.

## 2. The reconstruction preserves the class information

`recon_lda.py` — identical feature extractor (mu/beta log band power per channel) applied
to three signals:

| signal | S1..S9 | mean |
|---|---|---|
| raw | .56 .44 .59 .44 .39 .40 .58 .67 .56 | **0.513** |
| recon | .56 .42 .62 .46 .39 .39 .57 .67 .57 | **0.516** |
| residual (raw − recon) | .30 .25 .34 .25 .24 .26 .30 .31 .30 | 0.282 |
| chance | | 0.250 |

`recon ≈ raw` per subject, and the residual is near chance. **The bottleneck loses
nothing that matters for MI classification.** This settles it: the failure is in the
readout, not the representation.

## 3. The information ladder — where it is actually lost

`stamp_lda.py`, same LDA, walking from the reconstruction down to what the finetune head
actually receives:

| representation | mean acc |
|---|---|
| recon band power | 0.516 |
| `chan_mag` — per-channel stamp magnitudes `sqrt(a²+b²)` | 0.404 |
| `pool_mag` — after channel-attention pooling | 0.316 |
| `head_z` — what `MeSAEFinetune` actually sees | **0.279** |
| `chan_ab` — raw signed (a,b) | 0.270 |
| chance | 0.250 |

Two distinct losses:
- **recon → chan_mag**: reducing the code to magnitudes costs ~0.11.
- **chan_mag → head_z**: the channel-attention pool (`encode_post_stamp_expert`'s
  `softmax` over channels, then `einsum`) costs another ~0.12, taking it to near chance.

The second loss is structural: a softmax-weighted convex combination over channels
cannot represent a *contrast* between channels, which is exactly what MI decoding needs.

## 4. Four refuted hypotheses for the recon → chan_mag gap

### 4a. Spatial (topography) — refuted

Added `use_topo_feature` to `MeSAEFinetune` (per-channel projection into the head).
`feature_combo.py` / `feature_ablation.py` on EEGMMIdb:

| features | width | PCA64 |
|---|---|---|
| pooled mag (current) | 21 | 0.502 |
| chan_attn topo only | 1344 | 0.462 |
| per-chan mag (topo × scale) | 1344 | 0.468 |
| topo + raw power | 1408 | 0.497 |
| topo + bandpow | 1664 | 0.510 |
| topo + bandpow + cov | 3744 | 0.490 |

Explicit topography gives **no reliable gain** over the 21-dim pooled magnitude. The
best combination (0.510) is inside noise of the simplest baseline (0.502).

`feature_subjectwise.py` additionally showed that wide features carry a subject-identity
component: going trial-wise → subject-wise CV costs −0.069 for `cov` and −0.087 for
`topo+bandpow+cov`, versus −0.005 for the 21-dim `pool_mag`. Wide feature gains were
partly leakage, not signal.

### 4b. Temporal averaging — refuted

`temporal_bins.py` swept the number of time bins the stamp magnitudes are averaged into:

| features | width | PCA32 | PCA64 |
|---|---|---|---|
| chan_mag 1 bin | 1344 | 0.390 | **0.418** |
| chan_mag 2 bin | 2688 | 0.354 | 0.412 |
| chan_mag 4 bin | 5376 | 0.350 | 0.367 |
| chan_mag 8 bin | 10752 | 0.328 | 0.352 |
| raw bandpow | 128 | 0.494 | 0.482 |

More temporal resolution is **monotonically worse**. Temporal averaging is not the loss.

### 4c. Quadratic / cross-term structure — refuted

Band power of the reconstruction is `|Σᵢ aᵢDᵢ|² = Σᵢⱼ aᵢaⱼ⟨Dᵢ,Dⱼ⟩` — a quadratic form.
A linear readout over magnitudes cannot express cross-terms, so the hypothesis was that
second-order code features close the gap. `quadratic_test.py`:

| features | width | acc |
|---|---|---|
| chan_mag (linear, code) | 1344 | 0.418 |
| **codecov (second-order, code)** | 903 | **0.265** ← chance |
| chan_mag + codecov | 2247 | 0.330 ← *worse* than chan_mag alone |
| recon_cov (second-order, recon) | 2080 | 0.449 |
| recon_bandpow (recon) | 128 | 0.471 |

`codecov` is the exact second-order statistic of the signed coefficients and it is at
chance. Adding it *degrades* `chan_mag`. Refuted.

### 4d. Part of the gap was a measurement artifact

The 0.516 reference used 128 features with plain LDA; `chan_mag` used 1344 features
through PCA64 + shrinkage LDA. At **matched** preprocessing the comparison is
`chan_mag` 0.418 vs `recon_bandpow` 0.471 — a **0.053** gap, not the 0.112 originally
chased. Always compare within one preprocessing pipeline.

## 5. What survived

One ordering held across every probe, every feature family (band power, covariance,
magnitudes, cross-terms), and both datasets:

```
features computed on the RECONSTRUCTION   0.449 – 0.516
features computed on the CODE             0.265 – 0.418
```

No mechanistic story for *why* survived. The most likely remaining explanation — not yet
tested — is that the learned stamps are not band-selective: if each `Dᵢ` spans multiple
frequencies, "power in stamp i" never maps onto "power in mu", and no reduction of the
coefficients recovers band-specific power. `stamp_fingerprint_similarity.png` and the
per-stamp PSD panels from the v5 analysis would show this directly, as a read rather
than another experiment.

## 6. The masked-pretrain stage does not change any of this

Added 2026-09-17. Every probe in sections 1–5 read `best_tokenizer.pth`, because no
pretrain checkpoint had ever been produced. `mesae_pretrain_v9` (stopped at epoch 39/50,
converged: train 0.3318 / val 0.3430) is the first, so "is pretrain z better than
tokenizer z" was unmeasured rather than refuted. Now measured.

`pretrain_probe.py` + `probe_paired.py`: same 2592 trials, same folds, same PCA64 +
shrinkage LDA for both checkpoints, so the encoder is the only thing that differs. Both
have exactly 22 alive routed stamps of 120. Paired per-subject, 9 subjects:

| feature | tokenizer | pretrain | Δ | sd(Δ) | wins | p |
|---|---|---|---|---|---|---|
| `head_z` | 0.276 | 0.304 | +0.028 | 0.037 | 6/9 | 0.054 |
| `chan_mag` | 0.393 | 0.399 | +0.006 | 0.014 | 6/9 | 0.255 |
| `z_mean` | 0.269 | 0.260 | −0.010 | 0.034 | 3/9 | 0.418 |
| `recon_bandpow` | 0.477 | 0.488 | +0.011 | 0.020 | 8/9 | 0.147 |

**No feature improves significantly.** `head_z`'s p=0.054 is borderline alone and fails a
4-test correction (p<0.0125). Three of four move in the right direction but small.

Two consequences:

- **The `z_mean` head candidate is refuted.** Mean-pooling `stage_features` over valid
  channels and patches — dropping the channel softmax that section 3 identified as the
  larger loss — sits at chance in *both* stages (0.269 / 0.260). Removing the softmax is
  not sufficient; there is nothing linearly decodable in the pooled `z` to recover.
- **The section-5 ordering is unchanged by pretraining.** `recon_bandpow` (0.488) still
  beats `chan_mag` (0.399) still beats `head_z` (0.304), in the pretrain column exactly as
  in the tokenizer column. The decision below stands on both stages' outputs.

This also settles a question raised separately: whether to merge the two stages into one
run with a curriculum (tokenizer-only warmup, then activate the rest of the encoder and
ramp the mask). Merging optimizes the cost of a stage that has not been shown to pay.
Not worth the mechanism until stage 2 demonstrates a gain — which, with an MAE objective,
it has not.

*Amended 2026-09-17:* the merge was done anyway, for simplicity rather than accuracy —
see `0013`.

## Decision

**The finetune head decodes to signal space first, then extracts features.**

`recon = Σᵢ aᵢDᵢ + bᵢHᵢ` is a single einsum against the frozen dictionary — cheap and
differentiable. The head takes the code, reconstructs, then applies spatial filtering
and log-power. The FM interface stays the code; the head's first layer decompresses it.

Consequences:

- Removes the need to invent code-space equivalents of classical features per paradigm.
  Back in signal space: MI takes band power + spatial contrast, SSVEP takes coherent
  phase across patches, ERP takes signed time-domain amplitude at latency. All standard.
- Replaces, or bypasses, `encode_post_stamp_expert`'s channel softmax pooling, which
  section 3 identifies as the larger of the two losses (0.404 → 0.279).
- Backbone stays frozen. Explicitly preferred; unfreezing is the last resort.

## Rejected alternatives

- **Wider code-space features** (topography, temporal bins, cross-terms) — sections 4a–4c,
  all refuted, and wide features additionally invite subject-identity leakage (4a).
- **Unfreezing the backbone** — would likely work but discards the frozen-tokenizer
  property the whole two-stage design exists to provide.

## Open

- Are the stamps band-selective? Read the v5 fingerprint/PSD panels before building.
- SSVEP phase-coherence probe never run. `patch_len` 250 ms gives 4 Hz resolution against
  ~0.2 Hz class spacing, so cross-patch phase tracking is required, not optional. Test
  before committing to an SSVEP head.
