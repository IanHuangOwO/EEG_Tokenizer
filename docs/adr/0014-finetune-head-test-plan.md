# 0014 — Finetune head: what to expose, how to pool, compared to raw (draft)

Status: Proposed (draft). This implements the direction decided in ADR 0012. Loss
trimming (ADR 0015) waits until this is settled.
Date: 2026-09-17

## Questions

1. **What does the head see?** The reconstruction, stamp codes, encoder features, or
   the raw signal.
2. **How is it pooled?** Over channels, over time, and over stamps.
3. **Does the tokenizer add anything over raw EEG** when the head is held fixed?

## What we already know (ADR 0012)

Model v5 (tokenizer stage), BCICIV2a, per-subject shrinkage LDA:

| representation | acc | note |
|---|---|---|
| raw mu/beta log band power | 0.513 | the classical floor (published range) |
| recon mu/beta log band power | 0.516 | the reconstruction keeps the class information |
| `chan_mag` (per-channel stamp magnitude) | 0.404 | about 0.05 below recon at matched preprocessing (ADR 0012 §4d) |
| `pool_mag` (after channel softmax) | 0.316 | the softmax pool destroys the channel contrast |
| `head_z` (current `MeSAEFinetune` input) | 0.279 | |
| `z_mean` (mean-pooled encoder z) | 0.26–0.27 | refuted, in v9 tokenizer and v9 pretrain alike |
| chance | 0.250 | |

Rules that came out of those measurements:

- **Features computed on the reconstruction beat features computed on the code**, in
  every family tried.
- **A softmax (convex) pool over channels can't represent a C3−C4 contrast.** That is
  the main loss.
- **Wider code features gained nothing and leak subject identity.**
- **Temporal binning and second-order code features were refuted.**
- **The masked stage changed none of this** (ADR 0012 §6, v9).

Two finetune caveats:

- **Earlier finetune runs had spatial attention off.** Before ADR 0013 the finetune
  loader restored temporal mixing only. Those numbers are invalid as a statement about
  the pretrained backbone.
- **One known failure pattern:** a head with far more parameters than training samples
  memorizes. The signature is train at 100% while val stays flat at chance. Always check
  head size against sample count first.

## Design under test

The previous session's proposal, now written down:

```
backbone(x, bool_masked_pos=None).recon   [B,C,N,L]
  -> overlap_add_patches                   [B,C,T]   (same path as the 0.516 probe)
  -> mask by valid_channels
  -> spatial filter  Linear(C, K), signed, no softmax   (K ~ 8-16; OFF in v0)
  -> task block                            [B, F]
  -> Linear(F, num_classes)
```

Task blocks. The task is chosen by a config key, never inferred.

- **MI:** `rfft` over the whole trial, then `|·|²`, summed over mu (8–13) and beta
  (13–30), then `log`. Output is `[B, K·2]`.
- **ERP:** `avg_pool1d` down to about 20 Hz, then flatten. Output is `[B, K·T']`. Signed
  amplitude, no log.
- **SSVEP:** `rfft` over the whole trial, never per patch (a 250 ms patch gives 4 Hz bins
  against 0.2 Hz class spacing). Take the magnitude at each target frequency and its
  2nd harmonic, or use CCA-lite against sin/cos references.

New head class. `PerChannelHeadAttn` / `MeSAEFinetune` stay as the `head_z` comparison
arm.

## Reading from the reconstruction is not the foundation-model test

The convention is to read the encoder:

- **MAE:** the decoder is discarded after pretraining. Linear probing and finetuning both
  read encoder features.
- **BERT:** reads the CLS or pooled token.
- **EEG foundation models (LaBraM, CBraMod and similar):** mean-pool the encoder tokens
  into a small head, and mostly report full finetuning. As far as we recall, frozen
  linear probing on EEG foundation models is usually much weaker.

A head that reads the reconstruction evaluates the model as a **codec**: does the
information survive compression? Its best case is about equal to raw (0.516 vs 0.513).
It can never show that pretraining learned something beyond raw EEG. So `recon` is the
information-preservation ceiling here, and the encoder arm `z_chan` is the actual
foundation-model claim.

`z_mean` was chance in ADR 0012 §6, but it averaged over channels, and a channel
average cannot express a C3−C4 contrast. The conventional arm had therefore never been
tested fairly.

Caveat: after spatial attention, `z_c` already mixes information across channels. That
is fine for decoding, but it means `z_chan` is not channel-interpretable.

## Input arms (same head, same split, same seeds)

| arm | head input | role |
|---|---|---|
| `raw` | `x` itself (skip the backbone) | the floor for this head |
| `z_chan` | encoder `z`, time-mean per channel `[B,C,D]` → signed spatial filter → linear | **primary: the conventional foundation-model arm** |
| `recon` | the decoded reconstruction | information-preservation ceiling (expected ≈ raw) |
| `chan_mag` | per-channel stamp magnitudes, through the same spatial filter + linear, no band power | is code space usable once the softmax pool is gone? |
| `stamp_bandpow` | per-channel band power computed in code space from the stamp templates (below) | do the templates `D_i` carry the band information that `chan_mag` misses? |
| `head_z` | the current `MeSAEFinetune` | the existing head, as the reference to beat |

**`stamp_bandpow` uses `D_i` as a fixed stamp-to-band map.** `D_i` is the same for every
trial, so on its own it cannot be a feature. It becomes useful as a weight on the
amplitudes:

```
stamp_bandpow[c, band] = log mean_patches Σ_i ( a_ic²·E_D[i, band] + b_ic²·E_H[i, band] )
E_D / E_H[i, band] = band energy of template D_i / its quadrature partner H_i
```

This formula is exact per stamp. Because `H_i` is a quadrature partner, the `a·b` cross
term vanishes in every positive-frequency bin (checked numerically: relative error 2e-7).
`H` has its own energy table because `_quadrature` renormalizes it after zeroing DC and
Nyquist.

What it leaves out is the cross-stamp terms. It also needs no decoding: it is
`chan_mag`'s amplitudes, weighted by each template's spectrum. Reading the result:

- **≈ `recon`:** the stamps are band-selective, and the cross-stamp terms don't matter.
- **≈ `chan_mag`:** the band information lives in how stamps combine, not in the
  individual templates. That is ADR 0012's untested explanation, now testable.

Caveat: `patch_len` 50 at 200 Hz gives 4 Hz template bins. mu is only the 8 and 12 Hz
bins, so the band edges are coarse.

`raw` and `recon` go through the identical module, so their comparison is fair. It
answers "does the tokenizer keep the information", not "did it learn a better
representation" — that is `z_chan` against `raw`.

### Where `D_i` fits

`D_i` cannot be a per-stamp feature. Within one stamp, its band power is
`amp² · E_i(band)`, which is log amplitude plus a constant. That gives LDA the same
information as the amplitude alone. Across stamps, `D_i` is what separates the three
code-space readouts:

| input | stamp attribution | cross-stamp terms |
|---|---|---|
| per-stamp amplitude / phase | yes | no |
| `stamp_bandpow` (amplitudes weighted by `E_i`, summed) | lost in the sum | no |
| `recon` (`Σ a·D + b·H`, then band power) | lost | yes |

Feeding `D_i` together with every stamp's amplitudes is therefore `recon` minus the
cross terms. For attribution, keep the per-stamp amplitude and phase, and use `D_i` only
to name which band a relevant stamp covers.

## Stamp attribution: which stamps carry task information

No change to the stamps is needed. The code `(a, b)` per stamp, channel and patch
already separates the two kinds of event-related activity:

- **Amplitude `√(a²+b²)`:** the envelope, which carries induced activity (ERD/ERS).
- **Phase `atan2(b, a)`:** measured relative to each patch start. The trials are
  cue-aligned (BCICIV2a cue at sample 200), so a given patch sits at the same
  post-cue time in every trial, and consistent phase across trials means phase-locked
  (ERP-like) activity.

The hidden bottleneck (`W_down`, width 6) sits before that readout and has no physical
meaning, so it is not used.

The existing panels do not answer this question:

- `event_stamp_dynamics` is cue-locked but pools all classes.
- `pool_label_probe` decodes at the pool level (routed vs shared), not per stamp.

`stamp_relevance.py` measures, for each alive routed stamp and each shared stamp.
Baseline is the pre-cue patches; the MI window is 0.5–4 s after the cue.

1. **`decod`:** per-subject shrinkage LDA on that stamp's log post-cue power per channel.
   `decod_bl` uses dB change from baseline instead. Significance is a one-sided t-test
   of per-subject accuracy against 0.25, with BH-FDR across stamps.
2. **`erd_db`:** post/pre power change, which answers "event-related at all?".
   **`lri_db`:** `[dB(C3,R) − dB(C3,L)] − [dB(C4,R) − dB(C4,L)]`. Contralateral ERD
   makes this negative, so a significant negative value means task-specific in the
   expected direction.
3. **`itc`:** inter-trial phase coherence of the stamp's phase, post minus pre, max over
   channels. This flags ERP-like stamps. The bias floor is about `1/√(trials per subject)`.
4. **Causal importance** (once the head exists): zero stamp i in the reconstruction and
   measure the head's accuracy drop. Measures 1–3 are correlational; this one checks
   that the head actually uses the stamp.

Each row also reports the approximate top-k selection rate (`sel`) and the template's
peak frequency.

Caveats:

- **`dense_amp` is response, not use.** Read `decod` next to `sel`.
- **v10 stamps are not clean source maps.** They trained unfrozen on spatially mixed `z`
  (ADR 0013), so their topographies are not source maps. The rankings still hold.
- **Templates have 4 Hz bins,** so band labels are coarse.
- **The v10 data is small,** and BCICIV2a was not in pretraining.

Changing the stamps only becomes relevant if the task-relevant stamps turn out to span
several bands. A band-selectivity constraint would then make attribution cleaner.

## Protocol

- **Dataset:** BCICIV2a, `split_mode: intra_subject`. That is 9 per-subject models with
  a class-stratified trial split, so about 230 train / 58 val trials per subject.
- **Backbone frozen.** Unfreezing is the last resort (ADR 0012).
- **Reporting:**
  - Per-subject accuracy and kappa.
  - Paired t-test across the 9 subjects for each arm against `raw`, with a correction for
    multiple comparisons.
  - Report the last-epoch value alongside best-val-acc. With about 58 val trials,
    choosing the best epoch on val is optimistic.
- **`inter_subject` / LOSO numbers are reported separately** and never mixed with
  intra-subject numbers (subject-wise everything sits at 0.27–0.31).
- **Backbone:** `mesae_v10_small_uw01` for plumbing. Conclusions about the tokenizer need
  the full-data run.

## Build order (each step has one acceptance check)

1. **v0, MI block, spatial filter off, arm `raw`.**
   - Must reach about 0.51 mean.
   - A miss means a wiring bug (band edges, FFT length, valid_channels, split), not a
     design problem.
2. **v0, arms `recon` and `z_chan`.**
   - `recon` must be within noise of `raw`. `z_chan` is the result that matters.
   - A miss means a recon-path bug (overlap-add, valid mask, phase flags).
3. **Spatial filter on, `raw` and `recon`.** Must beat step 1 (the CSP-like gain).
4. **Arms `chan_mag` and `stamp_bandpow`** through the same filter. Run them only if
   the LDA probe shows `stamp_bandpow` above `chan_mag`; otherwise `chan_mag` alone.
5. **ERP block.**
6. **SSVEP block,** only after the phase-coherence probe (ADR 0012 open item) shows the
   reconstruction keeps cross-patch phase.

## Implementation notes

- **Config:** `model_params.MeSAE.finetune` gets
  - `input` (`raw | z_chan | recon | chan_mag | stamp_bandpow | head_z`),
  - `task` (`mi | erp | ssvep`),
  - `spatial_filters` (0 = off).
- **Data:** `FinetuneDataset` already yields the whole trial. The `raw` arm can take
  `[B,C,T]` directly; the other arms patchify through `FinetuneCollate` as today.
- **Reconstruction:** decode inside `forward` with the backbone in eval mode under
  `no_grad` (frozen). Caching is an optimization for later.
- **Band power:** use the whole-trial FFT at the dataset's `sample_freq`. The band edges
  match `recon_lda.py` / `pretrain_probe.py` so the LDA numbers stay comparable.
- **Cleanup:** fix the stale `MeSAEFinetune` docstring. Its 0.373 vs 0.255 topography
  claim came from trial-wise CV and was shown to be leakage (ADR 0012 §4a).
- **Config paths:** `training_params.finetune.pretrained_checkpoint` already points at
  `mesae_v10/checkpoint/last.pth`. Switch it to the small run for plumbing.

## Open prerequisites

- **v10 probe** (`probe_v10.py` on `mesae_v10_small_uw01`, now including `z_chan` and
  `stamp_bandpow`): check that the recon > chan_mag > head_z ordering still holds under
  the fused, unfrozen training, and see where `z_chan` lands.
- **Stamp attribution** (`stamp_relevance.py`, same run).
- **Stamp band-selectivity:** read the stamp PSD / fingerprint panels (ADR 0012 open).
- **SSVEP phase-coherence probe:** not run yet.
