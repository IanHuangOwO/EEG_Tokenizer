# 0014 — Finetune head: what to expose, how to pool, compared to raw (draft)

Status: Proposed (draft). Experiment A has partial results; experiment B has its
baseline (steps 2-3 pass). Implements the direction decided in ADR 0012. Loss trimming (ADR 0015) waits
until this is settled.
Date: 2026-09-17

## Questions

1. **Features:** what does the head read? The raw signal, the reconstruction, stamp
   codes, or encoder features.
2. **Pooling:** how is it collapsed over time, channels, and stamps?
3. **Value of the tokenizer:** does it add anything over raw EEG when everything else is
   held fixed?

**Features and pooling are separate experiments** (A and B below). Each run changes one
factor, and every other factor stays at its fixed baseline. Some existing readouts
bundle a feature with a pooling choice: `head_z` (channel softmax), `z_mean` (channel
mean) and `pool_mag`. They are kept only as references, never as feature arms, because
a bad number from them cannot be attributed to either factor.

## What we already know (ADR 0012)

Model v5 (tokenizer stage), BCICIV2a, per-subject shrinkage LDA:

| representation | acc | note |
|---|---|---|
| raw mu/beta log band power | 0.513 | plain LDA, 128 features |
| recon mu/beta log band power | 0.516 | the reconstruction keeps the class information |
| `chan_mag` (per-channel stamp magnitude) | 0.404 | PCA64 |
| `pool_mag` (after channel softmax) | 0.316 | bundled: the softmax pool destroys the channel contrast |
| `head_z` (current `MeSAEFinetune` input) | 0.279 | bundled |
| `z_mean` (encoder z, mean over channels) | 0.26–0.27 | bundled; v9 tokenizer and v9 pretrain alike |
| chance | 0.250 | |

Rules that came out of those measurements:

- **A softmax (convex) pool over channels can't represent a C3−C4 contrast.**
- **Wider code features gained nothing and leak subject identity.**
- **Temporal binning and second-order code features were refuted.**
- **The masked stage changed none of this** (ADR 0012 §6, v9).
- **Compare only within one preprocessing pipeline** (ADR 0012 §4d).

Two finetune caveats:

- **Earlier finetune runs had spatial attention off.** Before ADR 0013 the finetune
  loader restored temporal mixing only, so those numbers say nothing about the
  pretrained backbone.
- **Watch for memorization.** A head with far more parameters than training samples
  memorizes: train goes to 100% while val stays flat at chance. Check head size against
  sample count first.

## Reading from the reconstruction is not the foundation-model test

The convention is to read the encoder:

- **MAE:** discards the decoder after pretraining. Linear probing and finetuning both
  read encoder features.
- **BERT:** reads the CLS or pooled token.
- **EEG foundation models (LaBraM, CBraMod and similar):** mean-pool encoder tokens into
  a small head and mostly report full finetuning. As far as we recall, their frozen
  linear-probe results are usually much weaker.

A head that reads the reconstruction evaluates the model as a **codec**: does the
information survive compression? Its best case is about equal to raw, so it can never
show that pretraining learned something beyond raw EEG. The feature arms therefore
play these roles:

- **`raw`:** the floor.
- **`recon`:** the information-preservation ceiling.
- **`z_chan` (encoder features):** the actual foundation-model claim.

`z_mean` was chance in ADR 0012 §6, but it averaged over channels, so the conventional
arm had never been tested fairly before `z_chan`.

Caveat: after spatial attention, `z_c` already mixes information across channels. That
is fine for decoding, but `z_chan` is not channel-interpretable.

## Experiment A — features (pooling held fixed)

**Fixed pooling for every arm:**

- **Time:** the one statistic that fits the feature. For power features, log power over
  the whole trial; for `z`, the mean over patches.
- **Channels:** none. All valid channels are concatenated.
- **Stamps:** as each feature defines them.
- **Readout:** per-subject shrinkage LDA, with PCA64 whenever the feature is wider than
  64, 5-fold stratified CV and identical folds for every arm. Script: `probe_v10.py`.

| arm | definition | width |
|---|---|---|
| `raw_bandpow` | raw signal (patched span, overlap-added), mu/beta log power per channel | 128 |
| `recon_bandpow` | the same statistic on the reconstruction | 128 |
| `stamp_bandpow` | the same statistic computed in code space from stamp amplitudes and templates (below) | 128 |
| `chan_mag` | per-channel dense stamp magnitude `√(a²+b²)`, alive stamps, time-mean | C·S |
| `z_chan` | encoder `z`, time-mean per valid channel | Cv·D |

### `stamp_bandpow`: using `D_i` as a fixed stamp-to-band map

`D_i` is the same for every trial, so it cannot be a feature on its own. Used as a
weight on the amplitudes:

```
stamp_bandpow[c, band] = log mean_patches Σ_i ( a_ic²·E_D[i, band] + b_ic²·E_H[i, band] )
E_D / E_H[i, band] = band energy of template D_i / its quadrature partner H_i
```

**The formula is exact for each stamp.** `H_i` is a quadrature partner, so the `a·b`
cross term vanishes in every positive-frequency bin (checked numerically: relative
error 2e-7). `H` has its own energy table because `_quadrature` renormalizes it after
zeroing DC and Nyquist.

**It still differs from `recon_bandpow`.** It needs no decoding, and it leaves out the
cross-stamp terms. Template bins are 4 Hz wide (`patch_len` 50 at 200 Hz), so the band
edges are coarse.

**Where `D_i` fits.** Within one stamp, band power is `amp²·E_i(band)` = log amplitude +
a constant, which gives LDA no more than the amplitude alone. Across stamps, `D_i` is
what separates the three code-space readouts:

| readout | stamp attribution | cross-stamp terms |
|---|---|---|
| per-stamp amplitude / phase | yes | no |
| `stamp_bandpow` | lost in the sum | no |
| `recon_bandpow` | lost | yes |

Feeding `D_i` together with every stamp's amplitudes amounts to `recon` minus the cross
terms. For attribution, keep the per-stamp amplitude and phase, and use `D_i` only to
name a stamp's band.

### Results — `mesae_v10_small_uw01/checkpoint/last.pth`

Run context: `last.pth`, 7 datasets × 3 subjects, fused run, stamps unfrozen, leaky
`mp_loss`. 21 of 60 routed stamps are alive. BCICIV2a (not in pretraining), 2592
trials, 9 subjects, chance 0.25. Everything below comes from the one pipeline above.

| arm | acc | Δ vs raw | p (paired) | wins |
|---|---|---|---|---|
| `raw_bandpow` | 0.482 | — | — | — |
| `recon_bandpow` | 0.490 | +0.008 | 0.41 | 5/9 |
| `stamp_bandpow` | 0.454 | −0.028 | 0.043 | 3/9 |
| `chan_mag` | 0.359 | −0.123 | 0.002 | 0/9 |
| `z_chan` | 0.341 | −0.141 | 0.001 | 0/9 |
| *ref* `head_z` (bundled) | 0.330 | −0.152 | <0.001 | 0/9 |
| *ref* `z_mean` (bundled) | 0.282 | −0.200 | <0.001 | 0/9 |

Against the v9 columns cached by `pretrain_probe.py` (same pipeline):

| arm | v9 tokenizer | v9 pretrain | v10 small | Δ vs v9 pretrain | p | wins |
|---|---|---|---|---|---|---|
| `recon_bandpow` | 0.477 | 0.488 | 0.490 | +0.002 | 0.75 | 6/9 |
| `chan_mag` | 0.393 | 0.399 | 0.359 | −0.040 | 0.068 | 2/9 |
| `head_z` | 0.276 | 0.304 | 0.330 | +0.026 | 0.11 | 6/9 |
| `z_mean` | 0.269 | 0.260 | 0.282 | +0.022 | 0.11 | 6/9 |

Reading:

- **The tokenizer still preserves the information.** `recon` ≈ `raw` (0.490 vs 0.482),
  even on the small-data v10.
- **The templates close most of the code-space gap.** `stamp_bandpow` recovers
  0.359 → 0.454 over `chan_mag` from the same amplitudes, leaving 0.036 below `recon`.
  - ADR 0012's untested explanation, that the stamps are not band-selective, is mostly
    **refuted**. The `chan_mag` gap came from pooling magnitudes across bands without
    each stamp's spectral weight, not from missing band information.
  - A code-space head that keeps stamp attribution is therefore viable. It costs about
    0.03 against raw (p = 0.043, uncorrected).
- **Encoder features are weak under this readout.** `z_chan` is 0.14 below raw, with 0
  of 9 subjects winning.
  - This is linear on a time-mean. A time-mean of `z` can cancel the oscillatory
    information that power features keep, so experiment B has to test richer time
    pooling before `z_chan` is written off.
  - It is also 2200 features against 288 trials per subject (PCA64).
- **`chan_mag` dropped vs v9** (−0.040, p = 0.068). Plausible causes are fewer alive
  stamps (21 of 60 vs 22 of 120) and the unfrozen dictionary. Not investigated.

## Experiment B — pooling (feature held fixed)

**Features:** `recon`, plus `raw` as the control. Also `z_chan` and `stamp_bandpow`, the
two backbone features experiment A says are worth pooling better.

**Readout:** a trained head, not LDA.

**Baseline:** channel concat, whole-trial statistic, `E`-weighted stamp sum.

One axis changes per run; the others stay at baseline:

| axis | variants |
|---|---|
| **B1 channel** | concat (baseline) · signed spatial filter `Linear(C, K)`, K ∈ {4, 8, 16} · softmax attention (known bad, reference only) |
| **B2 time** | whole-trial statistic (baseline) · mean of per-patch statistic · attention over patches · post-cue window only (0.5–2.5 s, where the raw lateralization lives, see below) · for `z_chan`: per-patch nonlinearity (e.g. `log(z²)` or a small MLP) before the mean |
| **B3 stamp** (code features only) | `E`-weighted sum (baseline, = `stamp_bandpow`) · per-stamp concat · learned `Linear(S, bands)` initialised from `E` · attention over stamps |

Task blocks (the task is chosen by a config key, never inferred):

- **MI:** `rfft` over the pooled window, `|·|²`, summed over mu (8–13) and beta (13–30),
  then `log`.
- **ERP:** `avg_pool1d` to about 20 Hz, then flatten. Signed, no log.
- **SSVEP:** `rfft` over the whole trial, never per patch (4 Hz patch bins against
  0.2 Hz class spacing). Magnitude at each target frequency and its 2nd harmonic, or
  CCA-lite.

The design proposed last session is B1 = spatial filter plus the MI block, on `recon`:

```
backbone(x, bool_masked_pos=None).recon -> overlap_add_patches [B,C,T] -> valid mask
  -> Linear(C, K) signed -> MI block -> Linear(F, num_classes)
```

New head class. `PerChannelHeadAttn` / `MeSAEFinetune` stay as the bundled `head_z`
reference.

### Results — experiment B baseline (concat, whole trial, MI block)

Backbone: `mesae_v10_small_uw01/checkpoint/last.pth` (frozen). `MeSAEFeatureHead` runs
BCICIV2a intra-subject: 228 train / 60 val trials per subject, balanced classes (15 per
class in val).

Every number is `balanced_acc`, which equals accuracy on a balanced val set. The main
number is the mean over each subject's last 10 epochs; best-val-epoch is listed as the
optimistic reading.

| arm | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | last-10 mean | best-val mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `raw` | .55 | .46 | .69 | .46 | .24 | .30 | .64 | .59 | .52 | **0.495** | 0.552 |
| `recon` | .52 | .43 | .68 | .48 | .23 | .33 | .63 | .61 | .47 | **0.486** | 0.548 |

- **Step 2 passes.** `raw` 0.495 matches the LDA probe on the same features (0.482), so
  the head is wired correctly.
- **Step 3 passes.** `recon − raw` = −0.009 (p = 0.35, 3/9 wins); on best-val the
  difference is −0.004 (p = 0.75). The reconstruction path is lossless for this readout.
- **Train/val gap is small** (last 10 epochs: train 0.61 vs val 0.49), so the linear
  head is not memorizing.
- **S5 and S6 sit near chance under every readout,** including LDA. Those subjects are
  hard, not broken.

**The optimizer setting matters.** The first `raw` run used the repo finetune defaults
(lr 1e-3 cosine, 50 epochs, dropout 0.3) and reached only 0.418 on the last-10 mean.
Train and val were both around 0.4 and val loss was still falling at the end, which is
underfitting, not overfitting. Every experiment-B run therefore uses lr 1e-2 (min 1e-3),
100 epochs, warmup 2, dropout 0.

Also found on the way (commit `fe969f6`):

- **Logged `acc` was a mean of per-batch accuracies,** which overweights a short last
  batch. It is now computed over all predictions.
- **Padded channels went into the head at log(eps).** That pinned val to a single class
  until they were zeroed.

## Stamp attribution — which stamps carry task information

No change to the stamps is needed. The code `(a, b)` per stamp, channel and patch
already separates the two kinds of event-related activity:

- **Amplitude `√(a²+b²)`:** the envelope, which carries induced activity (ERD/ERS).
- **Phase `atan2(b, a)`:** measured relative to each patch start. The trials are
  cue-aligned (BCICIV2a cue at sample 200), so a patch sits at the same post-cue time in
  every trial, and consistent phase across trials means phase-locked (ERP-like)
  activity.

The hidden bottleneck (`W_down`, width 6) sits before that readout and has no physical
meaning.

The existing panels do not answer this question. `event_stamp_dynamics` is cue-locked
but pools all classes. `pool_label_probe` works at the pool level (routed vs shared),
not per stamp.

`stamp_relevance.py <run> [onset] [post_end_s]` computes, for each alive routed stamp and
each shared stamp (pre-cue patches as baseline):

1. **`decod` / `decod_bl`:** per-subject shrinkage LDA on the stamp's log post-cue power
   per channel, or its dB change from baseline. One-sided t-test against 0.25,
   BH-FDR across stamps.
2. **`erd_db`:** post/pre dB change.
   **`lri_db`:** `[dB(C3,R) − dB(C3,L)] − [dB(C4,R) − dB(C4,L)]`. Contralateral ERD
   makes it negative.
3. **`itc`:** inter-trial phase coherence, post − pre, max over channels. The bias
   floor is about `1/√(trials per subject)`.
4. **Causal importance** (once the head exists): zero stamp i and measure the head's
   accuracy drop. Measures 1–3 are correlational only.

Each row also reports the approximate top-k selection rate (`sel`) and the template's
peak frequency.

### Results — same checkpoint, post window 0.5–2.5 s

The 0.5–4 s window gave the same ranking with slightly lower `decod`.

| stamp | kind | peak Hz | sel | decod | q | lri_db | q |
|---|---|---|---|---|---|---|---|
| 62 | shared | 12 | 1.00 | 0.480 | 0.007 | −0.53 | 0.41 |
| 63 | shared | 52 | 1.00 | 0.426 | 0.008 | −0.35 | 0.41 |
| 43 | routed | 12 | 0.97 | 0.400 | 0.008 | −0.19 | 0.52 |
| 46 | routed | 20 | 0.96 | 0.397 | 0.007 | −0.46 | 0.36 |
| 9 | routed | 24 | 0.96 | 0.383 | 0.003 | −0.42 | 0.38 |
| 4 | routed | 16 | 0.96 | 0.373 | <0.001 | −0.37 | 0.36 |
| 61 | shared | 8 | 1.00 | 0.363 | 0.025 | −0.08 | 0.81 |
| 21 | routed | 28 | 0.93 | 0.353 | 0.018 | −0.31 | 0.36 |
| 60 | shared | 0 (DC) | 1.00 | 0.349 | 0.002 | +0.30 | 0.41 |
| 16 | routed | 0 (DC) | **0.00** | 0.325 | 0.007 | −0.08 | 0.60 |
| … | routed | 32–100 | 0.12–0.79 | 0.24–0.31 | ≥0.09 | | |

Reading:

- **Task information concentrates in mu/beta stamps** (template peaks 8–28 Hz). Stamps
  peaking above about 40 Hz sit near chance. That matches MI physiology, and it is
  evidence the dictionary is organised by frequency.
  - Exception: shared stamp 63 (peak 52 Hz). The peak alone is a poor label for an
    always-on stamp; read its full spectrum before interpreting it.
- **The task-relevant routed stamps are selected in about 96% of patches.** They
  behave like extra shared stamps: 5 routed stamps have `sel` above 0.9 (7 above 0.75),
  so a large share of the 12 top-k slots is taken almost permanently, leaving few slots
  for content-dependent selection. This links to the
  high dead rate (39 of 60 dead) and is worth watching in the full run.
- **Response is not use.** Stamp 16 is never selected (`sel` 0.00), yet its dense
  response decodes at 0.325.
- **Lateralization points the expected way** for the mu/beta stamps (negative), but
  none survives FDR.
- **`itc` is at or below the bias floor (0.059) for every stamp**, so there are no
  phase-locked stamps. The two DC stamps (60, 16) have the highest `itc` in the
  0.5–4 s run (about 0.09), which is consistent with slow evoked potentials.
- **Don't read ERD sign from stamps.** Nearly every stamp shows a post-cue power
  *increase* (+0.2 to +0.6 dB, high-frequency stamps included), and the raw signal does
  not. A model-free check on the raw BCICIV2a signal (200-sample windows, Hann, pre-cue
  0–1 s):
  - **mu:** −0.40 dB at 0.5–1.5 s (p = 0.17), then turns positive.
  - **beta:** about 0 dB, but with significant lateralization at 0.5–1.5 s
    (LRI −0.59, p = 0.021) and 1.5–2.5 s (−0.39, p = 0.033), fading by 2.5–3.5 s.
  - **55–95 Hz:** +0.1 to +0.2 dB.

  So the stamp amplitude is not a faithful band-limited envelope, and ERD sign must be
  read from `raw` or `recon`. The class information in stamp power is still real
  (`decod`).

Caveats:

- **`dense_amp` is response, not use.** Read `decod` next to `sel`.
- **v10 stamps are not clean source maps:** they trained unfrozen on spatially mixed `z`
  (ADR 0013).
- **Coarse bands:** templates have 4 Hz bins.
- **Small v10 data**, and BCICIV2a was not in pretraining.

Changing the stamps only becomes relevant if relevant stamps turn out to span several
bands. A band-selectivity constraint would then make attribution cleaner.

## Protocol (experiment B)

- **Dataset:** BCICIV2a, `split_mode: intra_subject`. That is 9 per-subject models with
  a class-stratified trial split, about 230 train / 58 val trials per subject.
- **Backbone frozen.** Unfreezing is the last resort (ADR 0012).
- **Reporting:**
  - Per-subject accuracy and kappa.
  - Paired t-test across the 9 subjects against the `raw` arm under the same pooling,
    with a correction for multiple comparisons.
  - Report the last-epoch value alongside best-val-acc (about 58 val trials makes
    best-epoch selection optimistic).
- **`inter_subject` / LOSO numbers are reported separately**, never mixed with
  intra-subject numbers.
- **Backbone:** `mesae_v10_small_uw01` for plumbing. Conclusions about the tokenizer need
  the full-data run.

## Build order (each step has one acceptance check)

1. **A (done):** the feature probe above.
2. **B baseline, `raw` (done, passes):** concat + whole-trial MI block + linear.
   - Must reach about the probe's `raw` (0.48–0.51). A miss is a wiring bug (band edges,
     FFT length, valid_channels, split).
3. **B baseline, `recon` (done, passes):** must be within noise of `raw`. A miss is a recon-path bug
   (overlap-add, valid mask, phase flags).
4. **B1 channel** on `raw` and `recon`. The spatial filter must beat step 2.
5. **B2 time,** including the post-cue window and the `z_chan` nonlinearity variants.
6. **B3 stamp** on `stamp_bandpow`.
7. **Stamp attribution measure 4** (occlusion) with the best code-space head.
8. **ERP block,** then **SSVEP** (only after the phase-coherence probe).

## Implementation notes

- **Config:** `model_params.MeSAE.finetune` gets
  - `input` (`raw | recon | z_chan | stamp_bandpow | chan_mag | head_z`),
  - `task` (`mi | erp | ssvep`),
  - `pool_channel` (`concat | spatial:K | softmax`),
  - `pool_time` (`trial | patch_mean | attention | window:lo-hi`),
  - `pool_stamp` (`energy | concat | linear | attention`).

  Pooling is its own set of keys, so experiment B only ever edits one of them.
- **Data:** `FinetuneDataset` already yields the whole trial. The `raw` arm takes
  `[B,C,T]` directly; the other arms patchify through `FinetuneCollate`.
- **Reconstruction:** decode inside `forward` with the backbone in eval mode under
  `no_grad`. Caching is an optimization for later.
- **Band power:** whole-trial FFT at the dataset's `sample_freq`, with band edges
  matching `probe_v10.py`.
- **Cleanup:** fix the stale `MeSAEFinetune` docstring. Its 0.373 vs 0.255 topography
  claim came from trial-wise CV (leakage, ADR 0012 §4a).
- **Config path:** `training_params.finetune.pretrained_checkpoint` points at
  `mesae_v10/checkpoint/last.pth`. Switch it to the small run for plumbing.

## Open prerequisites

- **Stamp band-selectivity:** read the full spectra, especially the shared stamps (63).
- **Stamp dominance:** why about 7 routed stamps are selected almost always. Check this
  in the full-data run.
- **SSVEP phase-coherence probe:** not run yet.
- **Probe scripts are in the session scratchpad** (`probe_v10.py`,
  `stamp_relevance.py`). Move them into the repo if they are kept.
