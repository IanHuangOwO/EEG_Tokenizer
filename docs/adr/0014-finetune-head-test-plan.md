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

#### Feature arms under the same baseline pooling, and the first pooling change

Same protocol and optimizer. Paired tests are against `raw` concat, using each subject's
last-10-epoch mean.

| run | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | last-10 | Δ vs raw concat | p | wins | train last-10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `raw` concat | .55 | .46 | .69 | .46 | .24 | .30 | .64 | .59 | .52 | 0.495 | — | — | — | 0.61 |
| `recon` concat | .52 | .43 | .68 | .48 | .23 | .33 | .63 | .61 | .47 | 0.486 | −0.009 | 0.35 | 3/9 | 0.61 |
| `stamp_bandpow` concat | .51 | .44 | .58 | .45 | .28 | .37 | .59 | .59 | .43 | 0.470 | −0.025 | 0.22 | 2/9 | 0.61 |
| `z_chan` concat | .44 | .37 | .54 | .30 | .23 | .24 | .36 | .55 | .53 | 0.395 | −0.100 | 0.009 | 1/9 | **0.99** |
| `z_chan` concat, `z_proj` 2 | .42 | .39 | .62 | .26 | .20 | .33 | .43 | .60 | .48 | 0.413 | −0.081 | 0.020 | 2/9 | **0.82** |
| `raw` spatial:8 | .61 | .47 | .80 | .40 | .34 | .49 | .60 | .70 | .51 | **0.547** | **+0.053** | 0.092 | 6/9 | 0.74 |
| `recon` spatial:8 | .57 | .54 | .82 | .39 | .23 | .42 | .62 | .74 | .51 | **0.537** | +0.042 | 0.15 | 5/9 | — |
| `stamp_bandpow` spatial:8 | .57 | .43 | .74 | .37 | .33 | .32 | .66 | .68 | .60 | **0.522** | +0.027 | 0.36 | 5/9 | 0.77 |

Best-val means for the same runs: 0.552, 0.548, 0.541, 0.469, 0.487, 0.613, 0.602 and
0.602.

- **`stamp_bandpow` holds up under a trained head.** At −0.025 against `raw` (p = 0.22)
  it matches the LDA probe's −0.028 gap. Code-space features that keep stamp attribution
  cost about 0.03, and the difference is not significant with 9 subjects.
- **The `z_chan` number is not a fair reading yet: the head memorizes.** Train is 0.99
  while val is 0.40. The head has 3884 parameters (a 64×8 = 512-wide readout) for 228
  trials, which is the known memorization pattern from the finetune caveats above. It
  needs a smaller or regularized projection (`z_proj` 2–4, weight decay, dropout) before
  it says anything about the encoder. The best-val mean (0.469) gives a sense of the
  ceiling.
  - **Shrinking the projection to `z_proj` 2** (974 head parameters) narrows the gap
    (train 0.82 vs val 0.41) and lifts val only slightly (0.395 → 0.413). The arm is
    still 0.08 below `raw` (p = 0.020) and still overfits. The time-mean `z` readout
    stays the weakest input even with a much smaller head. Before calling the encoder
    weak, B2 still has to test per-patch nonlinearity before the time-mean, and stronger
    regularization (weight decay, dropout).
- **The signed spatial filter helps `raw`:** +0.053, 6/9 wins, p = 0.092. That is the
  expected CSP-like gain, but not significant at n = 9. The largest gains are on the
  weak subjects (S6 0.30 → 0.49, S5 0.24 → 0.34), and S3 reaches 0.80. The train/val gap
  widens (0.74 vs 0.55).
- **The reconstruction keeps the spatial contrast.** `recon` spatial:8 vs `raw`
  spatial:8 is −0.011 (p = 0.59, 4/9). Against its own concat baseline, the filter adds
  +0.051 to `recon` (p = 0.093, 6/9), about the same as the +0.053 it adds to `raw`. The
  one clear loss is S5, which stays at chance with `recon` (0.23 vs 0.34).
- **The filter helps the code-space arm the same way.** `stamp_bandpow` spatial:8 gains
  +0.052 over its own concat baseline (p = 0.104, 6/9) and lands 0.026 below `raw`
  spatial:8 (p = 0.33, 2/9). The signed filter applied to each stamp's (a, b) across
  channels works, which is what experiment C's spatial stage assumes.
- **Step 4 (B1 channel) outcome:** the signed filter adds about +0.05 to every input
  tried (`raw`, `recon`, `stamp_bandpow`). Consistent, but not significant at n = 9.
  The three inputs stay within 0.03 of each other under the filter:

  | input | concat | spatial:8 |
  |---|---|---|
  | `raw` | 0.495 | 0.547 |
  | `recon` | 0.486 | 0.537 |
  | `stamp_bandpow` | 0.470 | 0.522 |
  | `z_chan` | 0.395 / 0.413 (overfits) | not run |

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

## Experiment C — one head for every EEG paradigm (proposed)

**Question.** One finetune approach for all EEG classification, which also answers
*which stamp, at which time* carries the task information: a map over the patch (N) and
stamp (S) axes.

**Why the stamp code is the input, not encoder `z`:**

- **It carries the task information.** Band-weighted, it sits about 0.03 below raw under
  both the LDA probe and a trained head. `z_chan` sits 0.08–0.10 below raw and
  memorizes.
- **Its per-channel values are physical.** Stamp s's contribution at channel ch,
  `a·D_s + b·H_s`, sums with the others into the reconstruction at that channel. So a
  signed spatial filter applied within each stamp is valid even though the v10 encoder
  mixed channels before the stamps, and its activation patterns are real topographies.
  That is how channel information is captured without losing stamp identity: filter
  channels inside each stamp, never pool stamps away first.
- **It is complex-valued**, `c[n, ch, s] = a + i·b`. That makes it a time × channel ×
  template decomposition, and each paradigm reads a different function of it:
  - **Induced (MI, ERD/ERS):** `|u|²` after spatial filtering. This is what the
    spatial:8 runs exploit.
  - **Evoked (ERP):** the signed complex value at a latency, not its power.
  - **Steady-state (SSVEP):** each patch's phase is measured from its own start, so a
    stimulus at f advances by `2π·f·stride` per patch. A complex linear filter over
    patches can match that and recover frequency far finer than the templates' 4 Hz
    bins. Measured below.

**Proposed head** (the same architecture for every task; only the weights and
`num_classes` change):

```
c[n, ch, s]   dense a+ib, alive routed + shared stamps, valid channels
 1. spatial:  K signed filters shared across stamps (applied to a and b alike)  -> u[n, k, s]
 2a. induced: log sum_n w_ind[s, n] |u|^2        nonneg time weights             -> [k, s]
 2b. evoked / steady-state: sum_n T[s, n] u      complex time filter -> re, im   -> [k, s, 2]
 3. [k, s, 3] -> dropout -> linear(num_classes)
```

**Size control is mandatory.** The naive version has K·S·3 ≈ 600 features for about 230
trials, which is the memorization regime `z_chan` just showed. Two constraints:

- low-rank time weights: `w[s, n] = Σ_r p_r[s]·q_r[n]`, r = 1–2;
- a group penalty over stamps, so the head selects few stamps. That selection is also
  the interpretable output.

The target is about 1k parameters, like the heads above.

**Interpretable outputs:**

- **time × stamp map:** `|T[s, n]|` and `w_ind[s, n]`, weighted by the classifier.
- **topographies:** Haufe activation patterns of the K filters, plus which stamps use
  each filter.
- **induced vs evoked, per stamp:** whether it contributes through branch 2a or 2b.

**Validation:**

- **Reduction check:** with 2b off and uniform time weights, the head must reproduce
  `stamp_bandpow` spatial:8 on BCICIV2a.
- **Paradigms:** then each paradigm against its own `raw` baseline — MI (BCICIV2a,
  EEGMMIdb), ERP (Inria), SSVEP (BETA).
- **Stated risks:**
  - shared stamps (about 70% of reconstruction energy) may dominate the importance map;
  - the stamp ERD sign disagreed with raw, so read *which stamp, when*, not the
    direction of the change.

### Prerequisite result — SSVEP phase advance (BETA_4s)

Script: `phase_probe_beta.py`. Subjects 19–30, which v10 small never saw (1920 trials,
40 classes 8.0–15.8 Hz, chance 0.025). Stimulus window 0.64–3.5 s, 8 occipital
channels. `z_s = Σ_{n,ch} c[n+1]·conj(c[n])` per stamp. The 4 stamps peaking at 8–16 Hz
vote for f; the 4 peaking at 16–32 Hz vote for 2f.

| readout | acc | note |
|---|---|---|
| stamp phase advance, `−` orientation | 0.109 | fundamental stamps only: 0.111 |
| stamp phase advance, `+` orientation | 0.025 | chance: wrong direction |
| raw PSDA (power at f + 2f, 3 s Hann) | 0.529 | standard baseline |

- **The phase advance is in the code.** The class-mean observed advance matches the
  expected `2π·f·0.125 s` with circular agreement **0.949** (1 = perfect; random ≈ 0.16).
  The direction is reversed (`−`) because the quadrature partner uses a −90° convention.
- **Single trials are noisy.** The fixed readout (occipital sum, no learned weighting,
  4 stamps) reaches 4.4× chance, far below raw PSDA. Branch 2b is therefore worth
  building, but a raw-level SSVEP result is not shown. A learned spatial + temporal
  filter has to close that gap.
- **Not measured yet:** stimulus-locked ITC per class, and the same readout on
  `recon`/`raw`.

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
