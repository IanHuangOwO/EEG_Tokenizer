# 0014 — Finetune head: what to expose, how to pool, compared to raw (draft)

Status: Proposed (draft). Experiments A (features) and B (pooling) are measured on
`mesae_v10_small_uw01`; experiment C (one head for every paradigm) is specified, with its
SSVEP prerequisite measured, and not built. Implements the direction decided in ADR 0012.
Loss trimming (ADR 0015) waits until this is settled.
Date: 2026-09-17 (results through 2026-09-18)
Scripts: `probes/` (see its README)

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

## Experiment C — one head for every EEG paradigm (proposed, revised after A and B)

**Question.** One finetune approach for all EEG classification, which also answers
*which stamp, at which time* carries the task information: a map over the patch (N) and
stamp (S) axes.

### What A and B settled, and what they never varied

Settled:

- **Three inputs land within 0.03 of each other** once the channel filter is on: `raw`
  0.547, `recon` 0.537, `stamp_bandpow` 0.522. The codes are not the bottleneck.
- **Channel contrast is the only pooling choice tested so far that mattered:** about
  +0.05 on all three inputs.
- **Head size decides generalization:** 772-parameter heads generalize, 3884 memorize.
  Every new feature has to fit that budget.
- **Encoder `z` is the weak input** under a linear time-mean readout.

Never varied — this is what C is actually for:

1. **Time.** Every run in A and B collapsed the whole trial into one number per feature.
   The stamp analysis found the 0.5–2.5 s window beats 0.5–4 s, and raw beta
   lateralization is significant only early (0.5–2.5 s) and gone by 2.5–3.5 s. "When" is
   untested and is the cheapest remaining factor.
2. **Phase.** Every feature so far is power; `b` only ever enters through `a²+b²`. That
   hides ERP deflections, SSVEP frequency, and — for MI too — **sub-bin frequency
   resolution**: the phase advance locates a rhythm inside a stamp's 4 Hz bin, which is
   where individual mu peaks differ between subjects and states.
3. **Frequency resolution.** A and B reduce everything to two bands. Keeping the stamp
   axis gives ~25 stamp-shaped bands. Untested whether that helps.
4. **Cross-stamp structure.** `stamp_bandpow` drops the cross terms and sits 0.036 below
   `recon` in the probe. That gap *is* the cross-stamp part.
5. **Per-stamp features at all.** Everything so far summed the stamps away before the
   classifier.

Two blind spots that matter more than the head itself:

6. **Within-subject with plenty of data is the regime where a foundation model has the
   least to offer.** 228 training trials is enough for raw band power + LDA to be near
   optimal. A **few-shot curve** (5/10/20/50 trials per class) against the same raw
   baseline is a stronger test of the tokenizer than anything in A or B, and it is cheap.
7. **Transfer was never tested.** Every number is a per-subject model. A shared
   dictionary should pay off across subjects, where per-subject spatial filters do not
   transfer. ADR 0012 puts cross-subject at 0.27–0.31 for every feature, which is exactly
   why a real gain there would mean something.

Method fix: A/B head runs used one 80/20 split per subject (60 val trials) while the
probe used 5-fold CV, so 0.495 and 0.482 are not strictly comparable. C uses 5-fold CV
per subject, which also shrinks the noise that leaves +0.05 non-significant at n = 9.

### Why the stamp code is the input, not encoder `z`

- **It carries the task information.** Band-weighted, it sits about 0.03 below raw under
  both the LDA probe and a trained head. `z_chan` sits 0.08–0.10 below raw and memorizes.
- **Its per-channel values are physical.** Stamp s's contribution at channel ch,
  `a·D_s + b·H_s`, sums with the others into the reconstruction at that channel. So a
  signed spatial filter applied within each stamp is valid even though the v10 encoder
  mixed channels before the stamps, and its activation patterns are real topographies.
  That is how channel information is used without losing stamp identity: filter channels
  inside each stamp, never pool stamps away first. B confirmed the filter works on the
  code arm (+0.052).
- **It is complex-valued**, `c[n, ch, s] = a + i·b`: a time × channel × template
  decomposition, where each paradigm reads a different function of the same tensor.

### Head

```
c[n, ch, s]   dense a+ib, alive routed + shared stamps, valid channels
 1. spatial:   K signed filters per stamp (applied to a and b alike)      -> u[n, k, s]
 2a. induced:  log sum_n w[s,n] |u|^2          time weights, low-rank     -> [k, s]
 2b. evoked:   sum_n T[s,n] u                  complex time filter -> re, im
 2c. advance:  sum_n u[n+1] conj(u[n])         sub-bin frequency / rhythm steadiness
 2d. coupling: sum_n u[n,k,s1] conj(u[n,k,s2]) selected stamp pairs only (optional)
 3. features -> dropout -> linear(num_classes)
```

`2c` is new relative to the first draft: the BETA probe showed the advance is real, and
it is not SSVEP-specific — it measures how steady a rhythm is and where it sits inside
the band. `2d` is the only term that can close the `stamp_bandpow` → `recon` gap.

**Size control is mandatory.** The naive version has K·S·3 ≈ 600 features for ~230
trials, the regime where `z_chan` hit train 0.99 / val 0.41. Two constraints:

- low-rank time weights, `w[s,n] = Σ_r p_r[s]·q_r[n]`, r = 1–2;
- a group penalty over stamps, so the head keeps few stamps — that selection is also the
  interpretable output.

Target ~1k parameters, like the heads that generalized.

### Ablation ladder (one factor per run)

| step | adds | must beat |
|---|---|---|
| C0 | induced only, flat time weights | reproduce `stamp_bandpow` spatial:8 = 0.522 (wiring check) |
| C1 | learned time weights | C0 — tests "when" |
| C2 | per-stamp features instead of band sums | C1 — tests frequency resolution |
| C3 | phase advance (2c) | C2 — tests sub-bin frequency |
| C4 | evoked branch (2b) | needed for ERP; likely neutral on MI |
| C5 | cross-stamp coupling (2d) | tests the 0.036 recon gap |

C3's "must beat" bar above is written as C2 per the original ladder design, but C2 was
answered by existing data without a dedicated run (build-order step 9) — C3's operative
bar is C1's 0.536 (build-order step 10).

### Regime tests (on the best of C0–C4)

| test | why |
|---|---|
| few-shot: 5 / 10 / 20 / 50 trials per class | where a foundation model should win |
| leave-one-subject-out | transfer, the other place it should win |
| EEGMMIdb (109 subjects) | statistical power; 9 subjects cannot settle ±0.05 |
| Inria (ERP), BETA (SSVEP) | universality, each against its own raw baseline |

Plus two ablations for the known confounds: run once with shared stamps excluded (they
hold ~70% of reconstruction energy and are selected in nearly every patch), and inspect
the Haufe patterns to check the filters look like motor topographies rather than noise.

### Interpretable outputs

- **time × stamp map:** `|T[s,n]|` and `w[s,n]`, weighted by the classifier.
- **topographies:** Haufe activation patterns of the K filters, and which stamps use each.
- **induced vs evoked vs steady, per stamp:** which branch the classifier leans on.

### Predictions (recorded so they can be wrong)

- C1 and C2 give small MI gains, about +0.02 to +0.04.
- C3 helps SSVEP substantially, MI slightly.
- **Within-subject with full data, the codes will not beat raw** — they stay within ~0.03,
  as A and B already show.
- **If the tokenizer wins anywhere, it is few-shot and cross-subject.** If it loses there
  too, the honest conclusion is that this is a good codec and not yet a foundation model,
  and the next lever is the pretraining objective (ADR 0015) or unfreezing the backbone,
  not the head.

### Risks

- The SSVEP gap is large: the fixed readout gets 0.109 against raw PSDA 0.529, so the
  learned filter has a lot to close. This is the step most likely to fail.
- Shared stamps may dominate the importance map.
- Stamp power's ERD *sign* disagrees with raw, so read *which stamp, when*, never the
  direction of the change.
- The backbone is `mesae_v10_small_uw01` (3 subjects per dataset); any conclusion about
  the tokenizer itself needs the full-data run.

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

## Protocol

- **Dataset:** BCICIV2a, `split_mode: intra_subject` — 9 per-subject models, class-stratified.
- **Backbone frozen.** Unfreezing is the last resort (ADR 0012).
- **Splits:** A/B used one 80/20 split per subject (228 train / 60 val). **C uses 5-fold
  CV per subject**, which makes head numbers comparable to the probe and shrinks the
  noise that leaves ±0.05 non-significant at n = 9.
- **Optimizer (all trained heads):** lr 1e-2 → 1e-3 cosine, 100 epochs, warmup 2.
  Dropout started at 0 (A/B, and C0's original attempt); the overfitting diagnosed on
  the 200-feature `stamp_induced` head (step 7, follow-up (b)) established `dropout: 0.5`
  for the stamp_induced-family runs, current setting from follow-up (b) onward (follow-up
  (b), follow-up (c), C1, and every step after). The repo finetune defaults (lr 1e-3,
  50 epochs, dropout 0.3) under-train a linear head: 0.418 vs 0.495 on the same `raw`
  features.
- **Reporting:** per-subject `balanced_acc` (equals accuracy on the balanced val sets)
  and kappa; the headline number is each subject's mean over the last 10 epochs
  (tail-mean), with best-val-epoch quoted separately as the optimistic reading.
  Comparisons are single, uncorrected paired t-tests across the 9 subjects (each
  subject's own CV folds averaged together first — folds are not independent samples,
  never pair/test on raw per-fold rows; see `probes/ft_summary.py`). An earlier draft of
  this line said "corrected for multiple comparisons" — that doesn't describe what's
  actually run (one paired comparison per build-order step, not a batch of comparisons
  needing a multiple-comparisons correction); removed.
- **Acceptance-comparison methodology (C1 onward):** the bar for "step N beats step
  N−1" is a pre-declared point improvement in tail-mean `balanced_acc`, reported
  alongside its n = 9 significance statistics for transparency — but statistical
  significance is never treated as a pass/fail gate at this sample size. Single-factor
  experiment-C ablation steps (C1, C2, C3, ...) are underpowered by construction at
  n = 9 subjects; demanding significance before advancing would stall the ladder
  indefinitely, so each step reports its numbers honestly (as step 8 does for C1) and
  moves on rather than gating on p-values.
- **`inter_subject` / LOSO numbers are reported separately**, never mixed with
  intra-subject numbers.
- **Backbone:** `mesae_v10_small_uw01` for plumbing. Conclusions about the tokenizer
  itself need the full-data run.

## Build order (each step has one acceptance check)

Done:

1. **A — feature probe** (LDA, fixed pooling). `raw` 0.482, `recon` 0.490,
   `stamp_bandpow` 0.454, `chan_mag` 0.359, `z_chan` 0.341.
2. **B baseline `raw`** — 0.495, matches the probe: the head is wired correctly.
3. **B baseline `recon`** — 0.486, within noise of `raw`: the recon path is lossless here.
4. **B1 channel (spatial:8)** — about +0.05 on `raw` (0.547), `recon` (0.537) and
   `stamp_bandpow` (0.522). Consistent, not significant at n = 9.
5. **B feature arms** — `stamp_bandpow` −0.025 vs `raw` (not significant); `z_chan`
   −0.10 and memorizing, `z_proj` 2 only partly helps (0.413, train 0.82).
6. **C prerequisite — SSVEP phase advance** (`probes/phase_probe_beta.py`): present at
   class level (circular agreement 0.949), weak per trial (0.109 vs raw PSDA 0.529).

Next, in order:

7. **C0 — induced branch only, flat time weights.** Original acceptance check: must
   reproduce `stamp_bandpow` spatial:8 (0.522), a miss being "a wiring bug, not a design
   result." **Attempted, not passed:** `intra_subject_cv`, 9 subjects × 5-fold, tail-mean
   balanced_acc **0.456** — a miss, not noise: 7 of 9 subjects fall below 0.522, not
   scattered evenly around it. Per-subject tail means: S1 0.458, S2 0.468, S3 0.486,
   S4 0.396, S5 0.424, S6 0.328, S7 0.532, S8 0.572, S9 0.440.

   A final review dug into the miss and found the "wiring bug" framing itself was wrong,
   on three counts:

   - `stamp_induced` is not under-capacity relative to `stamp_bandpow` here — it's
     *over*-capacity. Its feature width is `K * len(alive_stamps)` = 8 filters × 25
     stamps (21 alive routed + 4 shared) = **200 features**, against `stamp_bandpow`'s
     `K * len(BANDS)` = 8 × 2 = **16 features** — a 12.5x widening, trained on 230
     trials/fold with `dropout: 0`, `weight_decay: 0.01` (hyperparameters tuned in
     experiment B for the 16-feature head, never revisited for this one).
   - The training log shows textbook overfitting, not a wiring failure: final-epoch train
     `balanced_acc` reaches **1.000** (vs. B's `stamp_bandpow spatial:8` run's final train
     `balanced_acc` of **0.79**), while val loss rises across the run instead of settling.
   - C0's best-single-epoch val mean is **0.527** — essentially equal to the 0.522 target.
     The feature itself carries the same class-relevant information `stamp_bandpow` does;
     the entire 0.066 gap in the tail (last-10-epoch) metric is the overfitting tail, not
     a missing signal.

   The acceptance criterion's assumed equivalence was also mathematically wrong,
   independent of how well this run trained. The plan (and a comment/docstring in
   `model/MeSAE/MeSAE.py`) claimed summing `stamp_induced`'s per-stamp powers over each
   band via the template energy tables (`E_D`/`E_H`) "would give back exactly the
   `stamp_bandpow` feature." False once the `log` is accounted for: `stamp_induced`
   computes `log(power)` per stamp before any band-summing could happen, and
   `log(sum_s w_s * p_s) != sum_s w_s * log(p_s)` in general — log doesn't distribute over
   a weighted sum. Even a perfectly-implemented, perfectly-trained `stamp_induced` head
   was never guaranteed to reproduce `stamp_bandpow`'s exact number; the two features are
   related but not nested/equivalent, so "a miss is a wiring bug" was never a valid test
   design.

   **Net:** the evidence points at head capacity/regularization mismatch, not a wiring
   bug. Still not marked done — whether `stamp_induced` matches `stamp_bandpow` once
   properly regularized is an open question. Recommended next steps, not yet run:

   (a) Re-run the existing `stamp_bandpow spatial:8` config under `split_mode:
   intra_subject_cv` (no code change, just a config swap) as a protocol-only control, to
   confirm the CV-vs-80/20 split change alone isn't the cause — predicted to land near
   0.52, since train-set size is identical between the two protocols (230 trials either
   way).

   (b) Re-run C0 (`stamp_induced`) with regularization matched to its larger 200-feature
   width (non-zero dropout, and/or stronger weight decay, and/or fewer epochs/early
   stopping) before concluding anything about `stamp_induced` as a feature.

   This blocked C1–C5 pending resolution; resolved by follow-up (c) below, which
   establishes a clean, fold-count-matched baseline that C1 (step 8) was then run
   against — C1–C5 are no longer blocked. Full per-subject/per-fold numbers:
   `.superpowers/sdd/2026-09-19-experiment-c-c0/task-3-report.md` (gitignored, run-local).

   **Follow-up (a) — protocol-only control, run.** `stamp_bandpow spatial:8` under the
   same `intra_subject_cv` 5-fold protocol (`dropout: 0`, everything else identical to
   the B baseline): tail-mean balanced_acc **0.476**. Per-subject tail means: S1 0.532,
   S2 0.476, S3 0.591, S4 0.384, S5 0.274, S6 0.374, S7 0.579, S8 0.637, S9 0.435. This
   is *not* near the predicted 0.52 — an unpredicted, important result in its own right,
   not a null one. With the identical known-good feature and identical per-fold
   train-set size (230 trials), swapping only the split protocol (80/20 → 5-fold
   intra-subject CV) already costs ~0.046 off the original 0.522 baseline. So the
   `intra_subject_cv` protocol itself is a confound, not just `stamp_induced`'s capacity
   — the capacity-only story for C0's original 0.066 gap (0.522 → 0.456) only explains
   the remaining ~0.020 of it (0.476 → 0.456, `stamp_bandpow`-CV vs `stamp_induced`-CV,
   same protocol). Log: `output/mesae_finetune_c0_control_bandpow_cv/artifacts/train_20260919_082029.log`.

   **Follow-up (b) — regularized C0 rerun, run.** `stamp_induced spatial:8`, `dropout:
   0.5`, **3-fold** CV — reduced from 5 for turnaround time (a mid-task scope decision,
   made once follow-up (a)'s result was in), so this number is *not* fold-count-comparable
   to the original C0 run or to follow-up (a), both 5-fold; treat it as a directional
   read, not a like-for-like replacement. Tail-mean balanced_acc **0.468**. Per-subject
   tail means: S1 0.525, S2 0.424, S3 0.507, S4 0.366, S5 0.382, S6 0.407, S7 0.564, S8
   0.596, S9 0.444 — a small +0.012 over the original `dropout: 0` C0 result (0.456),
   still short of follow-up (a)'s 0.476 and further short of 0.522. Final-epoch train
   `balanced_acc`, meaned over all 27 fold×subject runs, dropped from the original run's
   0.985 (range 0.965–1.000 across its 45 fold×subject runs) to **0.748** (range
   0.604–0.854) — dropout clearly worked as regularization, roughly halving the
   train/val gap, but that reduction in overfitting barely moved validation performance.
   Log: `output/mesae_finetune_c0_dropout05/artifacts/train_20260919_120349.log`.

   **Net across both follow-ups.** The evidence does not support a clean "C0 is sound,
   just needed regularization" story. Two things are true at once: (1) a meaningful
   share of the original 0.066 gap (~0.046 of it) traces to the `intra_subject_cv`
   protocol itself, not to `stamp_induced`'s capacity, so 0.522 was never an
   apples-to-apples number to reproduce once the protocol changed; and (2) matching
   dropout to `stamp_induced`'s width substantially fixed the overfitting symptom
   (train `balanced_acc` 0.985 → 0.748) without closing the validation gap
   (0.456 → 0.468, still below both follow-up (a)'s 0.476 protocol-matched floor and
   0.522). Overfitting genuinely went down while validation performance stayed flat —
   that combination is a more concerning signal than a pure capacity story would predict;
   it points at something beyond head width/dropout alone (e.g. `stamp_induced`, once
   regularized enough to stop memorizing, may simply carry less separable signal than
   `stamp_bandpow` under this protocol, or `dropout: 0.5` traded overfitting for
   underfitting without landing on a better middle ground). Follow-up (b)'s 3-fold count
   is also a genuine confound on top of this (less training data per fold than 5-fold,
   noisier per-subject estimates), so this reading is provisional pending a real 5-fold
   rerun of the regularized config, which is a further follow-up, not done here.

   **Superseded by follow-up (c) below:** that 5-fold rerun lands at 0.495 —
   *above* follow-up (a)'s 0.476 protocol-matched floor, not below it. The “barely
   moved” reading above turns out to be an artifact of follow-up (b)'s 3-fold confound,
   not a stable conclusion; treat follow-up (c)'s number as the current baseline and
   this paragraph as superseded history, not the final word on dropout 0.5.

   This is evidence, not a verdict. Whether 0.522 was ever a valid target to reproduce
   exactly (see the log-of-sum-vs-sum-of-logs point above) and what to do about the
   dropout/underfitting tradeoff are open questions for whoever scopes C1 next, not
   settled here. Step 7 stays out of "Done."

   **Follow-up (c) — missing 5-fold/`dropout 0.5` baseline, run (ADR 0014 Task 2).**
   Neither prior follow-up left a fold-count-matched `dropout 0.5` number: (a) used
   `dropout 0`, (b) used 3-fold. This run is `stamp_induced spatial:8`, `dropout 0.5`,
   full 5-fold CV, otherwise identical to the original C0 run. Tail-mean balanced_acc
   **0.495**. Per-subject tail means: S1 0.575, S2 0.453, S3 0.562, S4 0.411, S5 0.387,
   S6 0.355, S7 0.573, S8 0.665, S9 0.476. Final-epoch train `balanced_acc` mean
   **0.736** (range 0.626–0.848 across all 45 fold×subject runs) — dropout is still
   suppressing overfitting at 5-fold, similar in degree to follow-up (b)'s 3-fold 0.748.
   Log: `output/mesae_finetune_c0_dropout05_5fold/artifacts/train_20260919_153852.log`.

   All four `intra_subject_cv` numbers are now on record, settings labeled: original C0
   (`stamp_induced`, `dropout 0`, 5-fold) **0.456**; follow-up (a) (`stamp_bandpow`,
   `dropout 0`, 5-fold, protocol-only control) **0.476**; follow-up (b) (`stamp_induced`,
   `dropout 0.5`, 3-fold) **0.468**; follow-up (c) (`stamp_induced`, `dropout 0.5`,
   5-fold) **0.495**. The regularized 5-fold number is the highest of the three
   `stamp_induced` numbers and is the correct, fold-count-matched baseline for C1
   (build-order step 8) to beat.

   **Evidence standard, applied consistently:** none of the four numbers above
   (0.456/0.468/0.476/0.495) has been significance-tested at n = 9. Step 8 below runs a
   paired t-test on its own +0.040 comparison and finds it not significant; the
   similar-magnitude differences discussed earlier in this step — the ~0.046 split-protocol
   cost (follow-up (a) vs the original 0.522 target) and the ~0.020 capacity residual
   (follow-up (a) vs original C0) — were never tested and plausibly sit within the same
   noise band step 8's analysis establishes. None of these point differences should be
   read as more established than step 8's own (correctly hedged) result.

8. **C1 — learned time weights.** First test of "when"; the stamp analysis and the raw
   beta lateralization both point at 0.5–2.5 s.

   **Run, `stamp_induced spatial:8`, `pool_time=learned:2`, `dropout 0.5`, 5-fold CV**
   (ADR 0014 Task 2 — low-rank, rank-`R=2`, softmax-weighted time pooling replacing
   follow-up (c)'s flat time mean; everything else identical to follow-up (c)). Tail-mean
   balanced_acc **0.536** — beats follow-up (c)'s 0.495 baseline by +0.040, satisfying the
   plan's literal acceptance bar ("C1 must beat C0"). Per-subject tail means: S1 0.591,
   S2 0.486, S3 0.693, S4 0.368, S5 0.366, S6 0.310, S7 0.605, S8 0.667, S9 0.733.
   Per-subject diffs (C1 − follow-up (c)): S1 +0.016, S2 +0.033, S3 +0.131, S4 −0.043,
   S5 −0.021, S6 −0.045, S7 +0.032, S8 +0.002, S9 +0.257 — **paired t-test across
   subjects, n = 9** (this ADR's own convention, Protocol section): mean diff +0.040,
   t = 1.25, **p ≈ 0.25, not significant**. Wins 6/9 subjects (S1, S2, S3, S7, S8, S9),
   regresses on 3 (S4, S5, S6), and the gain is concentrated in two subjects (S3 +0.131,
   S9 +0.257 account for most of the +0.040 mean); the other four winning subjects move
   only +0.002 to +0.033. This is not a broad, consistent effect — it is directionally
   positive and meets the plan's beat-the-baseline bar, but not statistically
   distinguishable from noise at n = 9. (A fold-level paired t-test over all 45
   fold×subject rows, as `probes/ft_summary.py` computes by default, gives the same mean
   diff +0.040 but p = 0.022 and "wins 27/45" — that treats each subject's 5 folds as
   independent samples, which they are not, so it is pseudoreplicated and not the
   document's unit of analysis; noted here only to flag it as the wrong statistic, not as
   a corroborating second result.)

   The learned time head (`head.time.p` shape `[R, S]` + `head.time.q` shape `[R, N']`)
   contributes **128 parameters** at this run's shape (`R=2`, `S=25` alive stamps,
   `N'=39` patches — checked directly off the saved checkpoints' state dict, consistent
   across all 45 fold×subject runs). Task 1's smoke check reported 188 params, but at a
   different shape on both axes (`S=64`, all stamps forced alive; `N'=30`, a synthetic
   trial length) — not directly comparable to this run's 128, both `S` and `N'` differ.
   Final-epoch train `balanced_acc` mean **0.843** (range 0.709–0.944) vs. follow-up
   (c)'s **0.736** (range 0.626–0.848) — both observed values, and both stay well below
   the original unregularized C0 run's ~0.985. C1's train accuracy being higher than the
   baseline's, while still well short of the unregularized run's near-ceiling fit, is
   consistent with `dropout 0.5` still doing meaningful regularization in both runs; it
   does not by itself establish that C1's extra capacity is being used "productively" as
   opposed to some milder, still-regularized increase in memorization that happens to
   also correlate with the (non-significant) validation gain — these numbers can't
   distinguish those two stories, so no such interpretive claim is made here.
   Log: `output/mesae_finetune_c1_learned2/artifacts/train_20260919_192402.log`.

   **Weight decay note.** `train_finetune.py` builds its `AdamW` param groups without
   exempting low-dimensional parameters from weight decay, so `weight_decay: 0.01` also
   shrinks `head.time.p`/`head.time.q` toward zero — and near-zero values there mean
   near-uniform (flat) time weights, i.e. a pull toward C0's baseline, not away from it.
   This is conservative: it means C1's measured +0.040 is if anything understated, not
   inflated, by this detail. Worth knowing before tuning `learned:R` further (e.g. an
   `R=1` comparison); not changed here.

   **What the learned weights learned ("when"), reproduced from the saved checkpoints.**
   Aggregating `head.time.p`/`head.time.q` across all 45 fold×subject checkpoints
   (`w[s,n] = softmax_n(sum_r p[r,s]*q[r,n])`, the same formula the forward pass uses)
   into five 1-second bins over the ~5.0 s trial span (39 patches, `patch_stride=25` /
   `sample_freq=200` → 0.125 s/patch) gives mean softmax mass per bin ≈ **0.174 / 0.242 /
   0.222 / 0.191 / 0.171** against a per-bin uniform baseline of ≈0.18–0.21 (bins hold
   7-8 patches each) — a mild early/middle tilt, not a sharp lock. 56.5% of the 1,125
   (stamp, checkpoint) rows have their peak weight inside the 0.5–2.5 s window that the
   raw beta lateralization and the stamp analysis point at (chance rate ≈40% by
   trial-duration fraction), median peak time 2.00 s. That is a real, if modest, answer
   to this step's nominal "when" question: the learned weighting leans toward early-to-mid
   trial content, consistent with but not sharply locked onto the 0.5–2.5 s window.

   **C1's tail-mean beats the baseline, but not at statistical significance (n = 9).**
   The result is directionally consistent with proceeding to the next build step —
   see step 9 below, which found C2 already answered by existing data, so that step is
   C3 (phase advance) in practice — but the concentration of the gain in 2 of 9
   subjects is an open question this run doesn't resolve — whoever scopes C3 should
   treat "does learned time weighting generalize" as still unsettled, not confirmed. Possible follow-up, not run
   here per this task's scope (one ablation factor per run): an `R=1` comparison, to see
   whether the second rank-2 factor is pulling its weight or whether a simpler `R=1` head
   gives a similar (or more consistent) effect. Step 8 stays out of "Done" — one run at
   one rank, without significance at n = 9, is a first result, not a settled one.
9. **C2 — per-stamp features** instead of the two band sums. **Already answered by
   existing data, no new run needed.** The ablation ladder defines C2 as "per-stamp
   features instead of band sums" — but `2a. induced`'s formula (`log Σ_n w[s,n]|u|²`
   → shape `[k, s]`) has been per-stamp, never band-summed, since C0's very first
   implementation (step 7). C0 and C1 (`stamp_induced`) already *are* the per-stamp
   arm; there is no band-summed version of `stamp_induced` to compare against by
   building something new. The ladder's "adds: per-stamp features" line was written
   before the Head pseudocode's `2a` formula was finalized as per-stamp-native, and
   was never reconciled with it once that decision was made — a planning
   inconsistency, not a build gap.

   The actual comparison C2 asks for (band-summed vs. per-stamp, everything else held
   fixed) already exists as a side effect of step 7's follow-ups, same protocol
   (`intra_subject_cv`, 5-fold, flat time weights, `dropout` aside — see caveat below),
   same checkpoint: follow-up (a) (`stamp_bandpow spatial:8`, 16 band-summed features,
   `dropout 0`) **0.476** vs. follow-up (c) (`stamp_induced`, 200 per-stamp features,
   `dropout 0.5`) **0.495** — per-stamp ahead by +0.019. This is not a clean
   single-factor ablation (dropout also differs between the two runs, `0` vs `0.5`,
   because `stamp_bandpow`'s 16-feature width never needed the regularization fix
   `stamp_induced`'s 200 features did), so treat +0.019 as a directional read, not a
   significance-tested result (not paired-t-tested here; same underpowered-by-
   construction caveat as everything else in this document). Consistent with step 7's
   `stamp_bandpow`-vs-`stamp_induced` probe-level finding (ADR 0012 / experiment A):
   per-stamp features have never measured worse than band sums anywhere in this
   document. C2 is not marked "done" in the sense of a dedicated run — it's marked
   answered, by data already on record, and the next new build step is C3.
10. **C3 — phase advance (2c).**

    **Run, `stamp_induced spatial:8`, `pool_time=learned:2`, `include_advance=true`,
    `dropout 0.5`, 5-fold CV** (ADR 0014 Task 2 — concatenates the real/imaginary parts
    of `Σ_n u[n+1]·conj(u[n])` per stamp onto the existing induced-power features,
    tripling head input width from `K·S` (200) to `K·S·3` (600); everything else
    identical to C1, same checkpoint `mesae_v10_small_uw01/checkpoint/last.pth`). Tail-mean
    balanced_acc **0.489**, **below C1's 0.536** — C3 does not beat C1, failing this
    step's acceptance bar. Per-subject tail means (C3, with C3−C1 diff): S1 0.527
    (−0.064), S2 0.437 (−0.050), S3 0.589 (−0.104), S4 0.360 (−0.009), S5 0.361 (−0.005),
    S6 0.334 (+0.024), S7 0.540 (−0.065), S8 0.607 (−0.061), S9 0.645 (−0.088) — a paired
    t-test across subjects, n = 9 (this ADR's own convention, Protocol section): mean diff
    **−0.047**, t = **−3.36**, **p = 0.010**. Unlike C1's result, this one is
    significant at n = 9 — but per the Protocol section's acceptance-comparison
    methodology, significance is reported for transparency, not as the pass/fail gate;
    the plain acceptance bar ("C3 must beat C1") already fails on the point estimate
    alone. Only 1 of 9 subjects improves (S6, +0.024, itself C1's weakest subject); the
    other 8 regress, five of them by more than 0.06 (S1, S3, S7, S8, S9 all regress,
    S3 the worst at −0.104). Log:
    `output/mesae_finetune_c3_advance/artifacts/train_20260920_002958.log`.

    **The overfitting check.** Final-epoch train `balanced_acc` mean **0.9255** (range
    0.874–0.983 across all 45 fold×subject runs) — compared against three prior numbers
    on record: C1's **0.843** (range 0.709–0.944, step 8), the C0-family 5-fold
    baseline's **~0.736–0.748** (follow-up (b)/(c), step 7/9), and the original
    unregularized C0 run's **~0.985** (step 7). C3's 0.9255 sits well above C1's 0.843
    and the ~0.74 baseline, and close to the unregularized run's ~0.985 ceiling — despite
    using the *same* `dropout: 0.5` as C1. The anticipated 600-feature overfitting risk
    this plan's own Architecture section named upfront (the `z_chan` precedent) did
    materialize: tripling the head's feature width pushed train accuracy back toward the
    unregularized regime that `dropout: 0.5` was originally introduced (step 7) to
    suppress, and validation performance regressed in the same run, significantly, at
    n = 9. Per the "Size control is mandatory" paragraph above, this `K·S·3` regime was
    meant to carry two constraints together — low-rank time weights and a group penalty
    over stamps — targeting ~1k head parameters total; this run inherited only the first
    (low-rank time weights, via C1's `pool_time=learned:2`), and no group penalty over
    stamps was ever implemented, here or in C0/C1 either, following the same "run first,
    diagnose honestly" precedent those steps set — but that precedent's cost is now
    visible in this specific failure. Measured directly off the saved checkpoints' state
    dict (same method as step 8's 128-parameter figure), the full trainable `head`
    sub-module holds **4,244 parameters** for C3 vs. **1,844** for C1 — roughly 4.2x and
    1.8x the ADR's own ~1k target, respectively. So this run tested `2c` outside the
    regularization regime the spec itself mandates for a feature width this large, with
    only one of its two required size-control constraints in place.

    **What this implies for `2c`'s premise.** The data available from this run cannot
    show that sub-bin phase/frequency information helps MI decoding on top of induced
    power and learned time weights — the observed effect is a significant *regression*
    in validation balanced_acc alongside a large rise in train accuracy toward the
    unregularized ceiling, i.e. the added branch's capacity was used to memorize rather
    than generalize, at this `dropout` setting. This run does not distinguish "phase
    advance carries no useful MI signal" from "phase advance could carry useful signal
    but needs stronger regularization (e.g. a group penalty, higher dropout, or reduced
    stamp count) to realize it without overfitting" — no such regularization was run
    here (deliberately, per this step's own scope: one new factor vs. C1). The numbers
    as they stand are a negative result for `2c` under C1's exact regularization
    settings, not a resolved verdict on the branch's information content.

    **Concrete disentangling follow-ups, cheapest first, not run here.** (a) Near-free,
    no GPU — recommended first: a regularized linear probe on the advance features alone
    vs. the induced features alone, matching `probes/probe_v10.py`'s existing per-subject
    `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')` methodology with matched
    dimensionality between the two arms. Runs in minutes on CPU and answers directly
    whether phase-advance carries class-separable MI signal at all, with zero
    head-capacity confound — no neural-net training involved, so this run's overfitting
    can't contaminate the answer. (b) One further ~3.5h GPU run, more expensive: a
    capacity-matched C3 rerun, either advance-only (drop induced power, keep just the
    `2·K·S` = 400-feature advance branch) or C3 with dropout tuned to bring train accuracy
    back down near C1's 0.843 (from C3's 0.9255) — either separates "the extra width
    itself hurt" from "phase-advance is genuinely uninformative." (a) is recommended over
    (b): it is both the cheaper option and the one that actually isolates the variable in
    question, rather than just reducing the capacity confound without eliminating it.

    Step 10 stays out of "Done" — one run, at one dropout setting, with no size-control
    variant tried, is a first result on `2c`, not a closed question.
11. **Regime tests on the best of C0–C4:** few-shot (5/10/20/50 trials per class), LOSO,
    then EEGMMIdb for statistical power. These decide whether the tokenizer is worth
    anything over raw. With C0–C4 all resolved (C2 without a dedicated run, C3 and C4 as
    negative results), "the best of C0–C4" currently means C1's exact configuration —
    `stamp_induced spatial:8`, `pool_time=learned:2`, `include_advance` unset/false,
    `evoked_rank` unset, `dropout 0.5`, tail-mean **0.536** — not whatever `config/config.json` happens to be
    pointed at when this step is picked up (it is currently pointed at C4's regressed
    `evoked_rank=2` config from this session's last run).
12. **C4 — evoked branch (2b)**, then ERP on Inria and SSVEP on BETA against their own
    raw baselines.

    **MI run of `2b` (first half of this step).** `stamp_induced spatial:8`,
    `pool_time=learned:2`, `evoked_rank=2`, `include_advance` unset, `dropout 0.5`,
    5-fold CV — C1's exact config plus the signed rank-2 evoked time filter on the code
    `(a, b)` (2·K·S = 400 extra features, head input width `K·S` (200) -> `K·S·3` (600);
    cross-checked against C1's saved `artifacts/config.json`, the only differences are
    `evoked_rank` and `model_name`). Tail-mean balanced_acc **0.466**, **below C1's
    0.536**. Per-subject tail means (C4, with C4−C1 diff): S1 0.461 (−0.130), S2 0.425
    (−0.061), S3 0.571 (−0.122), S4 0.382 (+0.014), S5 0.343 (−0.023), S6 0.320
    (+0.010), S7 0.476 (−0.129), S8 0.574 (−0.093), S9 0.641 (−0.092). Paired t-test
    across subjects, n = 9: mean diff **−0.070**, t = **−3.63**, **p = 0.007**, wins
    **2/9** (S4, S6, both by ≤ 0.014). Log:
    `output/mesae_finetune_c4_evoked/artifacts/train_20260920_122034.log`.

    **The overfitting check.** Final-epoch train `balanced_acc` mean **0.958** (range
    0.926–0.987 across the 45 fold×subject runs) vs. C1's **0.843**, C3's **0.9255**,
    the C0-family's **~0.74** and the unregularized run's **~0.985** — the highest of
    any regularized arm, essentially at the unregularized ceiling. Trainable `head`
    parameters (same method as step 10, off `fold0_subj_1/best_finetune.pth`):
    **4,372** vs. C1's 1,844 and C3's 4,244, about 4.4x the ~1k target. Same regime as
    C3: 600-feature head, `dropout 0.5`, no group penalty over stamps.

    **Reading.** C4 does not beat C1 and is a significant regression at n = 9, with a
    larger regression and higher train accuracy than C3. This is the outcome the ADR
    pre-declared as likely: `2b` was expected neutral on MI, and the stamp analysis
    found no phase-locked stamps on BCICIV2a, so there is no evoked signal for the
    branch to read while its extra 400 features add memorization capacity. This run
    measures `2b` **on MI**, not the branch's real test; it does not show the evoked
    branch is useless, only that it does not help MI at C1's regularization. The real
    next step is ERP on Inria and SSVEP on BETA against their own raw baselines, which
    needs an ERP task block the head does not have yet (`task="mi"` only). Step 12
    stays out of "Done" until that is run.

    **Weight decay note.** As in C1, `train_finetune.py`'s `AdamW` decays the 2-D
    `head.evoked.p`/`head.evoked.q` (`optimizer_param_groups` exempts only `ndim <= 1`),
    but the direction differs. The evoked filter is `T = 1/N' + p^T q`, so decay only
    shrinks the deviation and parks `T` at the constant `1/N'`, the plain trial-mean of
    `(a, b)`, the classic time-locked average. The 400 extra evoked features therefore
    cannot be decayed away, and decay does not make C4's regression conservative the
    way it made C1's gain conservative. Not changed here.

    **Width confound.** C3 and C4 both regress at the same 600-feature head width with
    the same dropout-only size control (no group penalty), so the ladder currently has
    a shared width confound rather than two independent negative results. Neither run
    separates "the branch carries no MI signal" from "600 features at dropout 0.5
    overfit".
12a. **Raw control under 5-fold CV.** The question C1 could not answer by itself: under
    one identical protocol, does the tokenizer's code beat the raw signal? Every step
    C0-C4 compared code-space heads only against each other under
    `split_mode: intra_subject_cv`; the only raw numbers in this ADR (`raw spatial:8`
    0.547, `raw concat` 0.495) came from the older single 80/20 split, and about 0.046
    of an earlier apparent miss was traced to that protocol change alone, so they are
    not comparable to C1's 0.536. **Run**, `input=raw`, `pool_channel=spatial:8`,
    `pool_time=trial`, `dropout 0`, `freeze_backbone` true, 5-fold CV — differs from
    follow-up (a) (`stamp_bandpow`, tail-mean 0.476) **only in `input`** (the raw arm
    computes mu/beta log band power per spatial filter, 2 bands x 8 filters = 16
    features; cross-checked against follow-up (a)'s saved `artifacts/config.json`, the
    only differences are `input` and `model_name`). Tail-mean balanced_acc **0.530**.
    Per-subject tail means: S1 0.618, S2 0.494, S3 0.680, S4 0.456, S5 0.304, S6 0.412,
    S7 0.602, S8 0.672, S9 0.530. Log:
    `output/mesae_finetune_raw_control_cv/artifacts/train_20260920_173955.log`.

    **Paired tests** (subject level, n = 9, folds grouped within subject). Raw beats
    the matched code-space head, follow-up (a) `stamp_bandpow`, by **+0.054**
    (**p = 0.001**, raw wins **9/9** subjects) — the single-factor comparison, so on
    this 16-feature head the raw signal reads better than the stamp band-power code.
    Raw beats the clean C0 baseline (0.495) by +0.034 (p = 0.087, raw wins 8/9). Versus
    C1, **C1 − raw = +0.006** (0.536 vs. 0.530), **p = 0.847**, C1 wins **4/9**
    subjects (S3, S5, S7, S9; S9 alone is +0.203, and S4 −0.088 and S6 −0.102 go the
    other way).

    **The overfitting check.** Final-epoch train `balanced_acc` mean **0.759** (range
    0.588–0.888 across the 45 fold×subject runs), below C1's 0.843 and near the
    C0-family's ~0.74. Trainable `head` parameters (same method as step 10, off
    `fold0_subj_1/best_finetune.pth`): **612** (8x64 spatial filter 512, plus the
    16-unit BatchNorm/linear stack 16 + 16 + 64 + 4), well under the ~1k target and
    C1's 1,844.

    **Reading.** Under the same protocol, C1's 0.536 **matches** the raw signal (0.530):
    a +0.006 point difference, well inside noise and far below the +0.05 sort of gap
    that would count, with a non-significant paired test at n = 9 (significance is
    reported, never a gate, per the Protocol section). The matched-input comparison
    goes against the stamp band-power code (−0.054). So the best code-space head, C1,
    recovers raw-level accuracy but does not exceed it, which is what the ADR predicted
    ("within-subject with full data the codes will not beat raw", Experiment C blind
    spot 6). Read narrowly: this is within-subject full-data, the regime where a
    foundation model was expected to have the least to offer, so it neither refutes nor
    supports the tokenizer. Few-shot and cross-subject (step 11) remain the real tests.
    Also, "raw" here is the simple mu/beta band-power head at the same 16-feature
    width, not a stronger raw baseline (e.g. the LDA or a deeper raw model).
13. **C5 — cross-stamp coupling (2d)**, which tests the 0.036 `stamp_bandpow` → `recon`
    gap.
14. **Stamp attribution measure 4** (occlusion) with the best code-space head, plus the
    shared-stamp ablation and the Haufe topography check.

Dropped from the earlier plan: **B2 (time) and B3 (stamp axis) as standalone runs** —
they are C1/C2 inside the one head, which keeps the stamp axis instead of pooling it
away first. The `z_chan` variants (per-patch nonlinearity before the mean, stronger
regularization) stay open but are not on the critical path: the code arms are ahead of
it by 0.10.

## Implementation notes

- **Config:** `model_params.MeSAE.finetune` gets
  - `input` (`raw | recon | z_chan | stamp_bandpow | chan_mag | head_z`),
  - `task` (`mi | erp | ssvep`),
  - `pool_channel` (`concat | spatial:K | softmax`),
  - `pool_time` (`trial | patch_mean | attention | window:lo-hi`),
  - `pool_stamp` (`energy | concat | linear | attention`).

  Pooling is its own set of keys, so a run only ever edits one of them. Experiment C
  adds `pool_time` variants (low-rank learned weights) and the branch switches
  (`induced`, `evoked`, `advance`, `coupling`).
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
- **SSVEP phase advance:** measured (experiment C). Still open: the same readout on
  `raw`/`recon`, and stimulus-locked ITC per class.
- **Probe scripts live in `probes/`** (`probe_v10.py`, `stamp_relevance.py`,
  `phase_probe_beta.py`, `ft_summary.py`); their feature caches are under
  `output/<run>/probes/`.
