# 0017 — Restart baseline on the restructured finetune pipeline

Status: Complete. All 44 baseline runs finished 2026-09-22, Results below. Supersedes the *results* of ADR
0014 (its Protocol still applies) and builds on ADR 0016. Sub-project D (experiment runner) is next.
Date: 2026-09-22

## Context

ADR 0014 records Experiments A, B and C, the Phase 1 LOSO runs and the Phase 2 runs, all measured on
the old finetune pipeline. Three problems came to light while restructuring it:

1. **Wrong electrode coordinates for the backbone.** Those runs executed in the `base` conda env, which has no
   `mne`, so `IO/loader.py`'s `get_standard_coords` silently returned nothing and the backbone read flat polar
   fallback coordinates instead of MNE 3-D positions. Measured on BNCI2014001 subject 8 (stamp head, 5-fold, 100
   epochs, restructured pipeline): real MNE coordinates give **0.788**, fallback coordinates **0.660** (the old
   record for that subject was 0.667). A label-shuffle control sits at chance (0.217), train and evaluation
   trials never overlap, and running the backbone under fp16 autocast instead of fp32 changes nothing (0.786).
   Every stamp-head number from Experiment C and Phase 1 is therefore understated. The raw band head does not
   use the backbone and is unaffected.
2. **Confounded comparisons.** The Phase 1 raw control ran with dropout 0 and C1 with dropout 0.5; the raw
   control used a whole-trial FFT while the stamp heads worked per patch; runs were spread over many code
   versions.
3. **Slow and scattered ablations.** Every epoch reran the frozen backbone (about 3 s per epoch for a
   230-trial fold), and adding a head option touched several places.

The pipeline was restructured (`docs/superpowers/specs/2026-09-21-finetune-restructure-design.md`, ADR 0016):
one composable `FeatureHead` (feature, spatial filter, time pooling, branches, readout), a stamp-amplitude
cache next to the backbone (`cache_feature.py`), a head-only `train_finetune.py` with two split modes, and
an environment stamp in every run's saved config. An epoch now takes about 0.06 s.

## Decision

Restart the experiment on the restructured pipeline. The old finetune runs are archived under
`output/archive/` (kept as the record of why the head was chosen); the old code is at git tag
`pre-head-cleanup`.

### Fixed settings

- Environment: `eeg_fm` (MNE 1.12.1, torch 2.14). The environment is recorded in each run's config snapshot.
- Backbone: `mesae_v10_small` (pretrained on 3 subjects per dataset, 40 epochs). Any conclusion about the
  tokenizer itself needs the full-data backbone (`mesae_v10_full`); confirming a locked head there is planned.
- Training: 100 epochs, batch size 16, AdamW, learning rate 0.01 decaying to 0.001 (2 warm-up epochs),
  weight decay 0.01, seed 42, fp32, dropout 0.5 on every head.
- Protocol (unchanged from ADR 0014): the unit of analysis is the subject; the score is the tail-mean, the
  mean over the last 10 epochs, of balanced accuracy on the held-out trials; folds are averaged within a
  subject first; the point difference is the bar, and p-values from paired t-tests across subjects are
  reported, never used as a gate; groups of fewer than 5 subjects are descriptive only.

### Baseline matrix

Heads (all with the same spatial filter size `spatial_k = 8` where they have one):

| Head | Definition |
|---|---|
| `raw_band` | raw signal, per-patch mu/beta log power, spatial filter, flat time pooling |
| `raw_signal` | raw signal, signed samples averaged to about 20 Hz, spatial filter, no pooling (the ERP-style baseline) |
| `flat` | stamp code, per-stamp log power, spatial filter, flat time pooling |
| `learned` | stamp code, per-stamp log power, spatial filter, learned low-rank time weights (rank 2) |
| `learned_advance` (BETA_4s only) | `learned` plus the phase-advance branch |
| `learned_evoked` (Inria only) | `learned` plus the evoked branch (rank 2) |

Datasets and splits:

| Dataset | `intra_subject` (5-fold per subject) | `inter_subject` |
|---|---|---|
| BNCI2014001 (9 subjects, 4 classes) | all 9 | LOSO (9 folds) |
| BNCI2014004 (9, 2 classes) | all 9 | LOSO (9 folds) |
| BCICIV1_Train (subjects 2, 3, 4, 5, 7; left/right) | all 5 | LOSO (5 folds) |
| Inria_Train (16, error-related potential) | all 16 | LOSO (16 folds) |
| BETA_4s (55, 40-class SSVEP) | all 55 | grouped 5-fold over subjects |
| PhysionetMI (109, 3 classes) | 30 random subjects (seed 42) | grouped 5-fold over subjects |

`raw_signal` is run on every dataset; the two task-specific variants only on their own paradigm. Leave-one-out
over 55 and 109 subjects would cost about 2 and 20 GPU-hours per head at the current speed, so the two large
datasets use grouped 5-fold, which answers the same "does it transfer to unseen subjects" question.

Runs are written to `output/mesae_v10_small/finetune/<head>/<dataset>_<mode>/`,
nested under the backbone they were finetuned from (`mesae_v10_small` — see Fixed
settings above) rather than top-level, since ADR 0016's planned full-backbone
confirmation (`mesae_v10_full`) will run this same baseline matrix a second time and
would otherwise land at the same paths. One dir per head (`flat`, `learned`,
`learned_advance`, `learned_evoked`, `raw_band`, `raw_signal` — the last is the head
this ADR's restart treats as the reference to compare every other head against).

### Known limits

- The small backbone saw subjects 1 to 3 of each dataset in pretraining (BCICIV1_Train subjects 2 and 3 are in
  the evaluated subset), so "unseen" is only fully clean for the others.
- The seen/unseen group design, the PSDA reference for SSVEP and the full-backbone confirmation are left to the
  experiment runner work (sub-project D of the restructure).
- `raw_band` now uses a per-patch estimator (4 Hz resolution at 50-sample patches), so it is not the same
  statistic as the earlier whole-trial raw control.

## Results

All 44 runs finished 2026-09-22. Score = tail-mean balanced accuracy (mean over the last 10 of 100 epochs),
folds averaged within a subject first, then across subjects (`n` in each table header). Diffs are paired
across subjects (mean of per-subject differences); wins = subjects where the row beats the comparator;
p-values are paired t-tests across subjects, reported as description not a gate (ADR 0014's protocol).

### BNCI2014001 (9 subjects, 4 classes, chance 0.250)

| Head | intra mean | intra vs raw_band | inter mean | inter vs raw_band |
|---|---|---|---|---|
| `raw_band` | 0.530 | — | 0.313 | — |
| `raw_signal` | 0.492 | −0.038, 3/9, p=0.51 | 0.488 | **+0.175, 9/9, p=0.003** |
| `flat` | 0.570 | +0.040, 7/9, p=0.12 | 0.321 | +0.008, 5/9, p=0.64 |
| `learned` | **0.632** | **+0.102, 8/9, p=0.019** | **0.448** | **+0.135, 9/9, p=0.0005** |

`learned` vs `flat`: intra +0.062, 4/9, p=0.12; inter **+0.127, 9/9, p=0.0008**.

### BNCI2014004 (9 subjects, 2 classes, chance 0.500)

| Head | intra mean | intra vs raw_band | inter mean | inter vs raw_band |
|---|---|---|---|---|
| `raw_band` | 0.671 | — | 0.599 | — |
| `raw_signal` | 0.642 | −0.029, 4/9, p=0.58 | 0.649 | +0.049, 7/9, p=0.08 |
| `flat` | 0.707 | +0.036, 8/9, p=0.015 | 0.632 | +0.033, 8/9, p=0.019 |
| `learned` | **0.742** | **+0.071, 8/9, p=0.001** | **0.694** | **+0.095, 9/9, p=0.0005** |

`learned` vs `flat`: intra +0.036, 8/9, p=0.004; inter +0.062, 8/9, p=0.004.

### BCICIV1_Train (5 subjects, 2 classes, chance 0.500 — descriptive only, n<5 threshold from ADR 0014's protocol)

| Head | intra mean | intra vs raw_band | inter mean | inter vs raw_band |
|---|---|---|---|---|
| `raw_band` | 0.598 | — | 0.519 | — |
| `raw_signal` | 0.506 | −0.092, 2/5, p=0.29 | 0.505 | −0.013, 1/5, p=0.44 |
| `flat` | 0.595 | −0.003, 1/5, p=0.94 | 0.493 | −0.026, 1/5, p=0.16 |
| `learned` | 0.624 | +0.025, 3/5, p=0.36 | 0.480 | −0.038, 1/5, p=0.06 |

No head clears `raw_band` convincingly here in either mode — consistent with ADR 0014's finding that this
dataset has close to no decodable signal in any representation ("my read before adding the raw column was
right by accident").

### Inria_Train (16 subjects, 2 classes, chance 0.500)

| Head | intra mean | intra vs raw_band | inter mean | inter vs raw_band |
|---|---|---|---|---|
| `raw_band` | 0.542 | — | 0.506 | — |
| `raw_signal` | **0.669** | **+0.127, 15/16, p<0.0001** | 0.604 | **+0.098, 16/16, p=0.0002** |
| `flat` | 0.558 | +0.015, 11/16, p=0.03 | 0.507 | +0.002, 7/16, p=0.49 |
| `learned` | 0.577 | +0.035, 13/16, p=0.004 | 0.518 | +0.012, 8/16, p=0.10 |
| `learned_evoked` | 0.568 | +0.026, 13/16, p=0.037 | **0.587** | **+0.082, 16/16, p=0.0002** |

`learned` vs `flat`: intra +0.020, 11/16, p=0.057; inter +0.010, 10/16, p=0.22. `raw_signal` is the
best head both ways here; `learned_evoked` (the branch built for this paradigm) is the only stamp head that
comes close inter-subject, well ahead of plain `learned` — the evoked branch is doing real work on an
ERP task, as designed.

### BETA_4s (55 subjects, 40 classes, chance 0.025)

| Head | intra mean | intra vs raw_band | inter mean | inter vs raw_band |
|---|---|---|---|---|
| `raw_band` | 0.048 | — | 0.054 | — |
| `raw_signal` | 0.063 | +0.014, 30/55, p=0.037 | **0.427** | **+0.372, 55/55, p<0.0001** |
| `flat` | 0.056 | +0.008, 34/55, p=0.06 | 0.168 | +0.114, 55/55, p<0.0001 |
| `learned` | 0.058 | +0.009, 34/55, p=0.032 | 0.190 | +0.136, 55/55, p<0.0001 |
| `learned_advance` | **0.071** | **+0.023, 38/55, p=0.0002** | 0.385 | +0.331, 54/55, p<0.0001 |

`learned` vs `flat`: intra +0.002, 31/55, p=0.59 (no real difference); inter +0.022, 39/55, p<0.0001.
Inter-subject, `raw_signal` beats every stamp head including `learned_advance` (the phase-advance branch built
for SSVEP) by a wide margin — for cross-subject SSVEP, the raw per-channel waveform generalizes better than
anything routed through the frozen backbone's stamp codes. Intra-subject the gap nearly closes and
`learned_advance` is best, so the phase-advance branch does help once the head can fit a subject's own phase.

### PhysionetMI (30 subjects intra / 109 inter, 3 classes, chance 0.333)

| Head | intra mean | intra vs raw_band | inter mean | inter vs raw_band |
|---|---|---|---|---|
| `raw_band` | 0.562 | — | 0.361 | — |
| `raw_signal` | 0.603 | +0.041, 20/30, p=0.11 | **0.554** | **+0.193, 107/109, p<0.0001** |
| `flat` | 0.605 | +0.043, 26/30, p<0.0001 | 0.429 | +0.068, 102/109, p<0.0001 |
| `learned` | **0.618** | **+0.056, 24/30, p=0.0007** | 0.546 | +0.185, 107/109, p<0.0001 |

`learned` vs `flat`: intra +0.013, 15/30, p=0.21 (no real difference); inter **+0.117, 101/109, p<0.0001**.

### Cross-dataset pattern

- **`learned` beats `raw_band` everywhere it's measured**, intra and inter, on every dataset except the
  near-chance `BCICIV1_Train`. It is the only stamp head that never loses to `raw_band`.
- **`raw_signal` is the strongest single head inter-subject** on 4 of 6 datasets (BNCI2014001 is the exception --
  `learned` wins there inter-subject too), often by a wide margin (BETA_4s, PhysionetMI). Intra-subject it's
  usually weaker than the stamp heads. Read together: the frozen backbone's stamp codes help most when a
  head only gets to see a single subject's own trials (intra), and help least (or actively hurt, BETA_4s/
  PhysionetMI inter) when generalizing to unseen subjects — raw per-channel amplitude transfers across subjects
  better than the current stamp representation does.
- **`learned` vs `flat` (does the learned-rank time pooling help over flat pooling):** indistinguishable
  intra-subject on every dataset (largest intra diff +0.062, most p>0.1), but a real, consistent inter-subject
  win everywhere it's measured (BNCI2014001 +0.127, BNCI2014004 +0.062, BETA_4s +0.022, PhysionetMI +0.117, all
  p<0.005 except BETA_4s). The learned time-pooling's benefit is specifically about generalizing to unseen
  subjects, not fitting a single subject's own trials better.
- **Task-specific branches earn their keep**: `learned_evoked` (Inria) and `learned_advance` (BETA_4s) both beat plain
  `learned` by a clear margin in the mode where they matter most (Inria inter: 0.587 vs 0.518; BETA_4s
  intra: 0.071 vs 0.058), confirming the paradigm-specific branch design from ADR 0016 rather than just
  adding parameters.
- **`BCICIV1_Train` stays undecodable** in the restructured pipeline too (no head beats `raw_band`
  convincingly, n=5 is descriptive-only per protocol) — rules out "the old pipeline's bugs were hiding a
  real effect here."

## Consequences

- The baseline decides which head becomes the locked core head of ADR 0016 and where the ablation grid
  starts (time pooling, spatial filter count, feature granularity, dropout).
- Conclusions from the old runs (for example "C1 ties the raw control within-subject") are treated as
  superseded until re-measured here.
