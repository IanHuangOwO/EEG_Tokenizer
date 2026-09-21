# 0017 — Restart baseline on the restructured finetune pipeline

Status: In progress. The baseline runs were queued on 2026-09-22; the Results section is filled in as they
finish. Supersedes the *results* of ADR 0014 (its Protocol still applies) and builds on ADR 0016.
Date: 2026-09-22

## Context

ADR 0014 records Experiments A, B and C, the Phase 1 LOSO runs and the Phase 2 runs, all measured on
the old finetune pipeline. Three problems came to light while restructuring it:

1. **Wrong electrode coordinates for the backbone.** Those runs executed in the `base` conda env, which has no
   `mne`, so `IO/loader.py`'s `get_standard_coords` silently returned nothing and the backbone read flat polar
   fallback coordinates instead of MNE 3-D positions. Measured on BCICIV2a subject 8 (stamp head, 5-fold, 100
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
| `c0_flat` | stamp code, per-stamp log power, spatial filter, flat time pooling |
| `c1_learned` | stamp code, per-stamp log power, spatial filter, learned low-rank time weights (rank 2) |
| `c1_advance` (BETA_4s only) | `c1_learned` plus the phase-advance branch |
| `c1_evoked` (Inria only) | `c1_learned` plus the evoked branch (rank 2) |

Datasets and splits:

| Dataset | `intra_subject` (5-fold per subject) | `inter_subject` |
|---|---|---|
| BCICIV2a (9 subjects, 4 classes) | all 9 | LOSO (9 folds) |
| BCICIV2b (9, 2 classes) | all 9 | LOSO (9 folds) |
| BCICIV1_Train (subjects 2, 3, 4, 5, 7; left/right) | all 5 | LOSO (5 folds) |
| Inria_Train (16, error-related potential) | all 16 | LOSO (16 folds) |
| BETA_4s (55, 40-class SSVEP) | all 55 | grouped 5-fold over subjects |
| EEGMMIdb (109, 3 classes) | 30 random subjects (seed 42) | grouped 5-fold over subjects |

`raw_signal` is run on every dataset; the two task-specific variants only on their own paradigm. Leave-one-out
over 55 and 109 subjects would cost about 2 and 20 GPU-hours per head at the current speed, so the two large
datasets use grouped 5-fold, which answers the same "does it transfer to unseen subjects" question.

Runs are written to `output/baseline/<dataset>_<mode>_<head>/`.

### Known limits

- The small backbone saw subjects 1 to 3 of each dataset in pretraining (BCICIV1_Train subjects 2 and 3 are in
  the evaluated subset), so "unseen" is only fully clean for the others.
- The seen/unseen group design, the PSDA reference for SSVEP and the full-backbone confirmation are left to the
  experiment runner work (sub-project D of the restructure).
- `raw_band` now uses a per-patch estimator (4 Hz resolution at 50-sample patches), so it is not the same
  statistic as the earlier whole-trial raw control.

## Results

Pending. Tables (per dataset and mode: mean balanced accuracy per head across subjects, paired difference of
each head against `raw_band` and of `c1_learned` against `c0_flat`, wins out of n, and the chance level) are added
here when the runs finish.

## Consequences

- The baseline decides which head becomes the locked core head of ADR 0016 and where the ablation grid
  starts (time pooling, spatial filter count, feature granularity, dropout).
- Conclusions from the old runs (for example "C1 ties the raw control within-subject") are treated as
  superseded until re-measured here.
