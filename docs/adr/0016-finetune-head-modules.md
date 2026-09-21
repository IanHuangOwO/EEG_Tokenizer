# 0016 — Finetune head: modules to keep, the locked core head, and the ablation grid

Status: Proposed. Becomes Accepted when the step-0 rerun (below) and the Phase 2 results of
ADR 0014 are in.
Date: 2026-09-21

## Context

ADR 0014 ran the finetune head as one class (`MeSAEFeatureHead`) with flags, changing one
factor at a time on a frozen `mesae_v10_small_uw01` backbone. The evidence so far, all
balanced accuracy on BCICIV2a unless noted:

- **Spatial mix.** A signed `spatial:K` filter beat channel concat by about +0.05 on every
  feature (Experiment B1); a convex (softmax) channel pool cannot represent a contrast (ADR
  0012). The learned filters are not stable across folds (mean matched |cosine| 0.55
  within-subject, 0.65 in LOSO).
- **Feature source.** Within-subject, raw band power (0.530) and C1 (0.536) tie. Cross-subject
  (LOSO, unseen subjects), C1 beats raw: BCICIV2a at 100 epochs 0.396 vs 0.342 (p = 0.007,
  8/9), BCICIV2b at 30 epochs 0.684 vs 0.594 (p = 0.001, 9/9). `z_chan`, `chan_mag`,
  `pool_mag` and `head_z` were poor or overfit.
- **Time pooling.** Learned `learned:2` weights 0.536 vs flat 0.495 within-subject (not
  significant). The learned LOSO weights share one shape (peak near 1.4 s after the event),
  so they carry a stable signal. Flat vs learned has not been compared cross-subject.
- **Phase advance (C3)** and **evoked branch (C4)** each lost to C1 on motor imagery
  within-subject (0.489, p = 0.010; 0.466, p = 0.007). They were built for SSVEP and ERP and
  are being tested there in Phase 2.
- **Dropout.** Dropout 0.5 matters for the wide heads (C0 0.456 without, 0.495 with). The
  Phase 1 raw control used dropout 0 while C1 used 0.5, so those rows mix input and dropout.

The numbers in ADR 0014 before Phase 2 came from the `base` conda env without `mne`
(fallback coordinates); Phase 2 runs in `eeg_fm`.

## Decision

### Modules (swappable pieces of a head)

1. **Feature source:** `stamp_induced` (per-stamp log power) is the main option; raw band
   power is the control.
2. **Spatial mix:** signed `spatial:K` on the code amplitudes (a, b) before the power.
3. **Time pooling:** `learned:R` (default), with flat (`trial`) and `window:lo-hi` as ablation
   options.
4. **Optional branches:** phase advance (SSVEP only) and the evoked branch (ERP only). Each
   is kept only if Phase 2 supports it on its own dataset.

### Parameters, not modules

- **Dropout** is a head argument (default 0.5), ablated at 0.3, 0.5, 0.7. It is not a
  separate module.
- **K** (spatial filters) and **R** (time-weight rank) are head arguments.
- **Readout** is fixed as BatchNorm1d, then dropout, then linear. It is not a module and is
  not ablated.

### Locked core head, H\*

`stamp_induced` → `spatial:8` → `learned:2` → readout with dropout 0.5. This is C1 (1,844
parameters on 4 classes).

### Controls (kept for comparison, not candidates)

The raw band-power head (dropout matched), the PSDA power-spectral reference for SSVEP, and
the raw ERP head for ERP.

### Dropped

Cross-stamp coupling (C5), `z_chan`, `chan_mag`, `pool_mag`, `head_z`, channel-concat
pooling, softmax channel pooling, and the recon-input head (redundant with raw). The
ADR 0014 "group penalty" for width control was never implemented and stays out.

### Open item: a stronger raw control

The raw band-power control (two coarse bands, 16 features) is weak. On EEGMMIdb (Phase 2, 3 classes,
balanced accuracy) it sits at chance (0.366 unseen, 0.340 seen) with train accuracy 0.387, i.e. it
underfits, while C1 reaches 0.539 unseen (+0.174, 10/10 subjects). That gap shows the per-stamp features
carry far more usable signal than two band powers; it does not show the tokenizer beats a well-designed
raw pipeline. Before the head is locked, add a stronger raw control (for example more bands, or a small
learned filter bank on the raw signal) at a comparable feature count and dropout.

## Ablation grid

Leave one out from H\*, on LOSO of BCICIV2a and BCICIV2b at 100 epochs, dropout matched, in
`eeg_fm`, subject as the unit of analysis (ADR 0014 Protocol; point difference is the bar, p
reported not gated):

1. Time pooling: flat, `window:lo-hi`, `learned:R` for R = 1, 2, 4.
2. Spatial K: 4, 8, 16.
3. Feature granularity: per-stamp (`stamp_induced`) against band-summed (`stamp_bandpow`).
4. Dropout: 0.3, 0.5, 0.7, with the same values on the raw control.
5. Optional branches, each on its own dataset only.

## Step 0 before this ADR is Accepted

Rerun BCICIV2a LOSO C1 vs the raw control at 100 epochs in `eeg_fm` with matched dropout
(0.5 on both). This removes the environment and dropout confounds from the headline
comparison the design rests on.

## Consequences

- The head becomes a composition of the modules above, implemented behind the existing
  `MeSAEFeatureHead` constructor arguments so configs and the factory keep working.
- **Checkpoint compatibility is a hard constraint:** parameter names stay
  `head.spatial.weight`, `head.time.p/q`, `head.evoked.p/q`, `head.cls.*`, plus the `keep`,
  `E_D`, `E_H` buffers, because `viz.load_model` and every finished run load them.
- The refactor is verified by numerical equivalence with the current class (same state dict,
  same inputs, same logits), including one real saved checkpoint.
- Rows in ADR 0014 stay as measured; this ADR only decides what is kept and how the head is
  organised.
