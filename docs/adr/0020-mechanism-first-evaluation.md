# 0020 — Judge each change by its own mechanism, not by finetuning

Status: Accepted (2026-09-27). Standing protocol for every backbone experiment from now on.
Date: 2026-09-27

## Context

On the tiny corpus with one seed, the finetune numbers are too noisy to judge single changes:

- Two runs that differ only in the mp ranking (`mesae_tiny_skipdrop_graded_s1` vs `mesae_tiny_rankfix_s1`)
  differ by up to about 2 balanced-accuracy points per cell (loso, BNCI2014001/004/008).
- Each cell has 8-9 test subjects with large subject-to-subject spread, plus one head-training run.
- A loss change reaches the tasks only indirectly.

Backbone-level measures are much steadier: paired (same held-out windows, same fixed masks),
thousands of patches, deterministic. The spatial-embedding ablation is the model: shuffling the
coordinates raises masked MSE by 170-190%, which no seed can hide.

## Decision

Every backbone change gets a **mechanism card**, written before the run:

1. **Hypothesis:** what the change should do.
2. **Metric:** the backbone-level measure of exactly that (table below).
3. **Pass threshold:** against the reference run, measured on the same windows and masks, and
   clearly beyond the metric's own run-to-run spread (graded vs rankfix gives the first estimate).

The change is kept or dropped on its card. Finetuning is **not** run per change. It runs when a
recipe (several accepted changes) is frozen: 2 seeds, loso and few-shot, the full Compass
comparison. The closed-form ridge probe runs on every backbone as a cheap, deterministic
representation check, reported next to the card but not a pass criterion unless the card says so.

| Change type | Metric | Where |
|---|---|---|
| Spatial / coordinate embedding | masked MSE change with coords shuffled / at the mean position | `backbone_eval` |
| Time embedding, temporal bias | masked MSE change with time shuffled / constant; attention range and entropy | `backbone_eval`, `attention_range` |
| Skips, skip drop-path, deep-path use | masked MSE change with skips removed at eval (`skips_off`); attention LayerScale | `backbone_eval`, `attention_range` |
| Stamp dictionary, dedup (mp_loss) | per-stamp remove-one cost (a redundant stamp costs ~0), rank spread, energy share | `stamp_usage` |
| Trial term, seams | seam disagreement between overlapping patches | `backbone_eval` |
| Masked band power (STFT-type targets) | masked predicted/true power per band, log-spectral distance | `backbone_eval` |
| Channel imputation | random-channel / `motor3_to_bci22` MSE vs inverse-distance interpolation | `backbone_eval` |
| Representation linearity | closed-form ridge probe on pre-stamp z, Compass loso | `ridge_probe` |
| General reconstruction | masked / unmasked val MSE (ep 41-50) | training log |

All of these are in `analysis_pretrain --preset quick`.

## Consequences

- A change that passes its card but moves downstream the wrong way at the recipe freeze is
  revisited then, with 2 seeds, not dropped on one noisy cell.
- Earlier verdicts read under this rule:
  - Nested loss (ADR 0018): its dedup purpose is not met. The duplicate count it was judged on
    turned out to be the wrong metric (`stamp_usage`: every stamp, including 0.99-similar pairs,
    costs >= 7.7% MSE to remove), so the question is moot: there were no redundant stamps.
  - STFT loss (ADR 0019): passes its mechanism (masked band power up, distance down), but that
    did not reach the tasks; stays off.
  - Trial term: re-judge on seam disagreement (`backbone_eval`) against the reference.
- Stamp near-duplicate counts (`stamp_duplicates`) are descriptive only, not a dedup criterion.
