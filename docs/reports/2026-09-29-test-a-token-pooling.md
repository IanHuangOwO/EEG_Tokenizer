# Test A: is patch 100's MI gain from fewer tokens? (latent token pooling)

- **Date:** 2026-09-29
- **Question:** does averaging adjacent patch-50 z tokens (39 -> 19 per window, patch 100's count and stride) in the
  head reproduce patch 100's MI gain, and does it avoid patch 100's P300 loss?
- **Runs:** the three patch-50 backbones of the patch-length report (`mesae_tiny_notrial_s1`, `mesae_tiny_p50_s16_s2`,
  `_s3`), no retraining; head label `patch_probe_pool2` (`training_params.finetune.latent_pool` 2) against their
  `patch_probe`, and against the patch-100 backbones' `patch_probe`.
- **Protocol:** z probe (`latent_signed`, pca, spatial_k 8), 6 cells, finetune seeds 1-3, tail balanced accuracy; per
  condition mean +- SE over the 3 pretrain seeds; won when |diff| > 2 x sqrt(SE_a^2 + SE_b^2). Loso cells primary
  (loso-first rule for tiny corpora). Secondary: closed-form ridge probe on z with `pool=2`.
- **Verdict:** pooling does not help loso and costs P300 loso (-1.6, like patch 100's -1.9), so head-level pooling is
  no free substitute for longer patches. It recovers about 60% / 40% of patch 100's MI few-shot gain (004 / 001):
  that gain is roughly half statistical (fewer, smoother probe inputs, less overfitting), half the longer patch itself.
  Patch 50 stays the default, now with the P300 cost shown to come from time resolution.
- **Cards / ADRs:** [patch-length report](2026-09-29-patch-length.md), [patch-75 report](2026-09-29-patch-75.md);
  docs/finetune-caveats.md items 7-9; ADR 0020.

## Results

z probe, tail balanced accuracy (%), mean over 3 pretrain seeds (per seed in brackets):

| Cell | patch 50 | patch 50 pool2 | patch 100 | pool2 - p50 | threshold | pool2 vs p50 | p100 - p50 |
|---|---|---|---|---|---|---|---|
| BNCI2014004 loso | 72.3 | 72.9 (72.6 / 71.1 / 75.0) | 73.3 | +0.6 | 3.3 | tie | +1.1 |
| BNCI2014001 loso | 45.9 | 44.7 (44.3 / 44.7 / 45.0) | 46.8 | -1.2 | 1.0 | loses | +0.9 |
| BNCI2014008 loso | 68.7 | 67.1 (66.9 / 67.3 / 67.1) | 66.9 | -1.6 | 0.3 | loses | -1.9 |
| BNCI2014004 few-shot | 66.9 | 70.3 (69.9 / 70.0 / 71.1) | 72.5 | +3.4 | 2.7 | wins | +5.7 |
| BNCI2014001 few-shot | 32.0 | 33.6 (34.1 / 32.5 / 34.3) | 35.8 | +1.6 | 1.9 | tie | +3.7 |
| BNCI2014008 few-shot | 61.4 | 61.3 (60.4 / 62.0 / 61.4) | 60.7 | -0.1 | 1.2 | tie | -0.7 |

Closed-form ridge probe on z, mean over 3 pretrain seeds:

| Dataset | patch 50 | patch 50 pool2 | patch 100 |
|---|---|---|---|
| BNCI2014004 | 69.4 | 70.4 | 72.5 |
| BNCI2014001 | 38.4 | 40.3 | 40.5 |
| BNCI2014008 | 69.4 | 67.9 | 67.3 |

## Notes

- Share of patch 100's few-shot gain recovered by pooling: 004 3.4 / 5.7 = 60%, 001 1.6 / 3.7 = 43%; the ridge probe
  recovers all of 001's gain and a third of 004's.
- The 001 loso loss (-1.2, threshold 1.0) is narrow and patch 100 does not show it (+0.9); read it as "no loso gain"
  rather than a firm loss. The P300 loss is firm and matches patch 100.
- For comparison, the tiny -> small corpus step (one pretrain seed, 2026-09-28-overnight) gave +3.3 / +4.5 on the
  same two MI few-shot cells and also lifted MI loso; patch length is worth re-testing at medium scale.
- Patch 75's BNCI2014004 loso win (+3.0) is not explained here: pooling does not reproduce it.

## Files

`2026-09-29-test-a-token-pooling/`: `verdict_tables.md` (generated) and `scripts/` -- `run.sh` (driver: ridge probe
then finetunes per backbone), `ft_plan.py` (finetune plan, `latent_pool=2`), `verdict.py`, `run.log`, `ridge.log`.
Raw output: `output/<backbone>/finetune/patch_probe_pool2/` and `output/<backbone>/pretrain/analysis/ridge_probe_pool2.json`.
