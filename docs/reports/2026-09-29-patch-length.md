# Patch length 50 vs 100

- **Date:** 2026-09-29
- **Question:** does a 0.5 s patch (patch_len 100) beat the 0.25 s patch (50) downstream, and where?
- **Runs:** static 16-stamp tiny-corpus backbones, three pretrain seeds per patch length:
  patch 50 `mesae_tiny_notrial_s1`, `mesae_tiny_p50_s16_s2`, `_s3`; patch 100 `mesae_tiny_p100_s16_s1`, `_s2`, `_s3`.
  Only `patch_len` / `patch_stride` (100 / 50 vs 50 / 25) and the training seed differ.
- **Protocol:** z probe (`latent_signed`, pca, spatial_k 8), BNCI2014004 / 001 / 008 x loso / few-shot (Compass
  protocols), finetune seeds 1-3, tail balanced accuracy. Per backbone the mean over finetune seeds; per patch
  length the mean +- SE over the 3 pretrain seeds; a cell is won when |diff| > 2 x sqrt(SE_50^2 + SE_100^2).
- **Verdict:** a trade-off. Patch 100 wins both MI few-shot cells (004 +5.7, 001 +3.7) and loses P300 loso
  (-1.9); the other three cells tie. Per the card, patch 50 stays the default and patch 75 is the next card.
- **Cards / ADRs:** [patch-length card](../cards/2026-09-29-patch-length.md); ADR 0020.

## Results

Downstream z probe, tail balanced accuracy (%), mean over 3 pretrain seeds (per-seed values in brackets):

| Cell | patch 50 | patch 100 | diff | threshold | result |
|---|---|---|---|---|---|
| BNCI2014004 loso | 72.3 (71.9 / 70.4 / 74.4) | 73.3 (72.0 / 73.9 / 74.1) | +1.1 | 2.7 | tie |
| BNCI2014004 few-shot | 66.9 (65.9 / 65.3 / 69.4) | 72.5 (70.8 / 73.6 / 73.2) | **+5.7** | 3.1 | p100 wins |
| BNCI2014001 loso | 45.9 (46.7 / 45.9 / 45.0) | 46.8 (48.4 / 46.6 / 45.3) | +0.9 | 2.1 | tie |
| BNCI2014001 few-shot | 32.0 (31.3 / 31.4 / 33.5) | 35.8 (34.8 / 35.7 / 36.8) | **+3.7** | 1.8 | p100 wins |
| BNCI2014008 loso | 68.7 (68.7 / 68.8 / 68.7) | 66.9 (66.9 / 67.9 / 65.8) | **-1.9** | 1.2 | p100 loses |
| BNCI2014008 few-shot | 61.4 (61.0 / 62.2 / 61.0) | 60.7 (60.6 / 61.1 / 60.4) | -0.7 | 0.9 | tie |

Closed-form ridge probe on z (secondary), mean +- SE over seeds: same direction.

| Dataset | patch 50 | patch 100 | diff |
|---|---|---|---|
| BNCI2014004 | 69.4 +- 1.9 | 72.5 +- 0.8 | +3.1 |
| BNCI2014001 | 38.4 +- 0.9 | 40.5 +- 0.1 | +2.1 |
| BNCI2014008 | 69.4 +- 0.1 | 67.3 +- 0.7 | -2.1 |

Masked reconstruction MSE (reported, not judged: test masks are defined in patches, so patch 100 hides
twice the seconds): patch 100 is higher on every mask (token_runs 0.464 vs 0.403, time_block 0.838 vs 0.779).

## Notes

- The MI gain is in the few-shot cells, where a subject's own calibration trials train the probe; loso
  gains are within noise. Longer patches give 2 Hz spectral resolution and a whole mu/beta cycle per token.
- P300 loses where timing matters: BNCI2014008's 1 s trials are 3 patches at patch 100, so the peak shares a
  token with baseline. The loss is small (-1.9) but beyond the seed spread, which is tiny for P300 (SE ~0.3).
- Deviation: four small pretrain datasets were recompiled mid-run (see the card); patch-50 seeds 2-3 used the
  new caches. No systematic shift: their cells straddle seed 1 (e.g. 004 loso 70.4 / 74.4 vs 71.9).
- Patch 100 pretrains in about half the time (40-55 vs 75-80 min on the tiny corpus).
- Probe maps (seed 1): both patch lengths weight the first 0-1 s after the MI cue most, patch 100 more
  smoothly; P300 at patch 50 reads a sharp 0.19-0.31 s peak, at patch 100 a 0.375-0.625 s patch. The MI gain
  may partly be a smoother read of the cue-evoked response rather than better rhythm tokens (not tested;
  docs/finetune-caveats.md item 7).

## Files

`2026-09-29-patch-length/`: `verdict_tables.md` (the tables above, generated), seed-1 probe maps
`probe_maps_<cell>_seed1_p50_vs_p100.png`, and `scripts/` -- `pretrain.plan`, `ft_plan.py`, `finetune.sh`
(the finetune driver), `p100_s1_analysis.sh`, `verdict.py` (computes the verdict from each run's
`artifacts/group_eval.json`) and the queue logs. Per-backbone finetune summaries for seed 1 are in
`output/<backbone>/finetune/analysis/`.
