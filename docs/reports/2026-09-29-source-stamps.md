# Source-factorized stamps (spatial_rank 4), one-seed pilot

- **Date:** 2026-09-29
- **Question:** does forcing each stamp's per-channel gain column to rank K = 4 (fixed scalp topographies from
  electrode positions, `src` activations) make the stamp code a source-level, more usable representation?
- **Runs:** `mesae_tiny_p50_s16_k4_s2` (`stamp_bank.spatial_rank` 4) vs `mesae_tiny_p50_s16_s2` (same config, data and
  seed; only the factorization differs).
- **Protocol:** stamp head (`cw_stamp`: stamp_power, learned time pool, spatial_k 8) and z probe (`patch_probe`), 6 cells,
  finetune seeds 1-3, tail balanced accuracy; loso primary. backbone_eval masked MSE against the three patch-50 seeds.
- **Verdict:** not promising, shelved. The stamp head loses BNCI2014001 loso by 9.0 and wins nothing; the learned
  topographies are montage-wide rim-heavy gradients, not focal sources.
- **Cards / ADRs:** [source-stamps card](../cards/2026-09-29-source-stamps.md), [stamp-hidden report](2026-09-29-stamp-hidden.md).

## Results

Tail balanced accuracy (%), mean over finetune seeds 1-3 (stamp head: sd over seeds in brackets):

| Cell | z probe p50_s2 | z probe K4 | stamp head p50_s2 | stamp head K4 | stamp K4 - p50 |
|---|---|---|---|---|---|
| BNCI2014004 loso | 70.4 | 72.1 | 75.5 (0.3) | 74.0 (0.5) | -1.6 |
| BNCI2014001 loso | 45.9 | 43.4 | 42.9 (0.8) | 33.9 (0.7) | **-9.0** |
| BNCI2014008 loso | 68.8 | 68.5 | 59.3 (0.2) | 59.0 (0.3) | -0.2 |
| BNCI2014004 few-shot | 65.3 | 66.3 | 74.3 (0.6) | 72.6 (0.9) | -1.8 |
| BNCI2014001 few-shot | 31.4 | 31.1 | 45.2 (0.3) | 36.4 (0.3) | -8.8 |
| BNCI2014008 few-shot | 62.2 | 62.4 | 53.1 (0.2) | 53.5 (0.1) | +0.4 |

Masked MSE (test masks): token runs 0.369 (patch-50 seeds 0.397 / 0.407 / 0.405), random channel 0.338 (0.318-0.327),
channel cluster 0.355 (0.339-0.346), time block 0.794 (0.778-0.780).

## Notes

- Topographies (`topographies.png`, 16 stamps x 4, 10-10 montage): large-scale gradients weighted on the rim
  (frontal, temporal, occipital edges), several near-global maps; no focal central (C3 / C4) mu patterns. The
  reconstruction spends the limited rank on high-variance, scalp-wide content (likely artefact and global fields).
- The loss sits on the 22-channel set: BNCI2014004 has 3 channels, fewer than K, so its code loses nothing to the
  rank limit; BNCI2014001's lateralised motor detail does not fit 4 fixed maps per stamp.
- Unexpected side result: token-run masked MSE improved 9% (cross-channel pooling helps fill masked patches) while
  channel masks got worse.
- The stamp head still beats the z probe on MI few-shot on both backbones (004 74.3 vs 65.3, 001 45.2 vs 31.4): the
  spectral split of the stamps is useful there, the spatial factorization is not.
- Untried variants if revisited: larger K, topographies conditioned on z (per recording), a per-channel residual path,
  a sparsity prior on src.

## Files

`2026-09-29-source-stamps/`: `topographies.png` and `scripts/` (`run.sh` driver, `ft_plan.py`, `topo.py`, `run.log`).
Code: `stamp_bank.spatial_rank` in `model/MeSAE/MeSAE_modules.py` (StampBank._factorize), default 0.
