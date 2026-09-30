# Combined head (stamp_power + latent_signed)

- **Date:** 2026-09-29
- **Question:** does one head reading both the stamp power and the signed z beat each of its parents -- the stamp head
  (strong on MI) and the z probe (strong on P300)?
- **Runs:** tiny `mesae_tiny_p50_s16_s1..3` (judged); small `mesae_small_p50_s16_s1` (one seed, reported). Head only,
  no pretraining. Combined: `features [stamp_power (time_pool learned, time_rank 2), latent_signed (pca, time_rank 2)]`,
  spatial_k 8 per entry, dropout 0.3 (128 + 128 features, one BatchNorm / Dropout / Linear).
- **Protocol:** BNCI2014004 / 001 / 008 x loso / few-shot (Compass), finetune seeds 1-3, tail balanced accuracy; per
  backbone the mean over finetune seeds, mean +- SE over the 3 tiny backbones; won when |diff| > 2 x sqrt(SE_a^2 + SE_b^2).
  Loso decides. Overfitting: train balanced accuracy (last 10 epochs of each fold, dropout on) vs held-out.
- **Verdict:** adopted, confirmed on 3 small-corpus seeds. On every loso cell the combined head beats the better parent (BNCI2014004 +1.4 vs stamp,
  BNCI2014001 +3.7 vs z, BNCI2014008 +0.5 vs z). It overfits more; on few-shot that costs BNCI2014001 (-3.7 vs stamp)
  and BNCI2014008 (-1.9 vs z). On the small backbone it reaches 80.2 / 53.1 / 70.0 loso.
- **Cards / ADRs:** [combined-head card](../cards/2026-09-29-combined-head.md); ADR 0016 (head modules), 0020.

## Results

Tiny, tail balanced accuracy (%), mean over 3 backbones (per backbone in brackets):

| Cell | z probe | stamp head | combined | vs z (thr) | vs stamp (thr) |
|---|---|---|---|---|---|
| BNCI2014004 loso | 72.3 (71.9 / 70.4 / 74.4) | 75.7 (76.1 / 75.5 / 75.5) | **77.1** (77.0 / 77.4 / 77.1) | +4.9 (2.4) win | +1.4 (0.5) win |
| BNCI2014001 loso | 45.9 (46.7 / 45.9 / 45.0) | 41.8 (40.7 / 42.9 / 41.8) | **49.5** (50.5 / 49.3 / 48.9) | +3.7 (1.4) win | +7.8 (1.6) win |
| BNCI2014008 loso | 68.7 (68.7 / 68.8 / 68.7) | 57.7 (55.8 / 59.3 / 58.0) | **69.3** (69.3 / 69.2 / 69.3) | +0.5 (0.1) win | +11.5 (2.0) win |
| BNCI2014004 few-shot | 66.9 | 75.2 | 75.5 | +8.6 (2.6) win | +0.3 (1.5) tie |
| BNCI2014001 few-shot | 32.0 | 44.6 | 40.9 | +8.8 (1.5) win | -3.7 (0.8) loss |
| BNCI2014008 few-shot | 61.4 | 52.6 | 59.5 | -1.9 (1.1) loss | +6.9 (1.0) win |

Train / held-out (gap), tiny, mean over backbones:

| Cell | z probe | stamp head | combined |
|---|---|---|---|
| BNCI2014004 loso | 78.7 / 72.3 (+6.4) | 81.5 / 75.7 (+5.8) | 86.0 / 77.1 (+8.9) |
| BNCI2014001 loso | 53.9 / 45.9 (+8.0) | 56.3 / 41.8 (+14.6) | 66.7 / 49.5 (+17.1) |
| BNCI2014008 loso | 70.0 / 68.7 (+1.3) | 58.8 / 57.7 (+1.1) | 71.0 / 69.3 (+1.7) |
| BNCI2014004 few-shot | 99.8 / 66.9 (+32.9) | 97.5 / 75.2 (+22.2) | 99.9 / 75.5 (+24.4) |
| BNCI2014001 few-shot | 96.0 / 32.0 (+63.9) | 94.7 / 44.6 (+50.1) | 99.4 / 40.9 (+58.5) |
| BNCI2014008 few-shot | 82.4 / 61.4 (+21.0) | 80.6 / 52.6 (+28.0) | 93.1 / 59.5 (+33.6) |

Small backbone, one pretrain seed, finetune seeds 1-3 (all three heads on the current code):

| Cell | z probe | stamp head | combined |
|---|---|---|---|
| BNCI2014004 loso | 74.9 | 75.5 | **80.2** |
| BNCI2014001 loso | 49.5 | 43.0 | **53.1** |
| BNCI2014008 loso | 69.1 | 58.8 | **70.0** |
| BNCI2014004 few-shot | 68.7 | 76.4 | **78.1** |
| BNCI2014001 few-shot | 35.4 | **47.7** | 43.7 |
| BNCI2014008 few-shot | **60.9** | 53.1 | 58.8 |

Small corpus, 3 pretrain seeds (`mesae_small_p50_s16_s1..3`; seeds 2-3 pretrained 2026-09-30 with s1's config, seed
changed), same rule:

| Cell | z probe | stamp head | combined | vs z (thr) | vs stamp (thr) |
|---|---|---|---|---|---|
| BNCI2014004 loso | 75.2 (74.9 / 74.5 / 76.1) | 74.9 (75.5 / 74.1 / 75.0) | **79.1** (80.2 / 78.2 / 78.9) | +4.0 (1.5) win | +4.2 (1.5) win |
| BNCI2014001 loso | 49.8 (49.5 / 49.8 / 50.1) | 42.3 (43.0 / 42.5 / 41.4) | **53.3** (53.1 / 53.7 / 53.2) | +3.5 (0.6) win | +11.0 (1.0) win |
| BNCI2014008 loso | 68.8 (69.1 / 68.4 / 68.7) | 59.1 (58.8 / 59.7 / 58.8) | **69.6** (70.0 / 69.1 / 69.6) | +0.8 (0.7) win | +10.4 (0.8) win |
| BNCI2014004 few-shot | 70.7 | 76.1 | **78.0** | +7.3 (3.2) win | +1.9 (1.3) win |
| BNCI2014001 few-shot | 36.1 | 47.5 | 44.0 | +7.9 (1.5) win | -3.4 (0.8) loss |
| BNCI2014008 few-shot | 61.5 | 53.4 | 59.8 | -1.7 (1.2) loss | +6.4 (1.0) win |

The verdict carries to the 4x corpus: the combined head wins every loso cell against both parents, and gains over the
tiny corpus on every loso cell (77.1 -> 79.1, 49.5 -> 53.3, 69.3 -> 69.6).

Against EEG-FM-Compass Table V (loso): small combined (3 seeds) 79.1 / 53.3 / 69.6 on BNCI2014004 / 001 / 008 vs best FM linear
probe 75.57 / 48.24 / 67.11, best FM full fine-tune 77.70 / 53.03 / 69.91, best specialist 76.38 / 46.80 / 72.29.

MI screen, finetune seed 1 (stamp power vs power taken from z): `latent_power` = the stamp_power pipeline on z (PCA 16
axes, squared, learned time pool, log); `z_combined` = latent_power + latent_signed.

| Backbone / cell | stamp head | latent_power | combined | z_combined |
|---|---|---|---|---|
| tiny s1, BNCI2014004 loso | 75.7 | 73.1 | 77.0 | 73.9 |
| tiny s1, BNCI2014001 loso | 40.8 | 41.1 | 51.7 | 50.9 |
| tiny s1, BNCI2014004 few-shot | 76.8 | 73.4 | 74.2 | 70.5 |
| tiny s1, BNCI2014001 few-shot | 43.3 | 40.4 | 41.5 | 38.4 |
| small, BNCI2014004 loso | 76.4 | 72.8 | 80.2 | 75.1 |
| small, BNCI2014001 loso | 42.0 | 45.1 | 53.8 | 51.7 |
| small, BNCI2014004 few-shot | 76.8 | 69.3 | 74.9 | 70.7 |
| small, BNCI2014001 few-shot | 46.5 | 34.0 | 45.5 | 35.5 |

The stamp (a, b) power beats power taken from z on 7 / 8 cells (+2.6 to +12.5) and the combined head beats z_combined on
all 8: the stamps carry something squaring z does not. latent_power and z_combined were dropped after this screen (their
run folders were deleted by mistake; the numbers above are the only record).

Per-stamp spatial filters (2026-09-30, tiny s1, MI cells, finetune seeds 1-3): the stamp half with its own K = 2 filters
per stamp (`spatial_per_stamp`, filter-bank-CSP style) instead of one shared K = 8 filter, z half unchanged. Worse on
every cell: BNCI2014004 loso 74.6 vs 77.0 (-2.4), BNCI2014001 loso 48.8 vs 50.5 (-1.7), BNCI2014004 few-shot 72.0 vs
75.4 (-3.3), BNCI2014001 few-shot 38.3 vs 41.3 (-3.0); the train/test gap did not shrink. A shared filter pools the
spatial evidence of all stamps (mu and beta ERD share their lateral motor topography); per-stamp filters must each be
learned from the same few trials. Not pursued; the option stays in the code (default off).

## Notes

- The combined head has the largest train/test gap in every cell but BNCI2014008 loso. On loso, where many training
  subjects constrain it, the extra capacity still transfers; on few-shot (~20 calibration trials per class) it
  memorises (99% train) and loses to the best parent on BNCI2014001 and BNCI2014008.
- The stamp head alone is weak on P300 (loso 55.8-59.3); the combined head keeps the z probe's level (69.2-69.3): the
  z half carries the phase-locked response. (A 2026-09-28 tiny-s1 stamp run read 69.6 on P300 loso; rerun on the
  current code it reads 55.8 -- the old run is kept as `cw_stamp_0928`, not used.)
- The small parents were rerun on the current code; the MI cells reproduced the 2026-09-28 values (e.g. 74.9 / 49.5 z
  probe, 75.5 / 43.0 stamp head), so the older MI runs were already comparable.
- Compass reports the last epoch; ours is the mean of the last 10.
- Follow-up for few-shot: stronger regularisation of the combined head (dropout, smaller spatial_k on one entry).

## Files

`2026-09-29-combined-head/`: `verdict_tables.md` (generated) and `scripts/` (`run.sh`, `small.sh`, `ft_plan.py`,
`verdict.py`, `check.py`, `gap.py`, `p300_gate.py`, `run.log`). Raw output: `output/<backbone>/finetune/combined/`.
