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
- **Verdict:** adopted. On every loso cell the combined head beats the better parent (BNCI2014004 +1.4 vs stamp,
  BNCI2014001 +3.7 vs z, BNCI2014008 +0.5 vs z). It overfits more; on few-shot that costs BNCI2014001 (-3.7 vs stamp)
  and BNCI2014008 (-1.9 vs z). On the small backbone it reaches 80.2 / 53.1 / 70.0 loso.
- **Cards / ADRs:** [combined-head card](../cards/2026-09-29-combined-head.md); ADR 0016 (head modules), 0020.

## Results

Tiny, tail balanced accuracy (%), mean over 3 backbones (per backbone in brackets):

| Cell | z probe | stamp head | combined | vs z (thr) | vs stamp (thr) |
|---|---|---|---|---|---|
| BNCI2014004 loso | 72.3 (71.9 / 70.4 / 74.4) | 75.7 (76.1 / 75.5 / 75.5) | **77.1** (77.0 / 77.4 / 77.1) | +4.9 (2.4) win | +1.4 (0.5) win |
| BNCI2014001 loso | 45.9 (46.7 / 45.9 / 45.0) | 41.8 (40.7 / 42.9 / 41.8) | **49.5** (50.5 / 49.3 / 48.9) | +3.7 (1.4) win | +7.8 (1.6) win |
| BNCI2014008 loso | 68.7 (68.7 / 68.8 / 68.7) | 62.3 (69.6 / 59.3 / 58.0) | **69.3** (69.3 / 69.2 / 69.3) | +0.5 (0.1) win | +6.9 (7.3) tie |
| BNCI2014004 few-shot | 66.9 | 75.2 | 75.5 | +8.6 (2.6) win | +0.3 (1.5) tie |
| BNCI2014001 few-shot | 32.0 | 44.6 | 40.9 | +8.8 (1.5) win | -3.7 (0.8) loss |
| BNCI2014008 few-shot | 61.4 | 52.9 (s2, s3 only) | 59.5 | -1.9 (1.1) loss | -- |

Train / held-out (gap), tiny, mean over backbones:

| Cell | z probe | stamp head | combined |
|---|---|---|---|
| BNCI2014004 loso | 78.7 / 72.3 (+6.4) | 81.5 / 75.7 (+5.8) | 86.0 / 77.1 (+8.9) |
| BNCI2014001 loso | 53.9 / 45.9 (+8.0) | 56.3 / 41.8 (+14.6) | 66.7 / 49.5 (+17.1) |
| BNCI2014008 loso | 70.0 / 68.7 (+1.3) | 63.8 / 62.3 (+1.5) | 71.0 / 69.3 (+1.7) |
| BNCI2014004 few-shot | 99.8 / 66.9 (+32.9) | 97.5 / 75.2 (+22.2) | 99.9 / 75.5 (+24.4) |
| BNCI2014001 few-shot | 96.0 / 32.0 (+63.9) | 94.7 / 44.6 (+50.1) | 99.4 / 40.9 (+58.5) |
| BNCI2014008 few-shot | 82.4 / 61.4 (+21.0) | -- | 93.1 / 59.5 (+33.6) |

Small backbone, one seed (parents from 2026-09-28, same head configs, older code and caches -- not rerun):

| Cell | z probe | stamp head | combined |
|---|---|---|---|
| BNCI2014004 loso | 74.9 | 75.5 | **80.2** |
| BNCI2014001 loso | 49.5 | 43.0 | **53.1** |
| BNCI2014008 loso | 69.1 | **70.4** | 70.0 |
| BNCI2014004 few-shot | 68.7 | 76.4 | **78.1** |
| BNCI2014001 few-shot | 35.4 | **47.7** | 43.7 |
| BNCI2014008 few-shot | **60.7** | 60.5 | 58.8 |

Against EEG-FM-Compass Table V (loso): small combined 80.2 / 53.1 / 70.0 on BNCI2014004 / 001 / 008 vs best FM linear
probe 75.57 / 48.24 / 67.11, best FM full fine-tune 77.70 / 53.03 / 69.91, best specialist 76.38 / 46.80 / 72.29.

## Notes

- The combined head has the largest train/test gap in every cell but BNCI2014008 loso. On loso, where many training
  subjects constrain it, the extra capacity still transfers; on few-shot (~20 calibration trials per class) it
  memorises (99% train) and loses to the best parent on BNCI2014001 and BNCI2014008.
- The stamp head alone is unstable on P300 loso (69.6 / 59.3 / 58.0 across backbones); the combined head is not
  (69.2-69.3): the z half carries the phase-locked response.
- Tiny stamp-head seed 1 has no BNCI2014008 few-shot runs (not part of that earlier head set).
- The small parents were not rerun (user decision): the small comparison mixes code versions; its combined numbers
  are this code.
- Compass reports the last epoch; ours is the mean of the last 10.
- Follow-up for few-shot: stronger regularisation of the combined head (dropout, smaller spatial_k on one entry).

## Files

`2026-09-29-combined-head/`: `verdict_tables.md` (generated) and `scripts/` (`run.sh`, `small.sh`, `ft_plan.py`,
`verdict.py`, `check.py`, `gap.py`, `p300_gate.py`, `run.log`). Raw output: `output/<backbone>/finetune/combined/`.
