# Card: combined head (stamp_power + latent_signed)

Written 2026-09-29, before the runs. Follows the source-stamps pilot: the stamp head (stamp_power) beats the z probe
on MI few-shot and BNCI2014004 loso, the z probe wins P300 and BNCI2014001 loso.

- **Change (head only, no pretraining):** `features: [stamp_power (time_pool learned, time_rank 2), latent_signed
  (pca, time_rank 2)]`, spatial_k 8 per entry, dropout 0.3; 128 + 128 features into one BatchNorm / Dropout / Linear.
- **Backbones:** `mesae_tiny_p50_s16_s1..3`. Parents: z probe (`patch_probe`) and stamp head (`cw_stamp`) on the same
  backbones (stamp head on s3 run here).
- **Hypothesis:** one head uses power where MI carries it and signed z where P300 carries it.

## Metric and decision

Per cell, tail balanced accuracy, mean over finetune seeds 1-3 per backbone, mean +- SE over the 3 backbones;
won / lost when |diff| > 2 x sqrt(SE_a^2 + SE_b^2). Loso cells decide; few-shot reported.

- **Default head** if on every loso cell the combined head is not beaten by the better parent and it beats the worse
  parent on >= 1 loso cell.
- Otherwise: paradigm-specific heads (stamp power for MI, z probe for P300), recorded.

## Result (2026-09-29 23:38): adopted

Loso, combined vs the better parent: BNCI2014004 +1.4 vs stamp (threshold 0.5), BNCI2014001 +3.7 vs z (1.4),
BNCI2014008 +0.5 vs z (0.1) -- wins all three, beats the worse parent too. Few-shot (reported): tie on BNCI2014004,
-3.7 vs stamp on BNCI2014001, -1.9 vs z on BNCI2014008 (larger train/test gap). Small backbone, one seed: 80.2 / 53.1 /
70.0 loso. Report: docs/reports/2026-09-29-combined-head.md.
