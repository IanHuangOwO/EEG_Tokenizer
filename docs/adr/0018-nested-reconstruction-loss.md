# 0018 — Nested reconstruction loss

Status: Rejected (2026-09-27, run `mesae_tiny_nested_s1`): 1 of 3 criteria passed. Kept: patch MSE +
per-stamp mp_loss, with the per-patch strength ranking (`c5a0f0c`). The nested-loss code was removed.
Date: 2026-09-27

## Context

The pretrain loss trained the stamp decoder with two terms (ADR 0015):

- **Patch MSE:** the sum of all active stamps must match the patch. It trains the stamps jointly but
  does not care how they split the content, so two stamps may learn the same shape.
- **mp_loss (ADR 0011):** the anti-duplicate term. Stamps are graded one at a time in rank order,
  each against what the higher-ranked ones left over, with those detached. A copy of a higher-ranked
  stamp finds nothing left to explain and gains nothing.

Since the recipe moved to 16 shared (always-on) stamps and no routed ones, two things changed:

1. mp_loss ranked the shared stamps by their index. That imposed one fixed global hierarchy (stamp 0
   first in every patch) instead of matching pursuit's per-patch ranking by strength. This was
   unintended. Fixed in `c5a0f0c`: shared stamps now rank per patch by strength `h`, like routed ones.
2. With every stamp always on, the patch term and the anti-duplicate term can be one objective. ADR
   0015 proposed this merge (its experiment 2) and noted the weighting risk.

## Decision (proposed)

Replace patch MSE + mp_loss by one **nested reconstruction loss**:

```
L_patch = sum_k w_k * || x - (sum of the strongest s_k stamps) ||^2
```

- Stamps are ranked per patch by strength. Nothing is detached: every nested set is trained jointly.
- The last size is every active stamp, so that term is exactly the old patch MSE.
- The smaller nested sets carry the anti-duplicate pressure: a stamp that repeats a stronger one's
  content leaves the smaller set's error higher than one that covers something new.
- Config: `loss.nested_sizes` [2, 4, 8, 16], `loss.nested_weights` [0.125, 0.125, 0.25, 0.5],
  `loss.mp_weight` 0. Half the weight stays on the full reconstruction, so the first stamps cannot
  dominate (a plain mean over 16 sizes would give full reconstruction 1/16 of the loss). Unset keys
  keep the plain patch MSE.
- Position weights (masked 1, visible `unmasked_weight`, padding 0), the trial term and the MoE
  load balance are unchanged.

In the literature this is a Matryoshka-style nested loss (Matryoshka representation learning,
Matryoshka SAEs). Unlike those, the order here is per patch, by strength, not one fixed order.

## Test

`mesae_tiny_nested_s1` is the graded run `mesae_tiny_skipdrop_graded_s1` (skip_drop [0.75, 0.5, 0.25],
batch 16, no temporal bias, seed 1, tiny corpus) with only the loss change above. Then the probe
(`latent_signed`, pca, spatial_k 8) and the protocol stamp head on BNCI2014004 / 001 / 008 loso, and
`analysis_pretrain --preset quick` on both runs.

Adopt if all three hold, set before the run:

1. Masked val MSE (mean of epochs 41-50) at most 3% above graded (0.528 -> <= 0.544).
2. `stamp_duplicates`: fewer stamp pairs at similarity >= 0.9 than graded. Graded has 3, all among its
   slowest stamps (#0, #1, #3 at 0.98-0.99), despite mp_loss (index order at the time). Corrected
   2026-09-27: an earlier draft said "graded: none", which was the archived baseline's count.
3. Probe and stamp head not significantly worse than graded (Wilcoxon over subjects). Graded:
   probe 60.3 / 47.1 / 69.3, stamp head 62.9 / 44.5 / 69.9 on 004 / 001 / 008.

## Consequences

- If adopted: the templates set `nested_sizes` / `nested_weights` and `mp_weight` 0; mp_loss stays in
  the code only for routed-stamp recipes.
- If not: keep patch MSE + mp_loss, now with the per-patch strength ranking.
- Either way, next is a log multi-resolution STFT loss on masked blocks (added, not replacing), on
  top of whichever loss wins here.

## Result (2026-09-27)

`mesae_tiny_nested_s1` vs `mesae_tiny_skipdrop_graded_s1`, tiny corpus, seed 1.

| Criterion | Graded | Nested | |
|---|---|---|---|
| 1. masked val MSE (ep 41-50) <= +3% | 0.5277 | 0.5274 | pass |
| 2. fewer stamp pairs >= 0.9 | 3 (0.992, 0.986, 0.975) | 3 (0.995, 0.961, 0.928) | fail |
| 3. downstream not significantly worse | | BNCI2014001 stamp head 44.5 -> 39.4 (1/9, p = 0.02) | fail |

Other downstream cells within noise (pre-stamp probe 004 -2.6, 001 +0.1, 008 +0.0; stamp head 004
+1.5, 008 +0.7). Unmasked val MSE +7% (0.128 -> 0.136).

- Ranking stayed top-heavy: the leading stamp ranked first in 62% of patches (graded 72%); 4 stamps
  first in >= 5% of patches (graded 3).
- Near-duplicate pairs sit among the slowest stamps in both runs, so they look structural (DC-zeroed
  templates need several similar slow shapes to span within-patch drift), not something either loss
  controls. Not verified (would need a remove-one-stamp ablation).
