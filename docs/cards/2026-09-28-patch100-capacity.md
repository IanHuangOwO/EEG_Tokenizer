# Mechanism card: patch_len 100 -- is the stamp dictionary capped by the patch space? (tiny corpus)

Written 2026-09-28, before any of these runs trained (ADR 0020).

- **Hypothesis (user, 2026-09-28):** at patch_len 50 a stamp spans 2 of 50 dimensions (template + quadrature
  partner), so ~23 stamps fill the patch space. Past that, templates cannot be distinct: static dictionaries
  above ~23 grow duplicates, and at 64 routed half the stamps die (45-56%) with cos 0.99-1.00 pairs -- a
  capacity limit no rescue weight fixes. A 100-sample patch (0.5 s, 2 Hz resolution) leaves room for more
  distinct stamps, so routing should become healthy and useful there.
- **Arms** (all patch_len 100, stride 50, 19 patches per 5 s window; everything else as
  `mesae_tiny_notrial_s1`, batch 16, seed 1):
  - S32: static 32 shared (active DOF 64) -- `mesae_tiny_p100_s32_s1`
  - S16: static 16 shared (DOF 32) -- `mesae_tiny_p100_s16_s1`
  - R: 64 routed, top-k 16, no shared (DOF 32), `aux_weight` 0.03 -- `mesae_tiny_p100_r64k16_s1`
- Compared only with each other: tokens, encoder stage lengths (19 -> 10 -> 5 -> 3), time resolution and the
  minimum masked run (3 patches = 1 s) all change with the patch length.

## Metrics

1. **Capacity (primary):** R's routed dead share <= 20% (64r at patch 50: 56%), and fewer near-duplicate
   pairs (nearest-neighbour cos >= 0.95, `stamp_duplicates`) in R and S32 than in the patch-50 64r run.
   Fails -> capacity is not the cause (the self-scoring selection is); routed is branched off.
2. **R vs S32:** `backbone_eval` masked MSE within 3% of S32 on every mask type (spread notrial vs rankfix:
   0.6-1.7%) -> top-k 16 of 64 matches twice the active DOF.
3. **R vs S16 (equal budget):** R's masked MSE beats S16's by > 3% on >= 2 of 4 mask types -> routing earns
   its place.
4. **Side:** routing entropy / dataset JS; ridge probe on z for BNCI2014004 / 001 (008's 1 s trials give 3
   patches: reported, not judged).

If 1 and 3 pass: the seed programme (seeds 2, 3) runs at patch 100 on S16 vs R. If 1 fails: routed code moves
to a branch.

## Results (2026-09-28)

| | S32 | S16 | R (64r, k16) | patch-50 64r (ref) |
|---|---|---|---|---|
| routed dead share / usage entropy / dataset JS | -- | -- | **58%** / 0.74 / 0.018 | 56% / 0.74 / 0.019 |
| stamps with a nearest neighbour cos >= 0.95 | 3/32 | 2/16 | 12/30 (listed) | -- |
| masked MSE token_runs / random_channel / channel_cluster / time_block | 0.476 / 0.327 / 0.349 / 0.833 | 0.459 / 0.351 / 0.367 / 0.843 | 0.432 / 0.343 / 0.360 / 0.838 | -- |
| ridge probe on z, 004 / 001 / 008 | 68.7 / 43.1 / 65.9 | 72.6 / 40.6 / 66.8 | 71.4 / 41.6 / 67.3 | -- |
| stamp - raw, 004 / 001 / 008 | -1.0 / -0.2 / -0.0 | -1.1 / -3.1 / +0.9 | -2.5 / -0.2 / +4.0 | -- |

**Verdict: fails at the primary.** R's dead share is 58% at patch 100, the same as at patch 50 (56%): the
patch space is not what kills routed stamps -- the self-scoring selection is. R vs S32: within 3% on
time_block only (token_runs -9%, random_channel +5%, channel_cluster +3%). R vs S16 (equal budget): better
by > 3% on 1 of 4 masks (token_runs -6%). z probe: R and S16 tie (spread 2.2).

The capacity part of the hypothesis holds for static dictionaries: S32 at patch 100 has 3 near-duplicate
stamps (patch 50 grows duplicates above ~23) -- but 32 static stamps reconstruct no better than 16.
Per the card, routed code moves to a branch.
