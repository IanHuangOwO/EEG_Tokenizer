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
