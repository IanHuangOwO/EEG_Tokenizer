# Card: stamp hidden vectors -- more than a copy of z? (diagnostic, no training)

Written 2026-09-29, before any number was computed.

- **Object:** the StampBank per-stamp MLP hidden `u_s = GELU(LN(z) W_down_s + b_down_s)` (hidden_width 8 per stamp and
  channel, 16 x 8 = 128 per token), which the stamp reads its gain pair from: `(a_s, b_s) = u_s w_amp_s + b_amp_s`.
  Not the StampBank output `h` (per-stamp strength [G, S]).
- **Question:** does `u` carry class information z does not make linearly available, or is it z seen through 16
  random lenses? Each `W_down_s` is trained only through its 2-dim readout `w_amp_s`, so ~6 of 8 dims face no
  reconstruction constraint.
- **Backbones:** `mesae_tiny_p50_s16_s1..3` (the three patch-50 seeds); `mesae_small_p50_s16_s1` reported, not judged
  (one seed). `u` is computed from the feature cache's z (the exact StampBank input, fp16) with the checkpoint's
  `input_norm`, `W_down`, `b_down`; a self-check recomputes the cached amp from the same z.

## Metrics

1. **Free share (M1):** per stamp, fraction of `u_s` variance (centred) in the null space of `w_amp_s` (the 6 dims
   the readout never sees); mean and range over stamps. Random directions give 6/8 = 0.75.
2. **Copy of z (M2):** R^2 of a ridge map z -> stacked `u` (held-out tokens); participation-ratio effective rank of
   `u` vs z; mean top canonical correlation between stamp pairs' `u_s` (1 = same subspace).
3. **Probes (M3, decides):** `ridge_probe` (loso) and `fewshot_ridge` with `u` in place of z (PCA 8 over the 128
   dims, same pipeline). Per condition mean +- SE over the 3 tiny seeds.

## Decision

- `u` beats z on >= 1 loso cell (|diff| > 2 x sqrt(SE_u^2 + SE_z^2)) and loses none: write an architecture card for
  per-stamp input (per-stamp spatial query or multi-scale stage input), the one design that can add information
  beyond z.
- Otherwise the per-stamp hidden idea is dropped; M1 / M2 record why (high free share, R^2 near 1 = copy of z).
- Few-shot cells are reported, not judged (loso-first rule on tiny corpora).

## Result (2026-09-29 15:00): dropped

`u` wins no loso cell and loses two (BNCI2014001 -3.3, threshold 2.1; BNCI2014008 -3.1, threshold 1.2; BNCI2014004
-3.4, threshold 3.8, tie); few-shot at or below z everywhere. Mechanism: ~60% of each stamp's hidden is never read
(random 75%), z -> u R^2 0.86-0.88, rank just under z's, stamp pairs overlap (canonical corr 0.63). The small backbone
is more of a copy (R^2 0.98, rank 2.4 vs 8.5, pair cc 0.83) and loses more (001 loso -8.1). Per-stamp hidden features
are a lossy re-expression of z; no architecture card follows. Report: docs/reports/2026-09-29-stamp-hidden.md.
