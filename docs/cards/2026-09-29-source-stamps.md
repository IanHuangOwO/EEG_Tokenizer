# Card: source-factorized stamps (spatial_rank K = 4), one seed pilot

Written 2026-09-29, before the run. Follows the stamp-hidden card (per-stamp hidden = copy of z; the stamps separate
waveform but not the spatial mixing of sources).

- **Change:** `stamp_bank.spatial_rank` 4. Each stamp's per-channel complex gain column is forced to rank K:
  `src_sk = mean_c W_s[c, k] g_cs` (unmixing), `g_cs <- sum_k A_s[c, k] src_sk` (mixing), with W and A fixed functions of
  electrode position (Fourier coordinate features -> MLP, any montage). `A_s[:, k]` is source (s, k)'s topography,
  `src_sk` its (a, b) activation: per patch 16 x 4 complex numbers instead of C x 16. Everything else as
  `mesae_tiny_p50_s16_s2` (same config, data and seed 2): run `mesae_tiny_p50_s16_k4_s2`. No sparsity prior beyond
  mp_loss in this pilot.
- **Hypothesis:** pretraining learns the spatial unmixing the few-shot heads cannot learn from ~20 trials per class;
  the stamp code becomes source-level (interpretable) and more linearly usable, most on MI (CSP-like content).

## Metrics (one seed: a pilot, "promising or not", not a verdict)

1. **Mechanism:** masked reconstruction MSE (backbone_eval) against the three patch-50 seeds (a rank-4 column cannot
   fit channel-local content, so some cost is expected: report it); topographies A rendered per stamp (smooth,
   plausible scalp maps vs noise).
2. **Downstream, loso primary:** stamp head (`stamp_power`, learned time pool, spatial_k 8 -- the `cw_stamp` head) and
   z probe (`patch_probe`) on BNCI2014004 / 001 / 008, finetune seeds 1-3, against the same heads on
   `mesae_tiny_p50_s16_s2` (paired: same seed and data). Few-shot cells reported.
3. **Promising if:** the K4 stamp head beats the p50_s2 stamp head by more than the patch-50 seed spread on >= 1 MI
   loso cell and loses none beyond it; or the K4 stamp head closes the gap to the z probe on MI loso. Then 2 more
   seeds (and a sparsity prior / K sweep) follow as a full card. Otherwise the factorization is recorded and shelved.
