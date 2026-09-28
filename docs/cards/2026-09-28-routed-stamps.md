# Mechanism card: routed stamps vs the static dictionary (tiny corpus)

Written 2026-09-28, while the runs trained and before any of their metrics were computed (ADR 0020).

- **Change:** the stamp dictionary grows to 64 stamps with top-k routing, the active budget fixed at 16
  slots (2 x 16 = 32 < patch_len 50, as the static recipe):
  - A static: 16 shared, 0 routed (`mesae_tiny_notrial_s1`)
  - B: 4 shared + 60 routed, top-k 12 (`mesae_tiny_routed4s60r_s1`)
  - C: 64 routed, top-k 16, no shared (`mesae_tiny_routed64r_s1`)
  Routed variants use the dead-stamp rescue (`aux_weight` 0.03); mp ranks shared first, then routed by
  residual gain. Everything else as A.
- **Hypothesis (user, 2026-09-28):** with every stamp always on, the stamp code is a learned filterbank --
  its power carries no more class information than raw band power (measured on A before the routed runs
  finished: stamp - raw +2.4 on BNCI2014004, 4/9 subjects; -0.3 on BNCI2014001). Routing makes the code
  content-dependent (which stamps fire), so it should carry information a fixed filterbank does not.

## Metrics

1. **Primary -- stamp vs raw (`stamp_vs_raw`):** closed-form loso ridge on log stamp power minus the same
   ridge on log raw band power, same trials and Compass split, BNCI2014004 / 001 / 008. Pass: the routed
   variant's stamp - raw gap exceeds A's by >= 2 points on at least 2 of 3 datasets.
2. **Routing works (`stamp_usage`):** routed usage entropy >= 0.8, dead share <= 10%, and selection depends
   on content: dataset JS clearly above an untrained routed model's (0.009).
3. **Guard rail:** masked / unmasked val MSE and `backbone_eval` masked MSE within max(3 x spread, 3%) of A
   (spread = graded vs rankfix).
4. **Side:** ridge probe on z, attention range.

Downstream finetunes only for a variant that passes.
