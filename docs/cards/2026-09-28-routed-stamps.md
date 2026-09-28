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

## Results (2026-09-28)

| | A static | B 4s+60r | C 64r |
|---|---|---|---|
| stamp - raw, BNCI2014004 | +2.4 (4/9) | -4.2 (3/9) | +2.0 (5/9) |
| stamp - raw, BNCI2014001 | -0.3 (6/9) | -0.8 (3/9) | +0.3 (5/9) |
| stamp - raw, BNCI2014008 | +0.0 (5/8) | +6.9 (8/8) | +8.4 (8/8) |
| routed usage entropy / dead share / dataset JS | -- | 0.73 / 45% / 0.038 | 0.74 / 56% / 0.019 |
| ridge probe on z, 004 / 001 / 008 | 67.6 / 40.0 / 69.2 | 71.4 / 40.2 / 69.2 | 71.2 / 39.8 / 69.6 |
| backbone_eval masked MSE, token_runs / time_block / random_channel | 0.405 / 0.780 / 0.318 | 0.412 / 0.761 / 0.312 | 0.392 / 0.779 / 0.313 |

Raw band power on 008 uses FFTs zero-padded to 1 s (its 1 s trials leave 250 ms segments).

**Verdict: neither routed variant passes.** Primary: gap over A >= 2 points on 1/3 datasets each (008 only;
B -6.6 on 004). Routing health fails for both (entropy < 0.8, half the routed stamps dead), though dataset JS
is above the untrained 0.009, so selection does depend on content. Guard rail passes (masked MSE within 3%
of A or better). Side: probe on z +3.6 / +3.8 on 004, within noise elsewhere.

The one consistent signal is P300: routed stamp power beats raw band power on 8/8 subjects (+6.9 / +8.4),
where static is +0.0. The MI hypothesis (routing adds class information beyond a filterbank) is not
supported; the dead-stamp share suggests the rescue (`aux_weight` 0.03) is too weak for 60+ routed stamps.
