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

## Follow-up: where the routed gain lives (2026-09-28)

The stamp coefficients are not a least-squares fit: on held-out pretrain windows 27-38% of their energy
(A / B / C) lies in near-null directions of the selected templates (singular value < 0.1 s_max), which
barely change the reconstruction (ridge fit: 0.3-5%, chance ~12%). Loso ridge on log stamp power from the
finetune code (every kept stamp) as-is, with that component removed ("clean"), and from a ridge fit of the
raw patch onto the same templates:

| | 004 net / clean / ridge | 001 | 008 |
|---|---|---|---|
| A static (16 stamps, cond 359) | 73.8 / 73.1 / 74.9 | 33.6 / 34.0 / 33.4 | 55.2 / 54.2 / 54.3 |
| C 64r (33 kept, 66 cols > 50, cond 1831) | 73.4 / 70.8 / 73.4 | 34.3 / 32.0 / 34.1 | **63.6** / 54.8 / 54.6 (clean < net 8/8) |

Static: the invisible part is neutral; the code is a learned filterbank (a ridge fit onto its templates
scores the same). Routed: the whole P300 gain over raw band power sits in the invisible part -- encoder
information carried by coefficients the reconstruction does not constrain; removing it (an L2 on (a, b), or
a ridge-fit head) would remove the gain. The probe on z still scores higher (008: 69.6).

## Recorded numbers (runs archived 2026-09-29)

From each run's `pretrain/analysis/` (now `output/archive/<topic>/<run>/`): backbone_eval masked MSE and log-spectral
distance per test mask, seam disagreement, masked MSE under an ablation as a multiple of the baseline,
attention_range block means, stamp_usage, ridge probes. A and rankfix stay live as references.

| | A static (notrial) | rankfix (spread) | B 4s+60r | C 64r |
|---|---|---|---|---|
| masked MSE token_runs | 0.405 | 0.412 | 0.412 | 0.392 |
| masked MSE random_channel | 0.318 | 0.317 | 0.312 | 0.312 |
| masked MSE channel_cluster | 0.339 | 0.338 | 0.333 | 0.335 |
| masked MSE time_block | 0.780 | 0.784 | 0.761 | 0.779 |
| masked MSE motor3_to_bci22 | 0.256 | 0.253 | 0.245 | 0.245 |
| seam disagreement | 0.0660 | 0.1080 | 0.0587 | 0.0575 |
| log-spectral dist token_runs | 1.202 | 1.166 | 1.188 | 1.191 |
| log-spectral dist time_block | 1.755 | 1.718 | 1.761 | 1.809 |
| log-spectral dist random_channel | 0.950 | 0.880 | 0.899 | 0.929 |
| ablation skips_off (x baseline) | 1.90 | 1.87 | 1.80 | 1.86 |
| ablation coords_shuffle (x baseline) | 1.65 | 1.64 | 1.63 | 1.71 |
| ablation time_shuffle (x baseline) | 1.18 | 1.17 | 1.23 | 1.16 |
| temporal attn mean |dt| s (block mean) | 0.69 | 0.69 | 0.72 | 0.71 |
| spatial attn dist ratio (block mean) | 0.43 | 0.42 | 0.47 | 0.41 |
| stamps ranked first >= 5% | 3/16 | 5/16 | 2/64 | 4/64 |
| redundant stamps (remove cost < 1%) | 0 | 1 | 44 | 43 |
| ridge probe on z, 004 / 001 / 008 | 67.6 / 40.0 / 69.2 | 69.8 / 37.8 / 69.8 | 71.4 / 40.2 / 69.2 | 71.2 / 39.8 / 69.6 |
| stamp - raw ridge, 004 / 001 / 008 | +2.4 / -0.3 / +0.0 | - | -4.2 / -0.8 / +6.9 | +2.0 / +0.3 / +8.4 |
| stamp power ridge net / clean / ridge-fit (008) | 55.2 / 54.2 / 54.3 | - | - | 63.6 / 54.8 / 54.6 |
| routing entropy / dead / dataset JS | - | - | 0.73 / 45% / 0.038 | 0.74 / 56% / 0.019 |
