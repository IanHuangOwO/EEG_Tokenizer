# Mechanism card: finest-only skips vs graded skip drop-path (tiny corpus)

Written 2026-09-28, before `mesae_tiny_finestskip_s1` finished training (ADR 0020). Evaluated
automatically by `decide.py` (docs/reports/2026-09-28-overnight/scripts/).

- **Change:** `skip_mode: finest` (only the finest skip; the two deeper skips removed; no drop-path)
  vs graded drop-path on all three skips (`mesae_tiny_notrial_s1`, same code and loss otherwise).
- **Hypothesis:** keeping the finest skip recovers what graded drop-path cost (channel imputation and
  visible detail: +20-85% band error on channel masks vs the no-drop baseline), while forcing
  everything coarser than a patch through the deep path keeps graded's time-gap gain (time-block
  masked MSE -10%, delta error -9%).
- **Metrics** (`backbone_eval`, 512 held-out windows, fixed masks):
  1. mean relative band error (5 bands) on random-channel + channel-cluster masks
  2. mean relative band error on the unmasked reconstruction
  3. time-block masked MSE
- **Spread:** graded vs `mesae_tiny_rankfix_s1` (near-identical runs), per metric.
- **Pass (finest wins) if all three:** 1 and 2 below the no-trial run by more than max(3 x spread, 5%);
  3 not above the no-trial run by more than max(3 x spread, 3%).
- **Consequence:** the winner's recipe goes to the small-corpus scale-up (card below).

## Result (2026-09-28 02:28): graded stays

| Test | Change vs no-trial | Threshold | Pass |
|---|---|---|---|
| channel-mask band error | -27.8% | < -7.6% | yes |
| unmasked band error | -93.4% | < -26.8% | yes |
| time-block masked MSE | +13.9% | < +3.1% | no |

Finest-only recovers imputation (channel masked MSE 0.251, baseline level), visible detail and seams
(0.066 -> 0.014), but loses temporal inference (time-block MSE 0.888, worse than the no-drop baseline's
0.858) and the ridge probe drops (004 / 001 / 008: 65.8 / 34.1 / 57.9 vs 69.4 / 38.7 / 69.8; P300 -12).
Full report: docs/reports/2026-09-28-overnight.md.

## Recorded numbers (runs archived 2026-09-29)

From each run's `pretrain/analysis/` (now `output/archive/<topic>/<run>/`): backbone_eval masked MSE and log-spectral
distance per test mask, seam disagreement, masked MSE under an ablation as a multiple of the baseline,
attention_range block means, stamp_usage, ridge probes. The graded run (notrial) stays live.

| | graded (notrial) | finest-only |
|---|---|---|
| masked MSE token_runs | 0.405 | 0.303 |
| masked MSE random_channel | 0.318 | 0.251 |
| masked MSE channel_cluster | 0.339 | 0.276 |
| masked MSE time_block | 0.780 | 0.888 |
| masked MSE motor3_to_bci22 | 0.256 | 0.205 |
| seam disagreement | 0.0660 | 0.0143 |
| log-spectral dist token_runs | 1.202 | 0.859 |
| log-spectral dist time_block | 1.755 | 1.632 |
| log-spectral dist random_channel | 0.950 | 0.830 |
| ablation skips_off (x baseline) | 1.90 | 3.26 |
| ablation coords_shuffle (x baseline) | 1.65 | 2.27 |
| ablation time_shuffle (x baseline) | 1.18 | 1.04 |
| temporal attn mean |dt| s (block mean) | 0.69 | 1.55 |
| spatial attn dist ratio (block mean) | 0.43 | 0.78 |
| stamps ranked first >= 5% | 3/16 | 4/16 |
| redundant stamps (remove cost < 1%) | 0 | 0 |
| ridge probe on z, 004 / 001 / 008 | 67.6 / 40.0 / 69.2 | 65.8 / 34.1 / 57.9 |
| stamp - raw ridge, 004 / 001 / 008 | +2.4 / -0.3 / +0.0 | - |
| stamp power ridge net / clean / ridge-fit (008) | 55.2 / 54.2 / 54.3 | - |
