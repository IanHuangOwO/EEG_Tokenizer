# Mechanism card: finest-only skips vs graded skip drop-path (tiny corpus)

Written 2026-09-28, before `mesae_tiny_finestskip_s1` finished training (ADR 0020). Evaluated
automatically by `output/queue/overnight/decide.py`.

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
Full report: output/reports/overnight/decision.md.
