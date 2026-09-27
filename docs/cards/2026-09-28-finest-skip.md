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
