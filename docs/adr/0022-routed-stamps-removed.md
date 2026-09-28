# 0022: Routed stamps removed; the stamp dictionary is static

Date: 2026-09-28. Status: accepted. Supersedes the routed/shared split of 0007 and the rescue and
selection-mode parts of 0011.

## Decision

The StampBank keeps only always-on stamps: every stamp is active at every patch position. Removed from
main: routed stamps, top-k selection, the `gain` selection mode, the dead-stamp rescue (`aux_loss`,
`aux_weight`, `fire_ema`, `dead_threshold_frac`, `aux_k_cap_frac`), stamp-router health metrics and
the routed/shared viz panels (pool energy share, energy rank, pool ablation). The full routed code
lives on the `routed-stamps` branch.

Config: `stamp_bank` = `{n_stamps, hidden_width}` (+ the optional quantization keys). A static
checkpoint's build_config from before this change (`n_shared_stamps`, `stamp_shared_hidden_width`,
`n_routed_stamps` 0) still builds, and its empty routed tensors are dropped on load; a config or
checkpoint with routed stamps raises and points to the branch. `aux_weight` joins the removed loss keys.

Downstream heads read z (the encoder output), not the stamp code (finetune template default
`latent_signed`): the stamps are the reconstruction objective and an interpretable filterbank view.

## Evidence (tiny corpus, seed 1; docs/cards/2026-09-28-routed-stamps.md, 2026-09-28-patch100-capacity.md)

- Routing never became healthy: 45-56% of routed stamps dead at patch_len 50 (4+60 and 64 routed) and
  58% at patch_len 100, where the patch space has room for 64 distinct templates -- so the cause is
  the self-scoring selection (a stamp is scored by its own output and gets no gradient unless picked),
  not capacity; earlier rescue-weight changes did not fix it either.
- Reconstruction: within 3% of the static dictionary at equal active budget (patch 50), better by > 3%
  on 1 of 4 mask types only (patch 100).
- z probe: routed vs static within run-to-run spread (004 +3.7 at patch 50, spread 2.2; tie at 100).
- The stamp code's one gain over raw band power (P300, 8/8 subjects) sat in coefficient directions the
  reconstruction does not constrain -- encoder information from z, which the z probe reads directly
  and scores higher on.

Seeds 2-3 were not run: two independent designs failed the same pre-registered primary metric.

## Verification

Static-model outputs bit-identical before/after on a fixed CPU batch of mesae_tiny_notrial_s1
(eval forward with and without a mask, loss, StampExtractor codes, tokenizer-phase gradients, fresh
init); the D_shared gradient differs at ~1e-6 run to run on the old code as well.
