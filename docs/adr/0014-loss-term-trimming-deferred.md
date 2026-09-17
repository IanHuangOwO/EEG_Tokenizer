# 0014 — Loss-term trimming (deferred)

Status: Deferred. Revisit only after the finetune-head work (the exposed features, the
pooling, and the raw-baseline comparison) is settled.
Date: 2026-09-17

## Context

The MeSAE pretrain loss currently has five terms (`MeSAE.get_loss`):

| term | config weight | what it does |
|---|---|---|
| patch MSE | `mse_patch_weight` | each patch reconstructs itself |
| trial MSE | `mse_trial_weight` | overlap-added continuous trial reconstructs |
| mp_loss | `mp_weight` | rank-ordered residual credit per patch (ADR 0011) |
| aux_loss | `aux_weight` | dead-atom rescue |
| ffn_lb | `ffn_lb_weight` | MoE FFN load balance |

In the masked phase, patch MSE, trial MSE and `mp_loss` all use the same per-position
weights: 1 on masked positions and `unmasked_weight` on visible ones (commits `9bfac32`,
`52cda87`).

The question was whether some of these terms are redundant. None were removed. What
follows was worked out during the discussion but never measured.

## Findings from the discussion

### mp_loss and patch MSE are not the same loss

With `r_m = x − Σ_{j<m} c_j`, mp term m equals `‖x − Σ_{j≤m} c_j‖²`. That is the error of
the first-m-atoms reconstruction. So:

- **Values overlap.** `mp_loss` is the mean prefix-reconstruction error over prefix
  lengths 1..K. Its last prefix (all atoms) equals patch MSE in value.
- **Gradients differ.** The higher-rank contributions are detached, so term m trains
  only atom m. That is what denies duplicates their reward (overlap 0.432 → 0.275 in
  ADR 0011). Patch MSE trains all atoms jointly and does not care how they split the
  content.

**Possible merge.** Drop the detach and use
`L = mean_k ‖x − Σ_{j≤k} c_j‖²`. This is Matryoshka-SAE style, and k = K is exactly
patch MSE, so the two terms would become one. It still penalizes duplicates, but
through a different mechanism, because the gradient now also moves the higher-ranked
atoms. The ADR 0011 overlap metric has to be re-measured before this replaces
`mp_loss`.

### No trial-level mp_loss

Rank order exists only within a patch: each patch picks and ranks its own top-k. Rank m
in neighbouring patches is usually a different atom, so overlap-adding rank-m
contributions mixes unrelated atoms. Duplicates are a within-patch problem. The only
thing a trial-level mp term would add is seam consistency, and trial MSE already
provides that. Rejected.

### Patch MSE stays; trial MSE is a removal candidate

The overlap-added trial is a crossfaded average of overlapping patches:

- If every patch is perfect, trial error is 0.
- Trial error is at most the weighted patch error. That is why the logged `mse_trial`
  sits below `mse_patch` (for example 0.020 vs 0.037 at the end of the tokenizer phase
  in `mesae_v10_small_uw01`).

Beyond patch MSE, trial MSE adds two things: it down-weights patch edges, and it rewards
neighbouring patches whose errors cancel.

- **Patch MSE cannot go.** A token must faithfully encode its own patch. With trial MSE
  alone, a patch could be wrong at its edges while its neighbour compensates, so the
  stitched plots would look fine while the individual tokens are wrong.
- **Trial MSE is optional.** It was added in `26ce2d1` to give cross-patch context
  training pressure for the oscillator atoms, which ADR 0010 withdrew. Patch MSE
  already backpropagates through temporal attention, so the only unique effect left is
  seam consistency. It was never ablated.

## Planned experiments (in order, one change per run)

1. **`mse_trial_weight: 0.0`.**
   - Compare the logged plain `mse_trial` against the reference run. It is still logged
     when its weight is 0.
   - If it barely rises, drop the term. If seams appear, keep it.
2. **Matryoshka prefix loss replacing `mp_loss` and patch MSE.**
   - Re-measure the per-patch co-selected overlap (ADR 0011 table), `mse_patch`, and the
     dead-feature rate.
3. **Revisit `aux_weight` / `ffn_lb_weight`** only if the terms above change collapse
   behaviour.

Use the same small 7-dataset × 3-subject config as the v10 small runs for every
experiment, and use the fixed-mp run (`mesae_v10_small_uw01_mpw`) as the reference.

## Consequences

Nothing changes now. The terms stay as they are until the finetune head is settled
(ADR 0012 open items).
