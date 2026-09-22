# Idea to revisit: drop routed/shared split, all stamps always-on

Date: 2026-09-22. Status: not scheduled, parked for later. No code written. Backbone/pretrain
change, out of scope for the current finetune-head restructure branch.

## The idea

Remove the routed-vs-shared distinction in `StampBank` entirely; make every stamp always-on
(no top-k selection). Reasoning offered: an atom that isn't suitable for a given patch/channel
should just learn near-zero `a, b` amplitude, so it fades out on its own rather than needing
explicit dead-atom bookkeeping. With `mp_loss` already discouraging duplicated atoms, dead
atoms shouldn't be a problem, and `dead_threshold`/`aux_loss`/`fire_ema`-as-selection-frequency
become unnecessary. A stamp becomes the smallest temporal unit (a phoneme-like primitive);
which channels it's expressed in (its per-channel `a, b`) gives it meaning, matching the
existing per-channel gain design.

## Where this is right

`dead_threshold`/`aux_loss` exist specifically to rescue atoms that get literally zero
gradient under hard top-k competitive selection — never selected means never updated. If every
stamp is always computed (no competitive selection step), that exact failure mode disappears
by construction, and the rescue machinery genuinely becomes unnecessary. That part is correct.

## The real risk: the sparsity budget is a hard ceiling, not an emergent property

`docs/adr/0011-matching-pursuit-residual-loss.md` and `CLAUDE.md`'s sparsity-budget section
establish, with measurements from this repo:

```
2 * (top_k + n_shared) < patch_len,  with margin
```

is enforced *architecturally* by top-k: only `top_k + n_shared` slots can be nonzero per
patch, period. It is not a statement about how many atoms *usually* fire — it is a hard cap on
how many the optimizer is *allowed* to use. Raising `top_k` from 12 to 24 (DOF 56 > patch_len
50) drove `recon_mse` to ~0 on every dataset at once, while activation kurtosis fell 6.68 ->
1.17 and cross-atom correlation quadrupled: the model did not learn a sparse, source-like code
with the extra slack, it used the freedom to exactly overfit every patch. `mp_loss` does not
prevent this — it is an anti-duplication/ranking objective that operates *within* an
already-selected top-k, not a sparsity-inducing penalty (no L1/L0 pressure on `a, b` amplitude
exists anywhere in the current loss). Making every stamp always-on removes the hard cap
entirely: with `n_stamps` say 64, that is `2 * 64 = 128` degrees of freedom per patch against
`patch_len = 50`, well past the line that already caused collapse at 56.

## Existing (inconclusive) evidence: `mesae_v10_all_share`

`output/pretrain/mesae_v10_all_share/` is architecturally close to this proposal already:
`n_shared_stamps=0`, `n_routed_stamps=32`, `stamp_top_k=32` — top_k equals the whole pool, so
every stamp is selected every patch (`stamp_router_entropy_frac: 1.0000`, `dead_feature_rate:
0.0000` in its logs). DOF = 64, worse than the 56 that already collapsed. It was flagged
elsewhere as an abandoned, uncommitted side branch and was never run through the kurtosis/
cross-atom-correlation diagnostic ADR 0011 used to detect the collapse. Its logs show
`mse_patch ~= 0.072`, which does not look collapsed to zero, but that reads as under-trained
rather than evidence the failure mode was avoided — not a clean answer either way. One point
worth following up: its logs also carry a `k_eff` metric (effective sparsity, ~14-15) already
close to what a working "soft sparsity" would look like, worth digging into before dismissing
the idea.

## The phoneme framing argues for keeping a hard selection step, not dropping it

Phoneme inventories are categorical: a discrete on/off choice, not "mostly silent, occasionally
leaking a little energy." The part of the framing worth keeping is dropping the routed-vs-shared
*naming* split (unify into one pool, meaning comes from the per-channel spatial pattern) —
without also giving up a hard cap on how many atoms can be active per patch. That cap can stay
as top-k, or move to a differentiable substitute (entmax/sparsemax, learned hard-concrete/L0
gates) if a harder discrete gate is undesirable, while keeping the collapse risk bounded.

## If revisited

A proper controlled test, not a full retrain: rerun the exact kurtosis/cross-atom-correlation/
`recon_mse` diagnostic from ADR 0011 against a dense, no-top-k (or unified-pool-with-a-cap)
config, on the same small data slice ADR 0011 used. Check `mesae_v10_all_share`'s existing
checkpoint against that diagnostic first — cheaper than a fresh run and may already answer the
question.

## Why parked

Backbone/pretrain architecture change; out of scope for the current finetune-head restructure
branch (`restructure/finetune-head`) and the restart baseline in progress. Revisit once those
are settled.
