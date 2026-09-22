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

## Follow-up: what if the total pool is shrunk to respect the DOF cap?

Asked in discussion: if `n_stamps` (all always-on, no top-k) is small enough that
`2 * n_stamps < patch_len` with margin, does the collapse mechanism above still apply?

**No, not the DOF-collapse mechanism specifically** — if the always-on pool is sized so its
DOF matches or stays under today's actual budget (e.g. `n_stamps ~= 16` for DOF 32, matching
`2*(top_k=12 + n_shared=4)` today), the reconstruction is equally under-determined whether the
active slots were chosen dynamically per patch or are a fixed always-on set of the same size.
The math is the same either way, so gradient descent has no more DOF slack to exploit than it
does today. Caution: "32 stamps, all on" is DOF 64 (worse than the 56 that already collapsed)
— easy to confuse with the DOF number 32 in `CLAUDE.md`'s "current default sits at 32", which
refers to `2*(top_k+n_shared)`, not a stamp count. A DOF-safe always-on pool is closer to
`n_stamps ~= 16`, not 32.

**What does not go away is a separate, representational question.** Today's 64-atom pool with
top-k=16 selects a *content-appropriate subset* per patch (different atoms for artifacts,
mu-rhythm, line noise, different datasets' characteristics — the motivation for a big shared
vocabulary across the 7 pretraining datasets, ADR 0009). A fixed always-on pool of ~16 removes
that per-patch adaptivity: the same 16 atoms must jointly cover every patch of every dataset,
with no ability to pick a different subset for different content. Whether that costs anything
is untested, not implied one way or the other by the DOF argument.

One suggestive data point, from ADR 0011's own Consequences: live-atom count (atoms that ever
fire >=5 times) stayed at 19-23 whether the pool was 64 or 96 stamps ("a bigger bank just adds
dead weight; live-atom count tracks data diversity, not pool size"). That hints the data's real
diversity may only need something in the ~16-24 range, which is close to a DOF-safe fixed pool
— but it is a statement about total-ever-used atoms across the whole tracked period, not about
whether per-patch content-adaptive choice among them matters.

`mesae_v10_all_share` (n=32 all-on, DOF=64) is still not clean evidence for the DOF-safe
variant, since it sits in the unsafe regime. A real test of the shrunk-pool idea needs a fresh
small run at roughly `n_stamps=16`, `top_k=n_stamps` (all-on), `n_shared=0`, checked against
the same kurtosis/cross-atom-correlation diagnostic, separately from whatever the existing
checkpoint shows.

## Correction: the reconstruction number quoted earlier was the wrong one

In discussion, `mesae_v10_small`'s end-of-training `mse_patch` (0.106, logged during masked-
phase training) was cited as "how well 25 alive stamps reconstruct EEG." Pushback: that number
reflects the masked-prediction task's difficulty, not the architecture's raw capacity, because
`_position_weights` grades masked positions at weight 1.0 and visible ones at `unmasked_weight
= 0.1` — the logged number is dominated by blind prediction, not faithful reconstruction.

Checked directly: loaded the final trained checkpoint (epoch 40, masked-phase weights, spatial
attention on) and ran one forward pass with `bool_masked_pos=None` (no masking at all, same
weights). Result: `mse_patch = 0.0119` on BNCI2014001 subject 9 (not in this backbone's
pretraining set). For reference, three numbers from the same run:

| Setting | `mse_patch` |
|---|---|
| End of masked-phase training (epoch 40, logged, under the mask curriculum) | 0.1058 |
| End of tokenizer phase (epoch 10, unmasked, but no spatial attention yet) | 0.0373 |
| **Final trained weights, unmasked eval (this check)** | **0.0119** |

The architecture's DOF ceiling (`2*(top_k+n_shared) = 32` real numbers per channel, against a
50-sample patch) is identical in both the masked and unmasked eval — masking does not change
how many atoms can be active, only which positions the loss weights. So the ~9x gap between
0.106 and 0.012 is (mostly) the masked-prediction task, not a capacity limit of the ~21 alive
routed + 4 shared stamps. Correction: cite the unmasked number, not the masked-phase training
log, when asking how well the stamp code itself can represent EEG.

### Hypothesis: why so few atoms suffice, given the correction

`MeSAEPretrain.stage_features` runs the *whole* `[B, C, N, D]` trial tensor (every channel,
every patch) through `TSAEncoder`, whose blocks apply temporal attention (across all N patches)
then spatial attention (across all C channels) before `a, b` are derived (`dense_amp` reads the
encoder's output, not a per-patch-isolated feature). So the `32-DOF-vs-50-sample` framing per
patch is misleading in isolation: the encoder is not blindly compressing one 50-sample window
into 32 numbers — it is choosing 32 well-informed numbers for that window *given the entire
1000-sample, all-channel trial as context* via bidirectional attention. Two concrete channels
for that context to help:

- **Temporal:** slowly-varying rhythms (e.g. an alpha oscillation) repeat similar structure
  across neighboring patches, so the encoder can use those neighbors to sharpen its estimate of
  one patch's amplitude/phase rather than fitting 32 numbers to 50 samples in isolation.
- **Spatial:** nearby electrodes are correlated (volume conduction), so other channels' signal
  at the same time can sharpen one channel's estimate.

This is consistent with, and gives a mechanism for, why masking hurts so much: masking removes
exactly this borrowed context for the positions being graded, and the masking curriculum
(`random_to_complementary`) increases masking over training, shrinking available context
further — matching both the sharp jump at the tokenizer/masked-phase boundary (0.037 -> 0.089)
and the further drift to 0.106 by epoch 40.

**Not yet checked, falsifiable if revisited:** feed the encoder a single patch with corrupted
or zeroed neighboring context and see whether reconstruction degrades toward the naive
32-vs-50 compression limit; that would directly confirm context (not dictionary quality alone)
is doing most of the work.

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
