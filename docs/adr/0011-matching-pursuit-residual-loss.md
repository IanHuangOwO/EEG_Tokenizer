# Matching-Pursuit residual loss for stamp specialization (MeSAE StampBank)

## Context

Two failures, found in sequence, share one cause.

### The sparsity budget is a hard ceiling, not a knob

Raising `stamp_top_k` 12 -> 24 drove `recon_mse` to ~0.0000 on *every* dataset at
once, including unrelated ones. That is not a better representation, it is an
easier problem.

Each active slot contributes two free scalars (`a`, `b`) per channel. Active slots
per patch are `top_k + n_shared`. So the reconstruction has

```
2 * (top_k + n_shared)   degrees of freedom
```

against a `patch_len`-sample target. At `top_k=24`, `n_shared=4`, `patch_len=50`:
`2*28 = 56 > 50`. The active subset is over-determined relative to what it is
reconstructing, **regardless of which atoms were picked or what shape they hold**.
Gradient descent can solve the system per patch and hit ~0 loss without learning
any shared structure. Sparse coding requires the code to be genuinely
under-determined; past that line it stops being sparse coding.

Confirmed by the statistics collapsing in exactly the predicted direction:

| | top_k=12 | top_k=24 |
|---|---|---|
| mean excess kurtosis (activation) | 6.68 | 1.17 |
| off-diagonal \|correlation\| | 0.053 | 0.217 |

More slack -> more Gaussian, more mutually correlated activations: the
"arbitrary degenerate solution" corner rather than the source-like corner.

**Rule: keep `2 * (top_k + n_shared) < patch_len`, with margin.**

### Selection by amplitude cannot produce an incoherent dictionary

Even with the budget fixed, the selection rule has a structural blind spot.
`group_score_i = a_i² + b_i²` is a matched-filter score computed *independently per
atom against the same target*. If `D_i` and `D_j` are correlated, any signal with
energy in their shared direction scores *both* highly — so near-duplicates are
selected together, every time, by construction.

Matching Pursuit / OMP do not have this problem because they re-score against the
**residual** after each pick: a near-duplicate of an already-selected atom scores
~0 on what is left. Classical dictionary learning (K-SVD, MOD) never relies on the
selection step for incoherence either — it always pairs selection with a dictionary
update that reduces coherence directly. StampBank had only the selection half.

This is why "just add more data" is a weak fix: the reconstruction loss only
*mildly* disfavours wasted redundant capacity, and two atoms that co-adapt into
near-duplicates early in training sit at a symmetric fixed point that gradient
descent does not reliably escape. More data helped measurably but never closed the
gap (see table below).

## Decision

`mp_loss`, computed in `StampBank.forward`. It changes neither selection nor
`recon` — only the *training target* each routed slot is graded against.

1. Rank the `top_k` **routed** slots by `h` descending, per patch, independently.
   `h` is post-rms amplitude magnitude averaged over valid channels — "how much
   real energy this slot carries here". Note `h` is *not* `group_score`, which is
   pre-rms; rank 0 is therefore not necessarily the top-1 *selected* atom.
   Shared stamps are excluded: always-on baseline, not competing for content.
2. Grade rank 0 against the full target. Subtract its contribution (detached).
   Grade rank 1 against what remains. Continue through all `top_k` ranks, average.
3. `.detach()` on the residual is load-bearing — the residual is a *target*. Left
   attached, a slot could lower its loss by reshaping what it is judged against.

The anti-duplicate chain, which feeds machinery that already existed: a duplicate
ranked below a stronger atom faces a residual with that content removed -> its
best move is to output ~0 there -> its `amp` on that content shrinks ->
`group_score` (amp²-based) drops -> it stops being selected for it -> it drifts to
the dead threshold -> the existing `aux_loss` rescue re-aims it at unexplained
residual.

`mp_loss` is gated on `stamps_frozen` exactly like `aux_loss`: a frozen dictionary's
atoms cannot be reshaped, and leaving it on would add a second, unrelated objective
to the Masked stage.

## Results

Per-patch co-selected content overlap (spectral cosine between decoded segments of
atoms selected on the same patch; 0 = orthogonal, 1 = identical). All rows
`top_k=12`, 50 epochs, 2-dataset/6-subject slice unless noted:

| variant | overlap | activation corr |
|---|---|---|
| no regularizer | 0.432 | 0.063 |
| decorrelation only | 0.545 | 0.056 |
| negentropy only | 0.606 | 0.109 |
| decorrelation + negentropy | 0.611 | 0.067 |
| **mp_loss** | **0.275** | **0.054** |
| no regularizer, 3-dataset/32-subject | 0.411 | 0.061 |
| no regularizer, full 7-dataset (v1) | 0.302 | 0.053 |

`mp_loss` beats every alternative *and* the full-dataset reference, on a fraction
of the data, with reconstruction intact (`mse_patch` 0.0295).

### Rejected: ICA-style regularizers on activations

Decorrelation (off-diagonal correlation of activation series) and negentropy
(FastICA log-cosh surrogate, rewarding peaky/non-Gaussian activation) each did
improve the statistic they targeted — negentropy raised median kurtosis 0.35 ->
2.55. Both made *content* overlap worse.

They act on `dense_routed`: *when* and *how strongly* an atom fires. They never
touch *what it decodes*. Two atoms can have perfectly decorrelated firing patterns
while holding near-identical templates. Negentropy alone was actively
counterproductive (0.109 correlation, worse than no regularizer at all) — an atom
that is individually peaky says nothing about whether it duplicates another.

Deleted after measurement. The lesson generalises: regularize the variable the
problem is actually stated in.

### Rejected: direct spectral-overlap penalty

Penalizing pairwise spectral similarity between co-selected atoms was considered
and not built. Satisfying "low overlap across many pairs" is most easily achieved
by partitioning the frequency axis into disjoint narrow slices — i.e. collapsing
the bank into a filter bank, discarding the whole point of learned non-sinusoidal
atoms (a spike-wave complex legitimately spans several bands as one coherent
source).

`mp_loss` avoids this by construction: it never penalizes two atoms for *sharing*
content, it only declines to *reward* re-explaining what a stronger atom already
covered.

## Two follow-ons: one kept, one dropped

### Shared stamps join the residual chain — kept, and hardcoded

The first version ranked only the routed slots, leaving shared stamps outside the
chain entirely. That exemption was load-bearing in the wrong direction: a routed
atom ranked below an ungraded shared stamp was graded against a residual that still
contained that stamp's content, so it could still be rewarded for re-explaining it —
the exact flaw `mp_loss` exists to remove, displaced to the routed/shared boundary.
Worse, ungraded shared slots became a dumping ground for whatever `mp_loss` squeezed
out of routed, measured as line noise smearing across the shared pool.

Shared slots are now pinned ahead of routed in the chain (they are always-on, so
every reconstruction contains them regardless — subtract the baseline first, then ask
what is specific to this patch). Effect on 50Hz, the band with real signal:

| | owner | ownership | purity | shared-pool share | mse_patch |
|---|---|---|---|---|---|
| routed-only chain | routed 28 | 0.414 | 0.751 | 0.336 | 0.0295 |
| `mp_include_shared` | routed 22 | **0.661** | 0.787 | **0.163** | **0.0232** |

Shared-pool share of real line noise halved; one routed atom went from owning 41% to
66%; patch fidelity improved as well. No tradeoff appeared on any measured axis, so
this is not a knob — the flag was removed and the behaviour hardcoded.

### `exclusive_pool='shared'` — tried, measured, then dropped

`recon = recon - shared_recon + shared_recon.detach()` before the reconstruction
loss. Routed atoms are graded on the remainder after the always-on baseline's claim,
so re-emitting baseline content makes the sum overshoot rather than earn reward.
Shared keeps its own gradient through `mp_loss`.

The motivating hypothesis — that shared would then naturally absorb line noise — was
**wrong**, for the reason recorded in docs/adr/0010: line noise is dataset-conditional
and shared atoms are unconditional. 50Hz stayed with a routed atom (ownership 0.591,
purity 0.766).

Kept anyway, for an effect we were not looking for. The bank stops manufacturing
out-of-band energy:

| variant | 60Hz share of recon | vs raw (0.066%) | mse_trial | mse_patch |
|---|---|---|---|---|
| baseline | 0.150% | 2.3x | 0.0181 | 0.0295 |
| `mp_include_shared` | 0.183% | 2.8x | 0.0162 | 0.0232 |
| **`exclusive_pool='shared'`** | **0.087%** | **1.3x** | **0.0149** | 0.0319 |

60Hz is absent from the raw data here (verified model-free), so everything above the
raw share is invented. Exclusivity nearly eliminates it and gives the best trial-level
loss, at a patch-fidelity cost (0.0319 vs 0.0232).

**Dropped anyway.** A probe confirmed the win was real rather than an artifact of the
shared block shrinking (shared/total energy 0.906 either way, shared R^2 vs raw 0.906
vs 0.904), so this is not a case of the mechanism failing. It was removed because the
gain is confined to a metric — manufactured out-of-band energy — that only matters if
interpretability outranks reconstruction, while the cost lands on the primary
objective, and because detaching moves ~90% of the reconstruction (see the shared/raw
figure above) onto mp_loss alone. Keeping the MSE path plain was judged worth more
than the spectral cleanliness. `mp_include_shared` was hardcoded on at the same time;
it improved both metrics and had no such tradeoff.

## Consequences

- `mse_mp` **cannot reach zero**. Rank 0 alone can never equal the whole patch,
  rank 1 can never equal the whole remaining residual. It has a nonzero natural
  floor. Judge it by whether it falls *within* a run; never compare it to
  `mse_patch` (0.154 vs 0.030 is normal, not a failure).
- Adds a `[G, C, top_k, L]` materialization that the main `recon` einsum path
  deliberately avoids. Acceptable at current sizes; revisit if `top_k` or batch
  grows a lot.
- Pool size is not the lever it looks like. At 1.5x pool (96 stamps), live atoms
  (>=5 firings) stayed at 19-23, versus 19 at 64 stamps and 24 for v1 on the full
  dataset. Live-atom count tracks **data diversity**, not pool size; a bigger bank
  just adds dead weight (dead rate 0.70 -> 0.82). Stay at 64/4/12 and scale data
  instead.
- `decorr_weight` / `negent_weight` are gone from config. Their negative result is
  recorded here rather than as zero-weighted dead code.
