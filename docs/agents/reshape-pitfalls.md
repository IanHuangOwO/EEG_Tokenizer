# Reshape/view pitfall: silent axis scrambling

`tensor.reshape(...)`/`.view(...)` never transposes anything. It reinterprets the
tensor's *current* logical dimension order as a flat sequence, then rechunks that
sequence into the new shape. If the new shape implies two axes should swap places
relative to each other, or merge together, but those axes are not already adjacent
in that order, reshape still "succeeds" — it just silently produces the wrong
data instead of raising an error. There is no exception, no NaN, no shape
mismatch — just quietly corrupted tensors that still train (badly).

## The rule

Before calling `.reshape(...)`/`.view(...)`:

- **Merging two axes into one** — safe only if they are already adjacent in the
  tensor's current dimension order. `(A, B, C).reshape(A*B, C)` is fine. `(A, B,
  C).reshape(A*C, B)` is NOT fine (silently scrambles B and C) — permute to
  `(A, C, B)` first, *then* merge.
- **Splitting one axis into two** — always safe, regardless of what's around it.
  `(A, B, C).reshape(A, B1, B2, C)` (B == B1*B2) never needs a permute.
- **Flattening all trailing/all leading dims** (`x.reshape(x.shape[0], -1)` or
  `x.reshape(-1, x.shape[-1])`) — always safe, same reasoning as the merge case
  but trivially adjacent by construction.

When in doubt, write the permute explicitly and let `reshape` only ever merge or
split axes that are already next to each other. Test with `torch.arange(...)`
and small dummy dims — the bug is invisible in shape asserts (shapes always
"match") and only shows up by checking actual values.

## Instances found and fixed (2026-08-05/06 session)

All three were `reshape` calls that merged/reordered non-adjacent axes without a
preceding permute:

1. **`IO/preprocessing.py` `window_continuous_signal`** (originally `IO/dataset.py`'s
   `EEGDataset._window_subject_signal`, moved verbatim in a later refactor) —
   `trials.reshape(N * T, C).T`
   on a `(N, C, T)` tensor. Intent: concatenate each subject's trials end-to-end
   per channel, to produce `(C, N*T)`. Actual effect: scrambled channel and time
   together, since `C` sits between `N` and `T` in memory and reshape merged `N`
   and `T` as if they were adjacent. **Every tokenizer-stage and pretrain-stage
   training run** (`assemble_trials=True`, the default for `mode='pretrain'`/
   `mode='tokenizer'`) trained on this corrupted signal. `FinetuneDataset` (`assemble_trials=False`)
   was not directly affected, but any backbone finetuned from a tokenizer/pretrain
   checkpoint inherited weights trained on the corrupted data.
   Fix: `trials.permute(1, 0, 2).reshape(C, N * T)` — permute first so the merge
   only ever combines the now-adjacent `N`,`T`.

2 & 3. **`model/MeSAE/MeSAE.py` `_patch_pyramid_levels`** (feeds
   `_hierarchical_recon_loss`, the actual training loss for MeSAE tokenizer +
   pretrain stages) — two separate instances in the same function:
   - `recon.reshape(B*C, L, N)` on a `(B, C, N, L)` tensor, intending to swap `N`
     and `L` to put the patch axis last for `avg_pool1d`. Scrambled within-patch
     samples across patch boundaries instead.
   - `rp.reshape(B, C, -1, L)` after `avg_pool1d`, on a `(B*C, L, N//win)`
     tensor, intending the inverse swap back. Same trap.

   Together these meant the reconstruction loss gradient itself was computed
   over scrambled data — corrupting training independently of bug #1, and would
   have kept corrupting it even after #1 was fixed.

   Fix: rewrote to never need a mid-pyramid transpose at all — merge `B,C`
   once (the one unavoidable copy, since `recon` arrives non-contiguous from an
   upstream permute), then every pyramid level only **splits** the `N` axis in
   place (`reshape(B*C, n_groups, win, L)` + `.mean(dim=2)`), which is always
   safe per the rule above. Verified against the old (correct but slower)
   permute+`avg_pool1d` implementation — bit-identical output, and the finest
   pyramid level is bit-identical to the raw input again (the function's own
   documented invariant).

## Where to look if this happens again

Any `.reshape(`/`.view(` in the data pipeline (`IO/`) or a model's loss/pooling
code that merges or reorders more than one axis at a time is worth a second look
— grep for `\.reshape\(|\.view\(` and check each hit against the rule above.
`.permute(`/`.transpose(` calls are never suspect on their own (they always do a
real reorder) — the risk is specifically `reshape`/`view` skipping the permute.
