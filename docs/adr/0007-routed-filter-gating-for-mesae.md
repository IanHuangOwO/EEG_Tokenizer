# Optional routed Filter gating for MeSAE (revertible)

> **Update:** after training/validating the opt-in path below, the decision was made to
> keep routing as MeSAE's only path — `use_router` and the dense (no-router) code path are
> removed, routing is no longer optional, and `n_filters` is now derived as
> `n_routed_filters + n_shared_filters` rather than an independently configurable value.
> The revert-path described below (flip a flag) no longer exists; reverting now means going
> back to a pre-this-ADR commit. The rest of this document (root cause, router design,
> health metrics) still accurately describes the mechanism, just read every "opt-in"/
> "off by default" as historical context for how it got adopted, not current behavior.
> A later follow-up gave `TopKSAE.forward` an optional `k_groups` override so routed and
> shared Filters could run different sparsity levels (`sae_k_routed`/`sae_k_shared`)
> through the *same* dictionary — and was then reverted after it caused repeated
> `dead_feature_rate` mass-collapse events. Root cause: `fire_ema` (the dead-feature
> tracker) averaged firing rate as one flat mean over every row regardless of group, but
> the routed group's row count (`M*n_routed_filters`) dwarfed the shared group's
> (`M*n_shared_filters`, e.g. 60:4) — so dictionary atoms that specialized for the small
> shared pool were structurally outvoted into looking dead by the large routed pool's
> traffic, not from genuine disuse. `k_groups` is fully removed from `TopKSAE`; `sae_k` is
> back to a single value shared by every Filter (config key renamed from
> `sae_k_routed`/`sae_k_shared` to `sae_k`). "`TopKSAE` is untouched" below is accurate
> again — the router still only gates decoded output, never the SAE's own selection.

## Context

Cross-dataset codebook diagnostics (`viz/codebook.py`, `model/base_codebook_checker.py`)
surfaced a pattern: `filter_usage_histogram.png` shows real per-dataset differentiation in
*which dictionary atoms* fire, but `filter_relation.png`'s usage-frequency correlation
between Filters stays high (~0.75-1.0) regardless. Recon attention topomaps corroborate it
— Filters converge onto attention over small clusters of spatially-nearby channels rather
than onto distinct signal characters (e.g. an SSVEP-like posterior/occipital group vs. an
MI-like sensorimotor group vs. a P300-like midline group).

Root cause: `MeSAEPretrain` (`model/MeSAE/MeSAE.py`) has no competitive pressure between
Filters. `ExpertChannelPool` gives each Filter its own query, but:

1. `self.sae` is a **single `TopKSAE` instance reused across all `n_filters`** (see its
   docstring: "gives the sparse feature dictionary a single reusable vocabulary across
   filters instead of a per-filter-unique one"). Whenever two Filters' pooled views land
   in a similar region of embedding space, they draw on the *same* dictionary atoms —
   which is exactly what happens if pooling collapses toward channel-locality, since
   nearby-channel content is the easiest, lowest-reconstruction-error signal to converge
   on.
2. `forward()`'s decode step is dense and unconditional:
   `recon = recon_per_filter.sum(dim=1)` (MeSAE.py:240) — every Filter always contributes
   to every patch's reconstruction. Nothing in the loss rewards a Filter for specializing
   on a subset of patches/signal-types; the gradient is free to let every Filter drift
   toward whatever's easiest (channel locality), because none of them are ever excluded.

MeSAE was deliberately kept non-routed at the ADR level (`docs/adr/0003`, "that reasoning
doesn't transfer to a non-discrete, non-routed SAE") — so introducing routing is reversing
a stated design choice, not a small tweak. It should be optional and cheaply revertible,
not a rewrite of the existing dense path.

## Decision

Add an **opt-in** routed-gating path, off by default, alongside the existing dense path —
never replacing it:

1. **New class, `FilterRouter`, in `model/MeSAE/MeSAE_modules.py`** (alongside
   `ExpertChannelPool`/`MultiHeadDecoder`/`PerChannelHeadAttn` — same file, same
   "duplicated not cross-imported" convention `PerChannelHeadAttn`'s docstring already
   states). Logic ported unchanged from `model/MeFSQ/MeFSQ_modules.py`'s `Router`: scores
   each Filter's own pooled view (`[M, n_filters, D]`) against a learned per-Filter weight
   vector, top-k softmax-gates a **routed** subset, plus a Switch-Transformer-style
   load-balance loss. `n_filters` splits into `n_routed_filters` + `n_shared_filters`
   (routed/shared naming mirrors `docs/adr/0001`, but is a lighter construction than
   MeFSQ's: MeFSQ gives routed and shared pools fully separate `ExpertChannelPool`/VQ/
   decoder instances, MeSAE keeps the single existing `filter_pool`/`sae`/`decoder` and
   only gates their already-computed output — deliberately, to keep this a wrapper around
   the existing dense path rather than a second architecture living beside it). Shared
   Filters stay always-on at a fixed down-weighted baseline (0.2x, matching MeFSQ's
   constant), routed Filters get the top-k gate mask (zero weight if not selected for a
   given patch). `TopKSAE` (MeSAE.py) is untouched — the router gates *which Filters'
   decoded output survives into the sum*, not the SAE's own dictionary mechanics.

2. **`MeSAEPretrain.__init__`** gains one new optional arg, `use_router: bool = False`,
   plus router hyperparams (`n_routed_filters`, `n_shared_filters`, `top_k`) read
   from `model_params.MeSAE.pretrain` the same way `n_filters`/`sae_k`/etc. already are.
   `self.router` is only constructed when `use_router=True` — when `False` (every existing
   config/checkpoint, since the field won't exist there), `MeSAEPretrain` has exactly the
   same parameters and compute graph it has today. Nothing about the default path changes
   byte-for-byte.

3. **`forward()`** gets a small conditional block:
   ```python
   recon_per_filter = self.decoder(sae_out)             # [M, Q, C*patch_len]  (unchanged)
   if self.use_router:
       gate, lb_loss = self.router(pooled)               # [M, Q], scalar
       recon_per_filter = recon_per_filter * gate.unsqueeze(-1)
   recon = recon_per_filter.sum(dim=1).reshape(...)       # (unchanged either way)
   ```
   `lb_loss` is added to the total loss only when `use_router` is True (same pattern
   `docs/adr/0003` already uses for dropping `aux_loss` once the SAE freezes — a loss term
   only exists in the graph when the mechanism producing it is actually active).

4. **Revert path**: flip `use_router` back to `False` (or just don't set it — default is
   `False`) and `MeSAEPretrain` is the exact model that exists today, no code path
   different, no checkpoint incompatibility for anything trained without the flag. Because
   `self.router` is never constructed in the off state, there's no dead/unused submodule
   sitting in `state_dict()` either — off means genuinely absent, not just bypassed.

## Consequences

- `model/MeSAE/MeSAE_modules.py` gains `FilterRouter`; `TopKSAE`, `ExpertChannelPool`,
  `MultiHeadDecoder` are untouched.
- `model/MeSAE/MeSAE.py`: `MeSAEPretrain.__init__` and `.forward()` get the minimal
  conditional edits described above; `get_loss` gets one `if self.use_router:` branch to
  add `lb_loss`, mirroring the existing `aux_loss`-dropped-when-frozen branch.
- `config/config.json`: new optional keys under `model_params.MeSAE.pretrain`
  (`use_router`, `n_routed_filters`, `n_shared_filters`, `top_k`) — absent means
  today's dense behavior, so existing configs need no edits.
- `encode_post_sae()` (used by `MeSAEFinetune`) never sums across Filters at all — it
  returns each Filter's decoded output separately (`[B, C, N, Q, patch_len]`), and
  `MeSAEFinetune`'s `PerChannelHeadAttn` already learns its own per-Filter attention
  weighting downstream (its stage 2, "per channel, a learnable query attends over the H
  units"). So router gating doesn't need wiring into `encode_post_sae` for correctness —
  finetune already re-weights Filters on its own. Worth revisiting only as a diagnostic
  question later: does finetune's *learned* per-Filter weighting end up correlated with
  the router's gate, or ignore it entirely? Not a blocker for this ADR either way.
- Router load-balance loss reopens the standard MoE failure mode (router collapse: a
  couple of Filters absorb all top-k picks) — worth watching in
  `filter_usage_histogram.png`/`filter_relation.png` post-change, same panels that
  surfaced the original problem.
- This is a Tokenizer-stage-only change; the Masked stage already freezes
  `filter_pool`/`sae`/`decoder` wholesale (`freeze_sae`, ADR 0003) — freezing should extend
  to `self.router` too when present, same rationale (a frozen router's load-balance loss
  is as pointless as a frozen SAE's aux-k loss, ADR 0003 point 2).
