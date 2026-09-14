# Model analysis checklist

Running list of questions we want answered about a trained model, what
`check_model.py`/`viz/*` already answers today, and what's still a gap. Scope:
MeSAE (`unit_label='Stamp'`) first, since that's the active model — MeFSQ
questions can reuse the same shape once relevant.

Status legend: `[x]` = a panel/metric already answers this. `[ ]` = gap, not
computed anywhere yet. Each item names the file:function that either answers
it or is where it should live.

## 1. Where does context live — Routed vs Shared pool?

Direct question: how much of the reconstruction actually comes from the
always-on Shared pool (generic/context content) vs the competing Routed pool
(specialized content)? See `CONTEXT.md`'s Routed/Shared pool definitions.

- [x] Per-stamp firing rate / dead-atom rate — `fire_ema`, `dead_feature_rate`
  (`MeSAE_modules.py` `StampBank.forward`, logged every epoch). Routed-pool
  only (Shared is unconditional by construction, always "alive").
- [x] Per-stamp usage strength split by pool position — `extract_usage`
  (`plugin.py:339`) returns `[N, n_stamps]` with routed at `[:n_routed]`,
  shared at `[n_routed:]`; `plot_stamp_by_patch`/`plot_stamp_gallery` color
  shared stamps differently (`shared_color`, `n_routed` param) in every panel
  that shows individual stamps.
- [x] Router health (entropy/load-std/gate-entropy) — `stamp_router_entropy`
  etc., logged every epoch. Routed-pool only (no router over Shared, it's
  unconditional).
- [ ] **Recon-energy share: routed vs shared.** Nothing currently sums
  `h^2` (or `amp^2`) split at the `n_routed` boundary into one "% of total
  reconstructed energy explained by Shared" number, aggregated over a
  corpus/dataset. This is the most direct answer to the question and doesn't
  exist yet. Add to `MeSAECodebookChecker` (`plugin.py`): a
  `_render_pool_energy_share` method reading `extract_usage`'s already-cheap
  `[N, n_stamps]` (or a fresh `amp`-based pass if signed energy matters), sum
  `h[:n_routed]^2` vs `h[n_routed:]^2` per trial, plot as a bar/violin per
  dataset (does the split vary by dataset — a "conditional content" signal —
  or stay flat — a "shared IS the generic baseline" signal, per
  `docs/adr/0010`'s reasoning about why conditional content can't live in an
  unconditional pool).
- [ ] **Cross-dataset stability of each pool's usage.** If Shared truly
  carries generic content, its per-stamp mean-`h` should vary LESS across
  datasets than Routed's does (Routed is where dataset-specific content is
  allowed to live). Coefficient-of-variation of per-stamp mean usage across
  `usage_by_dataset` (already computed in `check_codebook`), split at
  `n_routed`, one number each. Cheap — reuses `filter_usage_and_activity.png`'s
  underlying data (`plugin.py`'s `_render_...` doesn't currently emit this
  number, `viz/codebook.py plot_usage_and_activity` would need an
  aggregate-summary variant or a small new panel).
- [ ] **Causal ablation: zero one pool, measure recon MSE delta.** The
  above are all correlational (usage strength ≠ proof of necessity). A
  `check_model.py --ablate-pool routed|shared` mode (or a
  `MeSAECodebookChecker` method) that re-runs `model.stamps.forward` with one
  pool's `amp` zeroed and reports the MSE increase per dataset would be the
  strongest test of "does context genuinely live here" — a pool whose removal
  barely moves MSE isn't load-bearing regardless of how often it fires.
  Needs a small `StampBank.forward` hook or a monkeypatch at the checker
  level; not yet built.

## 2. Differences between stamps — every current axis

- [x] **Raw waveform template** (`D_i`, content-free, no `z` dependence) —
  pairwise cosine similarity, `decoder_fingerprint_matrix`
  (`plugin.py:356`) → the direct successor to the old `filter_relation.png`,
  verifies `mp_loss` (`docs/adr/0011`) actually produced distinct atoms.
- [x] **Frequency content** — PSD per stamp, `extract_flat_stamp_psd`/
  `extract_flat_stamp_psd_by_patch` (`viz/extract.py`) feeding
  `plot_stamp_gallery`/`plot_stamp_by_patch`'s PSD cells.
- [x] **Topography (mixing column) consistency** — does one stamp id mean
  one scalp pattern across occurrences? `_render_identity_consistency`
  (`plugin.py:424`) via `plot_stamp_identity_consistency`
  (`viz/codebook.py:984`): within-id vs between-id cosine on centered mixing
  columns. Currently a within/between-id AGGREGATE only (one within-stat and
  one between-baseline per stamp), not a full pairwise matrix.
- [x] **Usage/co-occurrence pattern** — binary + weighted Jaccard on
  `dense_routed` usage, `plot_stamp_similarity` (`viz/codebook.py:597`) via
  `_render_patch_similarity` (`plugin.py:405`). Behavioral distance, not
  content.
- [x] **Semantic category (ICLabel)** — `extract_flat_stamp_gallery`'s
  ICLabel classification (`viz/extract.py:478`, `viz/iclabel.py`) — brain/eye/
  muscle/etc per stamp, shown in `plot_stamp_gallery`'s ICLabel row and used
  as the `label_agree` cross-check in `stamp_identity_consistency`.
- [x] **Temporal/event-locked firing pattern** — `event_stamp_dynamics_*.png`
  (`_render_event_stamp_dynamics`, `plugin.py:527`, `plot_event_stamp_dynamics`
  `viz/panels.py:738`) — selection-rate trajectory per stamp around an event
  onset; distinguishes stamps that react to an event from ones that don't.
- [x] **Per-dataset specialization** — `filter_usage_and_activity.png`
  (`plot_usage_and_activity`) and `dataset_relation.png`
  (`plot_dataset_relation`) — which datasets fire a given stamp, and how
  datasets relate to each other through shared stamp usage.
- [x] **Collective redundancy (global, not pairwise)** — effective rank of
  the whole usage matrix, `plot_unit_freedom` (`viz/codebook.py:847`),
  capped by `rank_ceiling` (`plugin.py:367`, `min(top_k, embed_dim)`).
- [x] **Real decoded content, cross-trial consistency** — `extract_stamp_content`
  (`plugin.py:371`) → `plot_patch_position_consistency`
  (`viz/codebook.py:729`) via `_render_patch_position_consistency`
  (`plugin.py:502`). Currently used for same-stamp-across-trials consistency
  only, not stamp-vs-stamp content distance (that's `decoder_fingerprint_matrix`'s
  job on the cheaper raw template instead).
- [ ] **Phase distribution per stamp.** Every firing carries a phase
  `atan2(b,a)` (quadrature gain pair, see `CONTEXT.md`'s Stamp definition) —
  currently only consumed by `recon`, never read as a diagnostic. Does a
  stamp fire at a consistent phase (phase-locked — plausible for a
  genuine oscillatory source like line noise) or an effectively random one
  (broadband/transient content, where phase carries no information)? A
  circular-variance histogram per stamp, computed alongside
  `_render_identity_consistency`'s existing per-firing loop (`plugin.py:452`
  already computes `mag`; phase is `atan2(o.amp[...,1], o.amp[...,0])`, free
  to add there) is the missing panel.
- [ ] **Full pairwise topography distance matrix.** `stamp_identity_consistency`
  gives within/between AGGREGATES; a `[n_stamps, n_stamps]` cosine matrix on
  each stamp's mean mixing column (parallel to `decoder_fingerprint_matrix`,
  but on the topography instead of the raw waveform) would show whether
  waveform-distinct stamps still project to similar scalp patterns, or
  vice versa — currently nothing renders this pairing directly.
- [ ] **Explicit pairwise co-firing/exclusivity matrix.** `plot_stamp_similarity`'s
  Jaccard is trial/patch-level aggregate; a raw `[n_stamps, n_stamps]`
  co-occurrence count within the same patch-position's top-k slate (which
  pairs are mutually exclusive vs always picked together) isn't rendered as
  its own matrix — would need `dense_routed`'s per-patch binary mask, easy to
  derive from what `check_codebook` already collects.
- [ ] **Energy/loudness distribution per stamp — "load-bearing vs marginal".**
  `k_eff` is an aggregate parsimony diagnostic; no panel currently shows the
  distribution of `h` per stamp (is a given stamp usually the dominant
  contributor when selected, or always a minor cleanup pick?). Cheap
  histogram off `extract_usage`'s existing `h`.
- [ ] **Rank position in mp_loss's residual ordering.** `mp_loss`
  (`docs/adr/0011`, `CONTEXT.md`'s Residual ordering entry) ranks slots by
  amplitude every forward pass; nothing logs which stamps consistently land
  at rank 0 (dominant, graded against the full patch) vs late rank (cleanup,
  graded against whatever's left). Would need a small addition to
  `StampBank.forward`'s existing `order`/`order_routed` computation to expose
  per-stamp mean rank as a diagnostic, not just consume it for the loss.
- [ ] **Fire-rate trajectory over training (not just a checkpoint snapshot).**
  `fire_ema` is a single scalar per stamp at whatever epoch the checkpoint was
  saved — no panel shows a stamp's alive/dead HISTORY across epochs (does the
  same atom flip dead→rescued→dead repeatedly, or converge once and stay?).
  Needs per-epoch `fire_ema` logging (a training-time addition, not a
  post-hoc `check_model.py` one) — out of scope for `check_model.py` itself,
  note here so it doesn't get lost.

## Adding a new question

1. State the question in one line, add it to whichever section above (or a
   new one).
2. Check whether an existing `check_codebook`/`check_pretrain` pass already
   computes the raw numbers (`trial_records`'s `usage`, or a `needs_raw_tensors`
   fresh-forward-pass record) before writing a new extraction path — most
   panels above share the same two hooks (`extract_usage`,
   `extract_stamp_content`).
3. Mark `[x]`/`[ ]` here once it's answered, with the file:function that
   answers it, so the next person doesn't re-derive the same panel.
