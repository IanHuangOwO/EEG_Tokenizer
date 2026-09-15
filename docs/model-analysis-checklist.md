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
- [x] **Recon-energy share: routed vs shared.** `_render_pool_energy_share`
  (`plugin.py`) → `pool_energy_share.png` (`viz/codebook.py`). Measured on
  `mesae_tokenizer_v4`: Shared pool (4 stamps) carries **68% of total recon
  energy on average** (range 48-85% across 8 datasets) despite Routed having
  15x more stamps (60).
- [x] **Cross-dataset stability of each pool's usage.** Folded into the same
  panel (`cv_routed`/`cv_shared` in `_render_pool_energy_share`). Measured:
  Shared's usage CV across datasets = 0.277, Routed's = 1.135 — Shared really
  is the stable/generic one, Routed genuinely varies per dataset, matching
  what the names imply.
- [x] **Causal ablation: zero one pool, measure recon MSE delta.**
  `_render_pool_ablation` (`plugin.py`) → `pool_ablation.png`. No StampBank
  change needed — zeroes `amp` for one pool's columns and re-decodes via the
  already-public `decode_selected`. Measured: ablating Shared costs
  **+844% to +5789%** MSE across all 8 datasets tested (incl. `BCICIV2a`,
  never in the training set) — consistently worse than ablating Routed
  (+679% to +2469%). Shared is the load-bearing pool, not a baseline on top
  of Routed's work — the opposite of what the energy share alone might
  suggest about "which pool does the interesting work". Open question this
  raises: is Routed's 60-stamp specialization budget underused relative to
  `docs/adr/0010`'s intent for it?
- [x] **Task-label information: which pool actually predicts it (with a raw
  baseline).** The panels above measure reconstruction mass/necessity, not
  task relevance — a pool can dominate recon while carrying no label
  information, or the reverse. `_render_pool_label_probe` (`plugin.py`) →
  `pool_label_probe.png` (`viz/codebook.py`): 5-fold CV logistic regression
  on trial-level usage, routed-only vs shared-only vs both vs a RAW-signal
  baseline (per-channel power of the stitched trial, no learned structure),
  against the per-trial task label (datasets with <2 classes or <5
  trials/class skipped). The raw column is load-bearing, not decoration —
  without it a chance-level stamp probe can't be told apart from "task has
  no decodable signal in anything" vs "tokenizer specifically lost it", and
  a high stamp probe can't be told apart from "tokenizer learned something"
  vs "trivially decodable from amplitude alone".

  Measured on `mesae_tokenizer_v4` (5/8 sampled datasets qualified),
  routed/shared/both/raw:
  - `Dial` 14.0/12.0/17.0/**8.5** (chance 8.3) — raw is at chance, every
    stamp probe clears it: real value-add from the tokenizer here.
  - `BCICIV1_Train` 50.0/46.0/49.5/**46.0** (chance 50.0) — raw is ALSO at
    chance: this task has no decodable signal in anything, not a tokenizer
    failure (my read before adding the raw column was right by accident).
  - `Inria_Train` 72.0/77.5/70.0/**74.5** (chance 50.0) — raw alone is
    already strong; Routed sits BELOW raw. Most of this task's signal is
    just loudness, the tokenizer isn't adding much on top for Routed.
  - `EEGMMIdb` 37.5/44.0/38.0/**38.0** (chance 33.3) — only Shared clears
    raw; Routed and both are statistically indistinguishable from raw.
  - `BCICIV2a` 28.0/28.0/28.5/**32.0** (chance 25.0) — raw beats every
    stamp probe on this held-out (never-trained-on) dataset — a real
    regression the earlier pass (no baseline) couldn't see at all.

  `both` combined still doesn't reliably beat either pool alone — likely
  mild overfitting from doubling feature count against a small per-fold
  trial count, not yet a real "combining hurts" finding; worth revisiting
  with more trials/regularization.

## 2. Differences between stamps — every current axis

- [x] **Raw waveform template** (`D_i`, content-free, no `z` dependence) —
  pairwise cosine similarity, `decoder_fingerprint_matrix` (`plugin.py`), now
  actually wired into `_render_fingerprint_similarity` →
  `stamp_fingerprint_similarity.png` (`viz/codebook.py`) — it existed before
  this pass but was never called from anywhere, silently dead. Measured on
  `mesae_tokenizer_v4`: off-diagonal cosine mean 0.002 (good, mostly
  orthogonal) but max 0.890 — at least one near-duplicate pair still slipped
  past `mp_loss` (`docs/adr/0011`), worth checking the heatmap for which pair.
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
- [x] **Phase distribution per stamp.** Extended `_render_identity_consistency`'s
  existing per-firing loop (`plugin.py`) to also collect each firing's
  channel-summed phase → `stamp_phase_consistency.png` (circular variance per
  stamp, `viz/codebook.py`). Measured: mean circular variance 0.920, **0 of
  30** units with ≥5 firings were phase-locked (<0.3) — no atom shows a
  consistent firing phase, consistent with `docs/adr/0010`'s decision to
  withdraw oscillator atoms (nothing here behaves like one).
- [x] **Full pairwise topography distance matrix.** Same extended loop also
  collects the raw signed `(a,b)` per firing (not just magnitude) →
  `stamp_topography_distance.png`, one `[n,n]` cosine-distance heatmap per
  dataset (channel spaces differ across datasets, so no single pooled
  matrix), reusing the same coherent-average + reference-phase-projection
  convention `viz/extract.py`'s `_used_flat_stamps` uses for one trial.
- [ ] **Explicit pairwise co-firing/exclusivity matrix.** `plot_stamp_similarity`'s
  Jaccard is trial/patch-level aggregate; a raw `[n_stamps, n_stamps]`
  co-occurrence count within the same patch-position's top-k slate (which
  pairs are mutually exclusive vs always picked together) isn't rendered as
  its own matrix — would need `dense_routed`'s per-patch binary mask, easy to
  derive from what `check_codebook` already collects.
- [x] **Energy/loudness distribution per stamp — "load-bearing vs marginal"**
  and **rank position in mp_loss's residual ordering** — both answered by one
  new panel, `_render_stamp_energy_and_rank` (`plugin.py`) →
  `stamp_energy_rank.png`. Rank is recovered from `usage` alone (no
  `StampBank.forward` change needed: each patch's routed row has exactly
  `stamp_top_k` nonzero entries, ranking them descending reproduces
  `mp_loss`'s own by-h order). Measured: mean h routed=0.186, shared=2.371
  (~13x) — consistent with the pool-energy-share finding above; 33/60 routed
  stamps fired at least once in this sample.
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
