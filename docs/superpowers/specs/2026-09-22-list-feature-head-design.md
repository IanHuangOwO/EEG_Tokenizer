# Design note: feature as a list (combine primaries, not just branches)

Date: 2026-09-22. Status: proposal, not scheduled. Not part of the finetune restructure
sub-projects A-D; a candidate follow-up once the restart baseline and D are done.

## Problem

Today's head config has one `feature` (mutually exclusive: `stamp_power`, `stamp_band`,
`raw_band`, `raw_signal`) plus two optional branches (`phase_advance`, `evoked_rank`) that
only attach to `stamp_power`. The branches are already "pick zero or more"; the primary
feature is not. That blocks a config like "stamp power together with raw band power in one
head", which is exactly the "stronger raw control" ADR 0016 flags as an open item, and blocks
combining stamp features across granularities (`stamp_power` + `stamp_band`) as an ablation.

## Proposal

Replace `feature: "<name>"` with `features: ["<name>", ...]`, a list of one or more entries
from `{"stamp_power", "stamp_band", "raw_band", "raw_signal", "phase_advance", "evoked"}`.
Each entry computes its own `[B, K, F_i]` block (or `[B, N', K, F_i]` when its own
`time_pool` is `none`) via the pipeline it already has; all blocks are concatenated before the
readout. `phase_advance` and `evoked` stay branch-only (they need `a, b` from a `stamp_*`
entry already in the list; at least one of `stamp_power`/`stamp_band` must be present for them
to be legal). A duplicate primary is redundant, not useful, and is rejected.

### Per-entry time pooling

The blocking issue: `raw_signal` only makes sense with `time_pool: "none"` (its own patch
axis becomes the feature axis) while the others use `flat`/`learned`/`window`. One global
`time_pool` cannot serve both in the same head. Resolution: `time_pool` (and `time_rank`,
`window`) move from a single top-level key to a per-entry override, with the top-level value
as the default for entries that do not override it:

```json
{"features": ["stamp_power", "raw_band"],
 "time_pool": "learned", "time_rank": 2,
 "overrides": {"raw_band": {"time_pool": "flat"}}}
```

`spatial_k` stays a single shared value (the same channel mixing makes sense for every
feature that reads the real channels).

### Validation

Each entry keeps its existing rules (`raw_signal` needs `time_pool: none`; `evoked`/
`phase_advance` need a `stamp_*` entry present and are not usable with `time_pool: window` for
`evoked`; `num_patches` required whenever any entry needs it). The resolver iterates the list
instead of a single value; `feature_dim` sums each entry's width.

### Backward compatibility

None required for old runs (standing decision). The new `features` list key replaces
`feature`; existing baseline configs move from `feature: "stamp_power"` to
`features: ["stamp_power"]` (or a small migration in the config loader that wraps a bare
string into a one-element list, if that is preferred at implementation time — a decision for
the implementer, not fixed here).

## Example configs this enables

- **Stronger raw control:** `features: ["raw_band", "raw_signal"]` — band power and
  phase-locked samples in one head, answering ADR 0016's open item.
- **Granularity ablation in one head:** `features: ["stamp_power", "stamp_band"]`.
- **Everything the tokenizer can offer:** `features: ["stamp_power", "phase_advance",
  "evoked"]` — today's C1 plus both branches together (currently impossible: branches only
  attach one at a time in practice since nothing stops setting both, but nothing tests a raw
  feature alongside them either).

## Cost / scope estimate

- `MeSAE_modules.py`: `resolve_head_config` (iterate the list, per-entry override resolution,
  validation), `feature_dim` (sum over entries), `FeatureHead.__init__`/`forward` (build one
  sub-module per entry, concatenate outputs). Existing per-feature math (`spatial_mix`,
  `FlatTimePool`, `LearnedTimePool`, `phase_advance`, `EvokedBranch`) is reused unchanged.
- `train_finetune.py`: `make_source`/`StampSource`/`RawSource` currently branch on a single
  `feature` string to decide whether the run needs the stamp cache, the raw signal, or both.
  A list may need both at once (e.g. `stamp_power` + `raw_band`), so the source layer needs a
  small extension (serve both cached stamp amplitudes and the raw signal for the same batch of
  trial indices) rather than picking one.
- Equivalence checking: every existing single-feature config is a one-element list, so the
  current equivalence scripts still apply unchanged as a regression check; new checks are
  needed for genuine multi-entry combinations (shape, that dropping the mixing/pooling of one
  entry does not disturb another).
- Estimated size: comparable to sub-project A's Task 1 (one focused implementation task), not
  a new sub-project on its own.

## Open question

Whether `experiments/` (sub-project D) specs should expose `features` sweeps as a first-class
sweep dimension (e.g. `"sweep": {"features": [["stamp_power"], ["stamp_power", "raw_band"]]}`)
or only as named `variants`. Left to D's design.
