# Migrate snapshot panels off BaseEpochChecker/MeSAEChecker — Sub-project B design

**Goal:** Retire `BaseEpochChecker`/`MeSAEChecker` (`model/base_checker.py`,
`model/MeSAE/plugin.py`'s `MeSAEChecker`) and replace their snapshot-rendering job with
three `tools/panels/panel_*.py` files (`panel_recon_signal`, `panel_stamp_by_patch`,
`panel_stamp_gallery`), backed by two new shared bundle-builders in
`tools/analysis/snapshot.py`. Migrate every caller — `analysis_pretrain.py`'s three
trial-selection modes, `analysis_finetune.py`'s per-class correct/wrong loop, and
`train_pretrain.py`'s in-training periodic snapshot — to build a bundle and call
`tools.panels.run_panels(...)` instead of `checker.check_pretrain`/`check_finetune`.
Delete the dead `has_attn_topo` code path and `_compute_spectra` along the way — this pass
is not a pure relocation, it removes real (unreachable) code as an explicit part of the
design, confirmed and approved with the user before writing this spec.

**Why:** Sub-project A built the panel framework and proved the contract with one
checkpoint-free, dataset-free panel (`profile`). This sub-project proves the framework
handles the harder, more common case: a panel that needs a real trial's model forward
pass, where that forward pass (a `SnapshotBundle`) must be computed once and shared across
several panels, not rebuilt per panel. `BaseEpochChecker`/`MeSAEChecker` existed to support
a second model that was never built (MeSAE is, and has been since MeFSQ's removal in ADR
0013, the only registered model) — the override machinery (`extract_psd`,
`compute_unit_colors`, `run_reconstruction`, `_render_topo_psd`) has exactly one
implementation each, forever, so collapsing them into concrete functions removes
indirection for zero present or near-future benefit, matching sub-project A's "replace it"
decision.

**Scope note (explicit, load-bearing):** this is sub-project B of a four-part
decomposition (A: framework + `panel_profile` — done; B: this; C: migrate the 8 codebook
panels off `BaseCodebookChecker`/`MeSAECodebookChecker`, separate, later; D: a
summary/aggregation panel, depends on B and C). B does not touch
`model/base_codebook_checker.py`, `model/MeSAE/plugin.py`'s `MeSAECodebookChecker`, or
`analysis_pretrain.py`'s `--analysis codebook` path — those are C's job, and stay exactly
as they are. `probes/` stays untouched, per the standing ruling for this whole effort.

## Current state (what exists today, relevant to B)

**Three live output panels** (confirmed empirically, not assumed — every other rendering
path `BaseEpochChecker` defines is dead for MeSAE, see below):
- `recon_signal` (`tools/viz/timeseries.py`'s `visualize_reconstruction`) — one file, a
  band-filtered orig-vs-recon grid for one trial, gated by `plot_recon`.
- `stamp_by_patch` (`tools/viz/panels.py`'s `plot_stamp_by_patch`, driven by
  `tools/viz/extract.py`'s `extract_flat_stamp_psd_by_patch`) — the real per-patch stamp
  grid.
- `stamp_gallery` (`tools/viz/panels.py`'s `plot_stamp_gallery`, driven by
  `tools/viz/extract.py`'s `extract_flat_stamp_gallery`) — whole-trial raw/recon view plus
  every stamp used somewhere in the trial.

Both `stamp_by_patch` and `stamp_gallery` are produced by ONE method today
(`MeSAEChecker._render_topo_psd`, `model/MeSAE/plugin.py:164-269`), gated by one flag
(`plot_topo_psd`) — they share the same trigger and largely the same per-trial setup, so
they become two panels that are always invoked together by every caller (not two
independently-toggleable options), not two panels split apart from logic that was written
to be separable.

**Confirmed 100% dead code for MeSAE** (every code path gated behind
`self.has_attn_topo`, and `MeSAEChecker.has_attn_topo = False` unconditionally, forever —
grep-verified: every caller of `plot_attn_topo`/`render_attn_topo`/`_render_stamp_panel`/
`plot_stamp_panel` is gated by this same flag, no exceptions):
- `BaseEpochChecker`'s default `_render_topo_psd` (`model/base_checker.py:201-220`) —
  MeSAE always overrides it, so even this "live for other models" default never runs
  today either.
- `_render_stamp_panel` (`model/base_checker.py:222-236`) and its only caller site
  (`_render_snapshot`'s `if plot_attn_topo:` block, lines 296-317).
- `render_attn_topo` (`plot_attn_topo` from `tools/viz/panels.py`, aliased on import)
  and `tools/viz/panels.py`'s `plot_stamp_panel` function itself — no other caller.
- `_compute_spectra` (`model/base_checker.py:147-199`) — shared FFT/band-crop helper
  written for the base default `_render_topo_psd` and `_render_stamp_panel`, BOTH dead;
  `MeSAEChecker`'s live `_render_topo_psd` override does its own inline FFT instead (never
  calls `self._compute_spectra`). Dead by the same transitive argument.
- `MeSAEChecker.extract_spectra` (`model/MeSAE/plugin.py:153-159`) — its own docstring
  already says "Unreachable while has_attn_topo=False."

**Three real callers of the class hierarchy today:**
1. `analysis_pretrain.py`'s `__main__` — three trial-selection modes under
   `if analysis in ('snapshot', 'both')`: a `visualize_params.pretrain.targets`-driven loop
   (lines 201-264), and a default per-subject/per-trial loop (lines 266-287). Both call the
   module-level `run(...)` (lines 77-88), which does `plugin.checker_cls()` +
   `checker.check_pretrain(...)`.
2. `analysis_finetune.py`'s `__main__` — one per-class correct/wrong loop (lines 130-180),
   calling its own `run(...)` (lines 76-87), `checker.check_finetune(...)`.
3. `train_pretrain.py:270-393` — builds `checker = entry.checker_cls()` once, then calls
   `checker.check_pretrain(...)` directly inside the training loop, every `viz_every_n`
   epochs, wrapped in `try/except` so a rendering failure never kills training.

**`checker_cls` repo-wide usage** (grep-verified, exhaustive): `train_pretrain.py:273`,
`analysis_finetune.py:80`, `analysis_pretrain.py:81` (all three call sites above),
`model/MeSAE/plugin.py:883` (the `BasePlugin(...)` instantiation at module bottom),
`model/base_plugin.py:23` (the dataclass field itself). `codebook_checker_cls` is a
separate field on the same dataclass — stays, C's job.

## Design

### `SnapshotBundle` and bundle-builders move to `tools/analysis/snapshot.py`

`SnapshotBundle` (the dataclass, `model/base_checker.py:27-61`) moves verbatim — it was
already stage-agnostic by design ("Stage-normalised input... `_render_snapshot` knows
nothing about pretrain vs. finetune"), just relocated.

Two new functions replace `check_pretrain`/`check_finetune`'s bundle-building halves
(everything before today's call to `self._render_snapshot`), each returning
`(bundle: SnapshotBundle, metrics: dict)`:

- `build_pretrain_bundle(model, dataset, trial_idx, config, device, epoch=None)` — today's
  `BaseEpochChecker.check_pretrain` body (`model/base_checker.py:321-376`), with
  `MeSAEChecker.compute_unit_colors` (`model/MeSAE/plugin.py:139-148`) and
  `_run_reconstruction_sae` (`model/MeSAE/plugin.py:30-54`) inlined directly — no more
  override indirection, one concrete implementation. `metrics` = `{'recon_mse': ...,
  **MeSAETrainer.epoch_metrics(model, out)}` (today's `_epoch_metrics` helper, inlined —
  `trainer.epoch_metrics` is `MeSAETrainer`'s own method, model/MeSAE/plugin.py:120-129,
  unaffected by this migration, still called the same way).
- `build_finetune_bundle(model, dataset, trial_idx, config, device, tag='')` — today's
  `BaseEpochChecker.check_finetune` body (`model/base_checker.py:388-440`), same
  inlining. `_patchify` (`model/base_checker.py:380-386`) moves in as a module-level
  helper both bundle-builders can call (finetune uses it directly; pretrain's dataset
  already yields pre-patched tensors, doesn't need it).

`_lookup_event_onset`/`_lookup_valid_range` (`model/base_checker.py:84-123`) move in as
module-level functions (they were `@staticmethod`s with no real class dependency).

### Three panels

- `tools/panels/panel_recon_signal.py` — `STAGES={'pretrain','finetune'}`,
  `NEEDS_CHECKPOINT=True`, `NEEDS_DATASET=True`. `run(ctx)`: reads `ctx.bundle`, calls
  `visualize_reconstruction` with the same arguments `_render_snapshot`'s `if plot_recon:`
  block builds today (`model/base_checker.py:264-275`).
- `tools/panels/panel_stamp_by_patch.py` — same `STAGES`/needs flags. `run(ctx)`: the
  `extract_flat_stamp_psd_by_patch` + `plot_stamp_by_patch` half of
  `MeSAEChecker._render_topo_psd` (`model/MeSAE/plugin.py:185-250`), reading from
  `ctx.bundle`.
- `tools/panels/panel_stamp_gallery.py` — same. `run(ctx)`: the
  `extract_flat_stamp_gallery` + `plot_stamp_gallery` half
  (`model/MeSAE/plugin.py:252-269`).

Per the user's decision: no separate `NEEDS_BUNDLE` flag. `NEEDS_DATASET=True` is the only
signal these panels need beyond `panel_profile`'s baseline — callers that select ANY
`NEEDS_DATASET=True` panel are responsible for resolving a trial and building a bundle onto
`ctx.bundle` before calling `run_panels`, the same way callers already had to resolve a
dataset today. This is a caller-side contract, not something `tools.panels`'
`any_needs`/`run_panels` need to enforce mechanically — B's task-level plan documents it
clearly and each caller follows the same pattern, checked at task-review time.

### `PanelContext` gains one optional field

`tools/panels/__init__.py`'s `PanelContext` dataclass gains `bundle: Optional[object] =
None` (kept untyped like `model`/`dataset` are today, to avoid importing `SnapshotBundle`
into the lightweight `tools/panels` package just for a type hint). Nothing else about the
dataclass or `build_panel_context`/`run_panels`/`any_needs` changes.

### Caller migrations

Each of the three real callers keeps its own trial-selection/looping logic (which varies
a lot: `analysis_pretrain.py` has three different selection strategies,
`analysis_finetune.py` searches for one correct+one wrong example per class,
`train_pretrain.py` iterates a fixed `viz_targets` list every N epochs) — that logic stays
exactly where it is, outside the panel system, since it's genuinely different per caller
and forcing it into a shared abstraction would fight the panels' own "terminal action on
already-prepared data" simplicity. What changes in each caller is only the *last step*:
where it used to call `run(config, output_dir, model, dataset, trial_idx, ...)` →
`checker.check_pretrain(...)`/`check_finetune(...)`, it now calls
`build_pretrain_bundle(...)`/`build_finetune_bundle(...)` to get `(bundle, metrics)`,
attaches `bundle` to a `PanelContext`, and calls
`tools.panels.run_panels(['recon_signal', 'stamp_by_patch', 'stamp_gallery'], stage, ctx)`.

`train_pretrain.py`'s migration is in-process, not a CLI/subprocess call — it already
holds a live `model`/`config`/`device` inside the training loop; it builds one
`PanelContext` per `viz_targets` iteration the same way the CLI scripts do, just without
going through `argparse` at all (no `--panel` flag involved — this is a direct Python call
to `tools.panels.run_panels`, always the same three panels, matching today's unconditional
`plot_recon=True, plot_topo_psd=True` defaults it already relies on).

### Retirements

- `model/base_checker.py`: deleted entirely (`BaseEpochChecker`, `SnapshotBundle` moved
  out, `_render_snapshot`/`_render_topo_psd`/`_render_stamp_panel`/`_compute_spectra`/
  `check_pretrain`/`check_finetune`/`_patchify` all either moved or deleted as dead).
- `model/MeSAE/plugin.py`: `MeSAEChecker` class deleted; `_run_reconstruction_sae`,
  `build_model`, `MeSAETrainer`, `MeSAECodebookChecker`, `MeSAEPlotter` all stay (only
  `MeSAEChecker` and its module-bottom `checker_cls=MeSAEChecker` reference go).
- `tools/viz/panels.py`: `plot_attn_topo`, `plot_stamp_panel` deleted (dead, no callers
  left). `plot_topo_psd_filter` (the base default's own panel function, also only called
  from the now-deleted dead base `_render_topo_psd`) — deleted too, same argument.
  `plot_stamp_by_patch`, `plot_stamp_gallery`, `plot_event_stamp_dynamics` (used by C, not
  B) stay.
- `model/base_plugin.py`: `checker_cls` field removed from `BasePlugin`; the now-unused
  `from model.base_checker import BaseEpochChecker` import removed.
- `model/MeSAE/plugin.py`'s module-bottom `BasePlugin(...)` call: `checker_cls=MeSAEChecker`
  line removed.
- `analysis_pretrain.py`/`analysis_finetune.py`: their module-level `run(...)` functions
  (and the now-orphaned `from model.factory import MODEL_REGISTRY` import, if nothing
  else in the file needs it — `analysis_pretrain.py`'s codebook branch still does, check
  per-file) deleted; the three trial-selection loops call the new bundle-builder +
  `run_panels` sequence instead.
- `--plot_attn_topo`-related plumbing: `analysis_pretrain.py`'s `run(...)` signature's
  `plot_attn_topo` parameter and the two call sites passing
  `check_cfg.get('plot_attn_topo', True)` go away with `run(...)` itself; `config/analysis.json`'s
  `check.plot_attn_topo` key becomes unread — left in the JSON file (not this sub-project's
  job to edit run configs), but `CLAUDE.md`'s Commands section note about
  `analysis_pretrain.py --analysis codebook` needing no change; check `CLAUDE.md` for any
  prose mentioning `plot_attn_topo` and update if found.

## Testing

No test suite exists (CLAUDE.md) — validation is smoke runs against real checkpoints,
matching every prior sub-project on this effort:
- Import sweep after the moves/deletions:
  `python -c "import tools.analysis.snapshot, tools.panels.panel_recon_signal, tools.panels.panel_stamp_by_patch, tools.panels.panel_stamp_gallery, analysis_pretrain, analysis_finetune, train_pretrain, model.factory, model.MeSAE.plugin"`.
- `git grep -n "BaseEpochChecker\|checker_cls\|has_attn_topo\|plot_attn_topo\|_render_stamp_panel\|plot_stamp_panel" -- '*.py'` — must return nothing live (only `codebook_checker_cls` survives, a different, unrelated field).
- Real-checkpoint comparison: capture `analysis_pretrain.py`'s current snapshot output
  (file listing + printed `recon_mse` etc.) BEFORE the migration starts, on the same
  checkpoint/subject/trial the earlier sub-projects' smoke tests already used
  (`output/pretrain/mesae_v10_all_share/checkpoint/last.pth`, `BNCI2014001`, subject 1,
  trial 0) — the render calls move verbatim, so filenames don't change:
  `sub{subject_id}_trial{trial_idx}{epoch_tag}_stamp_by_patch.png` and
  `..._stamp_gallery.png` (from `model/MeSAE/plugin.py:242,260` today), plus whatever
  `visualize_reconstruction` names its own output file — all three must appear identically
  before and after, with the same `recon_mse` printed (deterministic given `model.eval()`,
  no dropout).
- Same real-checkpoint comparison for `analysis_finetune.py`'s per-class loop (the
  checkpoint/config used by earlier sub-projects: `output/baseline/BNCI2014001_intra_raw_band/finetune/run_1_fold0/head.pth`).
- A short real training smoke run (few epochs, small subject list, matching the pattern
  earlier plans on this branch used for GPU smoke tests — see
  `docs/superpowers/plans/2026-09-21-phase2-infrastructure.md`'s Task 1 Step 3 for the
  convention) confirming `train_pretrain.py`'s periodic snapshot still fires at
  `viz_every_n` and produces the same three files, without crashing the training loop.

## Out of scope (explicit)

- `model/base_codebook_checker.py`, `model/MeSAE/plugin.py`'s `MeSAECodebookChecker`,
  `analysis_pretrain.py`'s `--analysis codebook` path — sub-project C.
- The summary/aggregation panel — sub-project D, depends on B and C both existing.
- `probes/` — untouched, standing ruling for this whole effort.
- Redesigning trial-selection logic in any of the three callers — moved as-is, only the
  final dispatch step changes.
- `extract_psd`'s exact role in gating whether `stamp_by_patch`/`stamp_gallery` attempt
  to render at all (today: a try/except boundary around `self.extract_psd(...)` in
  `_render_snapshot`) — preserved in spirit (a failure to extract should still skip both
  panels rather than crash the whole run, matching current behavior), but the precise
  shape of that guard in the new bundle-builder is a task-level implementation detail, not
  pinned down further here.
