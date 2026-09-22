# /tools framework + first panel (panel_profile) — Sub-project A design

**Goal:** Relocate `analysis/` and `viz/` under a new `tools/` package, establish a
`panel_*.py` contract for self-contained analysis panels (calculation via `tools.analysis`,
rendering via `tools.viz`, orchestration in the panel file itself), wire
`analysis_pretrain.py` to discover and run panels additively, and prove the contract with
`panel_profile.py` — a panel that needs neither a checkpoint nor a dataset and produces a
printed table instead of a plot, replacing today's standalone `profile_model.py`.

**Why:** The existing snapshot/codebook analysis lives in a class hierarchy
(`BaseEpochChecker`/`BaseCodebookChecker` + MeSAE-specific overrides, ADR 0004) built for
multi-model support that MeSAE — the only registered model since MeFSQ's removal (ADR
0013) — no longer needs. The user wants calculation code consolidated under `tools/analysis/`,
rendering under `tools/viz/`, and each analysis output ("panel") as one self-contained
`panel_{name}.py` file, easy to add or remove without touching a class hierarchy. This is
sub-project A of a four-part decomposition (A: framework + proof of concept, B: migrate
snapshot panels, C: migrate codebook panels, D: summary/aggregation panel) — too large for
one spec, so this document covers only A. `probes/` is explicitly out of scope for this
whole effort (deleted wholesale once the effort finishes, nothing ported — separate ruling,
not part of A).

**Scope note (explicit, load-bearing):** A does not touch `model/base_checker.py`,
`model/base_codebook_checker.py`, or `model/MeSAE/plugin.py`. The existing
`--analysis snapshot/codebook/both` path in `analysis_pretrain.py` keeps working exactly as
it does today, unchanged, driven by the untouched class hierarchy. A only *adds* a new,
parallel `--panel` path. B and C (separate, later specs) migrate the existing panels one at
a time and retire the classes.

## Current state (what exists today, relevant to A)

- `analysis/__init__.py` (from the just-completed `analysis_pretrain.py`/`analysis_finetune.py`
  split): 8 orchestration helpers — `_deep_merge`, `load_config`, `resolve_output_dir`,
  `select_subject_dataset`, `filter_config_to_subject`, `load_model`, `pick_trial`,
  `setup_mne_info`. No calculation logic here (`load_config`/`setup_mne_info` have zero
  callers repo-wide, confirmed dead before the split too — parked as a known minor, not
  A's job to fix).
- `viz/` package: `extract.py` (707 lines, pure calculation — `extract_head_psd`,
  `extract_filter_psd`, etc., already misplaced under `viz/` despite computing, not
  plotting), `panels.py` (849 lines, pure plotting), `codebook.py` (1318 lines, pure
  plotting), `timeseries.py` (156 lines, mixed — `visualize_reconstruction` does light
  banding calculation plus plotting), `topomap.py` (91 lines, geometry helpers shared by
  the plotting modules), `iclabel.py` (85 lines, ICLabel classification, calculation).
  Moving `viz/extract.py`'s calculation into `tools/analysis/` is B/C's job (it's called
  only from the classes A doesn't touch) — A moves the `viz/` package as-is, no internal
  reshuffling of its contents.
- `profile_model.py` (176 lines, repo root): standalone script, no checkpoint, no
  dataset — builds a fresh untrained model from `config/config.json`, runs forward passes,
  times each top-level module via forward hooks, counts parameters, prints a text table.
  `ProfilerHooks` class + `count_parameters` function + the timing loop in `profile_model()`
  are the calculation; the `print(...)` calls at the end are the "render" step (a table,
  not a plot).
- `analysis_pretrain.py`'s `__main__` block: parses `--config`, `--base-config`,
  `--checkpoint`, `--mode` (removed already), `--analysis` (`snapshot`/`codebook`/`both`),
  `--subject`, `--trial`, `--dataset`, `--recon_cmap`. `--checkpoint` and `--config` are
  effectively required today — every code path needs a model and a base config.

## Design

### Directory layout

```
tools/
  analysis/
    __init__.py      # today's analysis/__init__.py content, verbatim (git mv)
    profile.py        # NEW: count_parameters, ProfilerHooks, run_profile(config, device,
                       #   train_mode) -> a plain dict/dataclass of the computed stats
                       #   (param counts per component, per-component ms, loss ms, total ms)
  viz/                 # today's viz/ package content, verbatim (git mv), no internal changes
  panels/
    __init__.py        # empty (package marker) — no registry, discovery is filename-based
    panel_profile.py   # NEW: ctx -> tools.analysis.profile.run_profile(...) -> print table
```

`viz/extract.py`'s calculation content and `viz/timeseries.py`'s banding calculation stay
where they are — B/C's job to relocate when they migrate the panels that use them. A moves
`viz/` as one unit.

### The panel contract

A panel is a module at `tools/panels/panel_<name>.py` exposing:
- `STAGES: frozenset[str]` — subset of `{'pretrain', 'finetune'}`, which
  `analysis_*.py` script(s) may run it. `panel_profile.py`: `{'pretrain', 'finetune'}` (it's
  useful from either entrypoint — needs neither a checkpoint's stage nor a dataset).
- `NEEDS_CHECKPOINT: bool` and `NEEDS_DATASET: bool` — whether the CLI must resolve a
  model/dataset before calling this panel. Both `False` for `panel_profile.py`.
- `def run(ctx) -> None` — does the work (calculate via `tools.analysis`, then either
  render via `tools.viz` or print directly) and writes/prints its own output. No return
  value contract beyond "didn't raise" — panels are terminal actions, not composable
  functions (unlike D's summary panel, which reads OTHER panels' saved output files, not
  their Python return values — see D's own future spec).

No decorator, no registry class, no base class to subclass: discovery is
`glob.glob('tools/panels/panel_*.py')` — filename convention only. Adding a panel is
adding a file; removing one is deleting a file. (Considered a decorator-based registry
and class-based `Panel` objects — both add machinery for no benefit here: panels are
one-shot stateless actions, not objects needing multiple methods or shared mutable state.
A registry also means two edits — new file, plus a registration call — where filename
convention needs one.)

### `ctx`

A small dataclass, built once by `analysis_pretrain.py`'s `__main__` before dispatching to
selected panels:

```python
@dataclass
class PanelContext:
    config: dict
    output_dir: str
    device: torch.device
    args: argparse.Namespace          # raw CLI args, for panel-specific flags if ever needed
    model: Optional[torch.nn.Module] = None
    dataset: Optional[object] = None
    checkpoint: Optional[str] = None
```

A panel with `NEEDS_CHECKPOINT = NEEDS_DATASET = False` simply never reads
`ctx.model`/`ctx.dataset`.

### CLI wiring (`analysis_pretrain.py`)

New `--panel <name>` argument, `action='append'` (repeatable: `--panel profile --panel
<other>`). When `args.panel` is non-empty:
1. Import each named module from `tools.panels`, validate `'pretrain' in STAGES`
   (`analysis_pretrain.py` only ever runs `pretrain`-stage panels; `analysis_finetune.py`
   gets the same `--panel` flag wired the same way in a follow-up touch, filtering on
   `'finetune' in STAGES` — both scripts share the discovery/dispatch code, factored into
   `tools/panels/__init__.py` or `tools/analysis/__init__.py` as a small helper so it isn't
   duplicated between the two CLI scripts).
2. If any selected panel has `NEEDS_CHECKPOINT = True`, `--checkpoint`/`--config` are
   required as today (existing error path). If none do, they're optional — a
   checkpoint-free run (like `--panel profile`) doesn't need `config/analysis.json` or a
   checkpoint at all; `ctx.config` in that case comes from `config/config.json` directly
   (`profile_model.py`'s own current source), not the analysis overlay.
3. Build `PanelContext`, resolving `model`/`dataset` only if the selected panels need them.
4. Call each selected panel's `run(ctx)` in order given.

When `args.panel` is empty (the default, no `--panel` given), behavior is byte-identical to
today: `--analysis snapshot/codebook/both` drives the existing class-based path, unaffected
by any of the above.

### `profile_model.py` retirement

Deleted. `count_parameters`, `ProfilerHooks`, and the timing-loop body of `profile_model()`
move into `tools/analysis/profile.py` as `run_profile(config, device, train_mode) ->
ProfileResult` (a small dataclass: `param_map: dict[str, int]`, `time_stats: dict[str,
float]`, `loss_ms: float`, `total_ms: float`, plus the `children` name-order list needed to
print in the original order). `panel_profile.py`'s `run(ctx)`: reads `args.train` (the
`--train` flag moves from `profile_model.py`'s own `argparse` to
`analysis_pretrain.py`'s parser, panel-specific), calls `run_profile`, prints the same table
`profile_model.py` printed today (same columns, same format — a pure relocation of the
print statements, not a redesign of the report).

`CLAUDE.md`'s `python profile_model.py` Commands entry replaced with
`python analysis_pretrain.py --panel profile [--train]`.

## Testing

No test suite exists (CLAUDE.md) — validation is smoke runs, consistent with the just-completed
`analysis_pretrain`/`analysis_finetune` split:
- `git mv analysis viz` → `tools/analysis`, `tools/viz`; fix every importer
  (`model/base_checker.py`, `model/base_codebook_checker.py`, `model/MeSAE/plugin.py`,
  `analysis_pretrain.py`, `analysis_finetune.py`, `train_pretrain.py`) — `git grep` sweep
  for `from analysis import\|from viz\.` / `from viz import\|import viz\b` to confirm none
  missed.
- `python -c "import tools.analysis, tools.viz, tools.panels.panel_profile, analysis_pretrain,
  analysis_finetune, model.base_checker, model.base_codebook_checker, model.MeSAE.plugin"` —
  clean, proves the relocation didn't silently break the untouched class hierarchy.
- `python analysis_pretrain.py --panel profile` and `--panel profile --train` — compare
  printed table against a captured baseline from running today's `profile_model.py` and
  `profile_model.py --train` before the move (same component names, param counts, and
  comparable timing figures — timings won't be bit-identical run to run, but the same
  components/order/format must appear).
- `python analysis_finetune.py --panel profile` — same panel, same output shape, proving
  `--panel` dispatch genuinely works from both entrypoints via the shared helper, not just
  `analysis_pretrain.py`.
- `python analysis_pretrain.py --config config/analysis.json --checkpoint <existing
  checkpoint> --analysis snapshot ...` (no `--panel`) — reruns the existing snapshot path,
  confirms byte-identical behavior to before A (proves A is additive, not disruptive).

## Out of scope (explicit)

- Migrating any existing snapshot/codebook panel's logic into `panel_*.py` files (B, C).
- Any change to `model/base_checker.py`, `model/base_codebook_checker.py`,
  `model/MeSAE/plugin.py`.
- The summary/aggregation panel (D) — needs B/C's panels to exist first.
- Relocating `viz/extract.py`'s or `viz/timeseries.py`'s calculation content into
  `tools/analysis/` — B/C's job, tied to the panels that use them.
- `probes/` — untouched, per the standing ruling; nothing from it is ported into this
  panel system.
- Any panel-selection logic beyond the shared discovery/dispatch helper and the `--panel`
  flag itself — no config-driven panel lists, no default-panel-set-per-stage beyond "none
  unless `--panel` is given". `analysis_finetune.py` DOES get `--panel` wired (same shared
  helper, filtered on `'finetune' in STAGES`), since `panel_profile.py` declares both
  stages and testing it from both entrypoints is part of proving the contract — this is a
  small, symmetric addition, not deferred.
