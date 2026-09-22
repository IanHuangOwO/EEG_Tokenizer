# Split check_model.py into analysis_pretrain.py / analysis_finetune.py, new analysis/ package

**Goal:** Rename `check_model.py` to `analysis_pretrain.py`, strip every finetune-only code
path out of it so it contains only pretrain-stage analysis logic, move the finetune-only
code into a new minimal `analysis_finetune.py`, and pull the non-plotting orchestration
helpers currently squatting in `viz/__init__.py` into a new top-level `analysis/` package
that both scripts import from. This is the first step of a larger probe/analysis-dir
cleanup (the `probes/` directory itself is out of scope — see "Out of scope" below).

**Why:** `check_model.py` today branches on `mode == 'finetune'` in three separate places
(`run()`, the `__main__` CLI block, plus finetune-only helpers `_load_target_names`,
`_safe_name`, `_predict_all` that only the finetune branch calls) even though pretrain and
finetune analysis share almost nothing except the CLI scaffolding. `viz/__init__.py`
compounds this: its own docstring says "shared infrastructure for the viz package" but none
of its 8 functions (`_deep_merge`, `load_config`, `resolve_output_dir`,
`select_subject_dataset`, `filter_config_to_subject`, `load_model`, `pick_trial`,
`setup_mne_info`) render anything — they're config/dataset/model orchestration, homeless
because `check_model.py` needed *somewhere* to import them from. Splitting by stage now
(pretrain first, per the user's explicit ordering) makes each script single-purpose and
gives the later, properly-designed `analysis_finetune.py` a clean home instead of baking
today's as-is finetune branch into the renamed pretrain script.

**Scope note:** this pass moves `analysis_finetune.py`'s content **verbatim** — no redesign
of the finetune analysis logic itself. It becomes its own file so nothing is lost, and a
later task can improve it in place.

## Current state (what exists today)

- `check_model.py` (394 lines): module-level helpers `_load_target_names`, `_safe_name`,
  `_cap_subjects_by_trial_budget`, `_predict_all`; `run()` (dispatches to
  `checker.check_pretrain` or `checker.check_finetune` by `mode`); `__main__` CLI block that
  branches on `args.mode` / `overlay['mode']` throughout, including three sub-branches under
  `analysis in ('snapshot', 'both')`: a `mode == 'finetune'` branch (per-class correct/wrong
  snapshot pairs), a `visualize_params.<mode>.targets` branch (pretrain), and a default
  per-subject/per-trial branch (pretrain).
- `viz/__init__.py` (176 lines): `_deep_merge`, `load_config`, `resolve_output_dir`,
  `select_subject_dataset`, `filter_config_to_subject`, `load_model`, `pick_trial`,
  `setup_mne_info` — none of these plot anything; `viz/extract.py`, `panels.py`,
  `topomap.py`, `timeseries.py`, `codebook.py`, `iclabel.py` are the real plotting modules
  and are unaffected by this change.
- `model/base_checker.py`'s `BaseEpochChecker` already correctly separates the stage-specific
  hooks (`check_pretrain`, `check_finetune`) from the shared, stage-agnostic renderer
  (`_render_snapshot`, `_compute_spectra`) per ADR 0004 — **no change needed there**, it's
  already the right shape.
- `model/base_codebook_checker.py` / `viz/codebook.py`: pretrain-only today (codebook
  diagnostics always build `dataset_params.pretrain`, regardless of `--mode`) — confirmed
  explicitly by the user, `analysis_pretrain.py` owns this entirely.
- One live cross-file dependency on `viz/__init__.py` outside `check_model.py`:
  `train_pretrain.py:21` — `from viz import pick_trial`.
- One doc reference to the exact function that's moving: `docs/agents/adding-a-model.md:72`
  — "`enable_spatial`... Called at construction-time load (`viz/__init__.py load_model`)".
- `CLAUDE.md` documents `check_model.py` as a Commands-section entry (line 28) and mentions
  it again describing `fft_resolution` (line 121).
- `config/analysis.json` has a `"mode": "pretrain"` key that becomes redundant once mode is
  determined by which script runs, not a flag.

## Design

### New package: `analysis/`

`analysis/__init__.py` — the exact content of today's `viz/__init__.py` (all 8 functions,
verbatim, docstring updated to describe it as shared orchestration for the
`analysis_pretrain.py` / `analysis_finetune.py` scripts rather than "the viz package").

### `viz/__init__.py`

Trimmed to a short docstring noting `viz/` is plotting-only now (panels/topomap/extract/
codebook/timeseries/iclabel) and that orchestration helpers live in `analysis/`. The package
marker file stays (submodule imports like `from viz.extract import ...` are unaffected
either way).

### `check_model.py` → `analysis_pretrain.py`

Keeps, unchanged in content:
- `_cap_subjects_by_trial_budget`
- the codebook branch (`analysis in ('codebook', 'both')`)
- the `visualize_params.<mode>.targets` branch and the default per-subject/per-trial branch
  (both already pretrain-only — reached only when `mode != 'finetune'`)

Changes:
- `run()` loses its `if mode == 'finetune': return checker.check_finetune(...)` branch —
  only the `checker.check_pretrain(...)` call remains.
- `__main__` loses the `--mode` CLI argument (hardcode `mode = 'pretrain'`,
  `data_mode = 'pretrain'`) and the entire `if mode == 'finetune':` sub-branch under
  `analysis in ('snapshot', 'both')`.
- Import switches from `from viz import (_deep_merge, load_model, select_subject_dataset,
  filter_config_to_subject, pick_trial, resolve_output_dir)` to the same names from
  `analysis`.
- Module docstring updated: drop the finetune-mode mention, note this is pretrain-only.

Drops entirely (move to `analysis_finetune.py`): `_load_target_names`, `_safe_name`,
`_predict_all`.

### New `analysis_finetune.py`

- `_load_target_names`, `_safe_name`, `_predict_all` — moved verbatim.
- `run()` — finetune-only: the `checker.check_finetune(...)` call, same signature/kwargs as
  today's finetune branch of `check_model.py`'s `run()`.
- `__main__`: CLI args `--config` (default `config/analysis.json`), `--base-config`,
  `--checkpoint`, `--dataset`, `--recon_cmap`. Drops `--mode` (hardcode `mode = 'finetune'`,
  `data_mode = 'finetune'`), `--analysis` (no codebook path here), `--subject`/`--trial`
  (today's finetune branch never reads `args.subject`/`args.trial`). Body is today's
  `if mode == 'finetune':` block (per-class correct/wrong snapshot pairs), de-nested to
  top-level `__main__` flow, otherwise verbatim.
- Same config-resolution preamble as `analysis_pretrain.py` (overlay deep-merge onto the
  checkpoint's own `artifacts/config.json`, `dataset_params` override loop) — duplicated
  rather than shared, since both scripts need it and it's ~15 lines; not worth a third
  shared module for one caller-pair this small (YAGNI — revisit only if a third analysis
  entry point needs it too).

### Cross-file fixups

- `train_pretrain.py:21` — `from viz import pick_trial` → `from analysis import pick_trial`.
- `docs/agents/adding-a-model.md:72` — `viz/__init__.py load_model` → `analysis/__init__.py
  load_model`.
- `CLAUDE.md` line 28 (Commands section) — replace the single `check_model.py` entry with
  two entries, `analysis_pretrain.py` and `analysis_finetune.py`, describing each script's
  actual scope (pretrain: snapshot + codebook; finetune: per-class correct/wrong snapshot
  pairs). Line 121 (`fft_resolution` description) — update `check_model.py` mention to
  `analysis_pretrain.py`/`analysis_finetune.py` as appropriate (the PSD panel code path
  itself, `model/base_checker.py`, is unchanged).
- `config/analysis.json` — drop the now-redundant `"mode": "pretrain"` key.

## Out of scope (explicit)

- `probes/` directory: stays untouched. Per the user, it will be deleted wholesale once this
  whole refactor effort is done and the branch merges to `main` — not folded into
  `analysis/`.
- Redesigning `analysis_finetune.py`'s actual analysis logic — this pass only relocates it.
- `model/base_checker.py`, `model/base_codebook_checker.py`, and every `viz/*.py` plotting
  module — already correctly factored, no change.
- `config/config.json` and `IO/preprocessing.py`'s comments that mention `check_model.py` by
  name are historical/descriptive prose, not load-bearing; left alone (not part of this
  design — can be swept in a later docs pass if wanted).

## Testing

No test suite exists (CLAUDE.md — validation is smoke runs). Verification commands for the
implementation plan:
- `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -c "import analysis, analysis_pretrain,
  analysis_finetune, train_pretrain"` — clean import, no leftover `viz` references.
- `git grep -n "from viz import\|import viz$"` — only `viz.<submodule>` imports remain
  (extract/panels/topomap/timeseries/codebook/iclabel); no bare `from viz import <name>`
  survives outside those submodules.
- Real-checkpoint smoke: `python analysis_pretrain.py --config config/analysis.json
  --checkpoint <existing pretrain ckpt>` — compare output PNG paths/filenames against a
  pre-refactor run of `check_model.py` with the same config (same checkpoint, same
  `output/<model_name>/analysis/...` tree, same file names) — must match, since this is a
  pure relocation, not a behavior change.
- Real-checkpoint smoke: `python analysis_finetune.py --config <finetune overlay>
  --checkpoint <existing finetune head ckpt>` — same byte-identical-output check against a
  pre-refactor `check_model.py --mode finetune` run.
