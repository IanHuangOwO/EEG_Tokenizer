# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Environment:** run everything in the `eeg_fm` conda env (`/home/mamechin/anaconda3/envs/eeg_fm/bin/python`: torch 2.14, mne 1.12.1, mne-icalabel 0.9.0), not `base`. `base` has no `mne`, so `IO/loader.py`'s `get_standard_coords` silently falls back to flat polar coordinates from `metadata.json` (z=0, radius up to ~1) instead of MNE 3-D positions in meters. The Phase 1 LOSO runs and the 100-epoch BNCI2014001 rerun ran in `base`, so they used the fallback coordinates; the torch version difference does not matter, the coordinates might. Check the env before comparing any finetune number against them. Phase 2 (2026-09-21 on) runs in `eeg_fm`.

```bash
# Install dependencies (CUDA 11.8)
pip install -r requirements.txt

# Pretrain: one run, two phases -- unmasked tokenizer phase for
# training_params.pretrain.tokenizer_epochs, then masked phase (docs/adr/0013)
python train_pretrain.py --config configs/runs/<model_name>/pretrain.json

# Profile model (parameter counts + per-component forward-pass timing, no checkpoint/dataset needed)
python analysis_pretrain.py --panel profile [--train]

# Run Finetune stage: trains only the head on the frozen backbone's stamp-amplitude cache (or the
# patched raw signal for raw_* features); training_params.finetune.split picks intra_subject / inter_subject
python train_finetune.py --config configs/runs/<backbone>/finetune/<head>/<dataset>_<mode>.json

# Post-training checker, PRETRAIN stage (checkpoint -> topo/PSD/attn snapshot per subject,
# plus cross-dataset codebook/vocab diagnostics; base config auto-derived from the
# checkpoint's output/<model>/artifacts/config.json, configs/analysis_pretrain.template.json
# is a small overlay; tools/viz/extract.py, stamp_plots.py, timeseries.py, topomap.py are
# shared primitives it and tools/panels/ both call — not run directly)
python analysis_pretrain.py --config configs/analysis_pretrain.template.json --checkpoint <path>

# Post-training checker, FINETUNE stage (checkpoint -> per-class correct/wrong snapshot
# pairs; configs/analysis_finetune.template.json is a small overlay, different shape from
# the pretrain one -- no codebook block, dataset_params.finetune instead of .pretrain;
# --base-config is still required, a finetune run's artifacts/config_<timestamp>.json
# has no fixed name to auto-derive)
python analysis_finetune.py --config configs/analysis_finetune.template.json --base-config <path/to/artifacts/config.json> --checkpoint <head.pth>

# Compile raw datasets into per-subject bandpass+resample-baked .npz caches (run once, or
# after changing sample_freq/bandpass_filter — see configs/compile.json, docs/agents/adding-a-dataset.md).
# Verification (shape/labels/dead-channels/bandpass-rolloff) is baked in and runs automatically
# after compiling; --no-verify skips it, --deep also re-parses raw and diffs byte-for-byte,
# --verify-only skips compiling and just checks an existing cache (--dataset/--subjects narrow it)
python cache_dataset.py --config configs/compile.json

# Build the stamp-amplitude cache of the finetune datasets (frozen backbone run once per subject;
# the runner will do this automatically)
python cache_feature.py --config configs/runs/<backbone>/finetune/<head>/<dataset>_<mode>.json
```

No test suite exists. Validation runs during training.

### `configs/` layout

`configs/pretrain.template.json` / `configs/finetune.template.json` /
`configs/analysis_pretrain.template.json` / `configs/analysis_finetune.template.json` are
never pointed at by a real run directly — they're starting points.
`pretrain.template.json` holds only pretrain keys; `finetune.template.json` is an
overlay (only the finetune keys plus `"base_config"`), the same shape as a
`configs/runs/<model>/finetune/<head>/<dataset>_<mode>.json`. Analysis configs are split
the same way (below):
`analysis_pretrain.py` defaults `--config` to `analysis_pretrain.template.json`
(`dataset_params.pretrain`, `check.codebook` — pretrain-only), `analysis_finetune.py`
to `analysis_finetune.template.json` (`dataset_params.finetune`, no codebook block) —
different panels, different summaries, so one shared file stopped making sense once
finetune analysis grew its own shape. Pretrain and finetune RUN configs are split too: a
finetune overlay sets `"base_config"` to its backbone's pretrain file and `load_config`
(`tools/analysis/__init__.py`) deep-merges the two. To start a new run: copy
`pretrain.template.json` to `configs/runs/<model_name>/pretrain.json` and
`finetune.template.json` to its finetune overlays (pointing `base_config` at that
`pretrain.json`) — every model version gets its own folder from the start,
regardless of whether it has finetune runs yet (same as `output/`, where pretrain always
lands in `output/<model_name>/pretrain/`, see "Outputs"). Pretrain config lives at
`configs/runs/<model_name>/pretrain.json`; each finetune head/dataset gets its own overlay
at `configs/runs/<model_name>/finetune/<head>/<dataset>_<mode>.json` (`<mode>` is `intra`/`inter`,
matching `output/`'s own leaf-dir suffix exactly -- a dataset run under both protocols gets
two files, not one overwritten by the other) — see `docs/adr/0017` for why
finetune runs nest under their backbone, and `configs/README.md` for the full
convention — and edit that copy — never the template. `configs/finetune_eval_splits/*.json`
(seeded train/eval subject splits, see `training_params.finetune.split.eval_subjects:
"auto"` below) and `configs/compile.json`/`configs/montages.json` (not per-run) are
unchanged by this convention.

## Architecture

**MeSAE** is an EEG tokenizer: a TSA encoder feeding a sparse stamp dictionary
(StampBank, top-k routed + always-on shared stamps), reconstructing patches, pretrained
by masked reconstruction. MeFSQ (the earlier FSQ/VQ model) was removed, see
`docs/adr/0013`. See `CONTEXT.md` for canonical terms and `docs/adr/0009` for the stamp
dictionary.

### Data flow

```
EEG signals (raw dataset files)
  └─ datas/<Name>/loader.py    # dataset-specific loader, compile-time only (never runs at train time)
  └─ cache_dataset.py          # bandpass filter → resample, baked once into datas/<Name>/cache/*.npz
  └─ IO/dataset.py             # EEGDataset reads the compiled cache directly, channel-maps/pads,
  │                            # applies IO/preprocessing.py's Normalizer (zscore/robust/fixed)
  │    └─ EEGDataset → PretrainDataset / FinetuneDataset
  │         ├─ assemble_trials=True (pretrain): flattens Trials into continuous
  │         │  signal, cuts Windows (IO/preprocessing.py's window_continuous_signal)
  │         ├─ IO/preprocessing.py's slice_patches: Window → Patches, patch_stride for overlap
  │         └─ IO/masking.py: random / complementary / random_to_complementary masking
  │            strategies (block strategy removed) — PretrainDataset only; masks are
  │            ignored during the tokenizer phase
  └─ train_pretrain.py         # tokenizer phase (unmasked) -> masked phase, one run
```

`build_dataset_from_config` runs `sanity_check_base`/`sanity_check_wrapper` (`IO/dataset.py`)
automatically on every call — verifies the per-trial parallel arrays (labels/coords/subject_id/
dataset_name) stayed index-aligned through loading/padding/windowing, and that the wrapper's
`__getitem__` produces finite tensors, before training starts.

Pretraining is one run with two phases (`docs/adr/0013`):
- **Tokenizer phase** (epochs 1..`tokenizer_epochs`): `MeSAEPretrain.enter_tokenizer_phase()`
  -- only the `pool_after_blocks` encoder blocks run, temporal mixing only, no masking.
  Encoder + StampBank train jointly on single-channel features (0003's leakage rule).
- **Masked phase**: `enter_masked_phase(freeze_stamps)` -- all blocks, spatial attention +
  coord embedding on, `preprocess_params.mask` curriculum starts (counted from here),
  StampBank frozen only if `training_params.pretrain.freeze_stamps`.
- The phase is a buffer (`masked_phase`); a `load_state_dict` post-hook restores blocks and
  mixing flags, so no loader calls `enable_*` by hand.

Each model plugs in via `model/<Name>/plugin.py` (`Trainer`/`Checker`/`Plotter` in a
`BasePlugin`, registered in `model/factory.py`'s `MODEL_REGISTRY`) -- MeSAE is the only
one. See `docs/adr/0004-model-plugin-base-classes.md`.

- **`model/MeSAE/MeSAE_modules.py`**: `SpatialTemporalEmbeddings`, `TSABlock` (temporal
  attn -> spatial MHA -> MoE FFN, LayerScale 1e-4), `TSAEncoder` (UNet-style temporal
  pool/upsample at `pool_after_blocks`, `active_blocks` bypass), `StampBank`,
  and the finetune pieces (ADR 0016): `StampExtractor` (frozen-backbone stamp codes
  `[B, N', Cv, S, 2]`) and `FeatureHead` (one composable head: feature front-end, spatial
  filter, time pooling, optional branches, readout).
- **`model/MeSAE/MeSAE.py`**: `MeSAEPretrain` (phases, `get_loss`, `freeze_stamps`,
  `encode_post_stamp_expert`), `FinetuneModel` (frozen backbone + `StampExtractor` +
  `FeatureHead`), `build_finetune`.
- **`model/factory.py`**: `build_pretrain_from_config`, `build_finetune_from_config`,
  `optimizer_param_groups`.

### MeSAE sparsity budget — a hard ceiling, not a knob

Each active stamp slot contributes two free scalars (`a`, `b`) per channel, so the
reconstruction has `2 * (stamp_top_k + n_shared_stamps)` degrees of freedom against a
`patch_len`-sample target. **Keep `2 * (top_k + n_shared) < patch_len`, with margin.**

Past that line the active slots alone can fit any patch exactly regardless of what the
atoms contain, and it stops being sparse coding: measured at `top_k=24` (DOF 56 >
patch_len 50), `recon_mse` collapsed to ~0 on every dataset at once while activation
kurtosis fell 6.68 -> 1.17 and cross-atom correlation quadrupled. Current default sits
at 32. See `docs/adr/0011-matching-pursuit-residual-loss.md`.

### Config (`configs/pretrain.template.json` + `configs/finetune.template.json`, copied per run into `configs/runs/` — see above)

Key fields:
- `model_params.MeSAE.pretrain`: the one architecture block — `patch_len`, `embed_dim`, `enc_depth`, `pool_after_blocks` (also the tokenizer-phase block set), `moe_ffn`, `stamp_bank`, `loss`. `model_params.MeSAE.finetune`: head keys (numeric, validated at build; the checkpoint stores the resolved `head_config`) — `features` (list, one or more of `stamp_power`/`stamp_band`/`raw_band`/`raw_signal`/`phase_advance`/`evoked`; `phase_advance`/`evoked` need exactly one of `stamp_power`/`stamp_band` in the same list — see `docs/superpowers/specs/2026-09-22-list-feature-head-design.md`), `spatial_k` (one shared value across every entry), `time_pool` (`flat`/`learned`/`window`/`none`, default for every entry), `time_rank`, `window` (`[lo, hi]` s), `evoked_rank`, `overrides` (`{entry_name: {time_pool/time_rank/window/evoked_rank}}`, per-entry override of the defaults above), `dropout`; old bare `feature: "<name>"` configs/checkpoints still load (normalized to a one-element `features` list); defaults in `docs/adr/0016`
- `preprocess_params`: `window_length`, `window_pad_threshold`, `patch_length`, `patch_stride` (patch step in samples within a Window; equal to `patch_length` for non-overlapping patches, smaller for overlapping — see `IO/preprocessing.py`'s `slice_patches`), `sample_freq`, `bandpass_filter` (`l_freq`/`h_freq`), `normalization_type`, `masking_strategy` (random/complementary/random_to_complementary — the last ramps random into complementary over a curriculum, see `IO/masking.py`)
- `dataset_params.pretrain`: dataset name → `dataset_path`, `subject_to_use` (`["all"]` or list), `channels_to_use` — used by `train_pretrain.py` (masking applied only in the masked phase)
- `training_params.pretrain`: `model_name` (clean identity string, e.g. logged at startup — not a path), `output_path` (optional; where this run writes under `output/`, e.g. `mesae_v10_small/pretrain` — falls back to `model_name` when omitted, see "Outputs" below), `epochs` (total), `tokenizer_epochs` (unmasked phase length), `freeze_stamps`, `warmup_epochs`, `batch_size`, `device`, LR fields
- `dataset_params.finetune` (exactly one dataset per run) / `training_params.finetune`: `model_name`, `output_path` (same split as `training_params.pretrain`'s), `pretrained_checkpoint`, `learning_rate`, `min_learning_rate`, `weight_decay`, `epochs`, `warmup_epochs`, `batch_size`, `device`, `seed`, and the `split` block — `{"mode": "intra_subject", "n_folds": k}` (per-subject k-fold over that subject's own trials) or `{"mode": "inter_subject", ...}` with exactly one of `n_folds` (subject k-fold; k = number of subjects is LOSO) or `eval_subjects` (list, dict of named groups, or `"auto"` -- resolves to the cached/generated `configs/finetune_eval_splits/<dataset>.json` seen/unseen split for datasets registered in `tools.analysis.select_eval_subsets.DATASETS` (currently empty -- register a dataset there before using `"auto"`); errors rather than falling back to a different split mode for an unregistered dataset), plus optional `train_subjects` (ignored when `eval_subjects="auto"`, which fills it in) and `seed`. Old keys (`split_mode`, `cv_folds`, `train_val_split`, `freeze_backbone`, `backbone_lr_mult`) are gone; see `docs/superpowers/plans/2026-09-21-finetune-restructure-c-train-finetune.md`
- `training_params.visualize_params`: diagnostic/plotting-only params, no effect on training data — `cmap` (matplotlib colormap for topomap/PSD panels), `fft_resolution` (Hz/bin for analysis_pretrain.py's/analysis_finetune.py's diagnostic PSD panels — `tools/panels/panel_stamp_gallery.py`/`panel_stamp_by_patch.py`'s `n_fft = round(sample_freq / fft_resolution)`; the dead train-time `fft_patches` path in `IO/dataset.py` is unrelated and stays unwired), `psd_freq_range` (`[l, h]` or `null` — overrides the PSD panel's plotted frequency range independent of `bandpass_filter`; `null` falls back to `bandpass_filter`'s `l_freq`/`h_freq`), `bands` (Delta/Theta/Alpha/Beta/Gamma `[lo, hi]` edges for the band-filtered reconstruction time-series panel, `tools/viz/timeseries.py`'s `_canonical_bands` — each band is still clipped to `bandpass_filter`'s range), plus per-mode `pretrain`/`finetune` sub-keys (`targets`, `every_n_epochs`)

### Outputs

A run's actual write location under `output/` is `training_params.<mode>.output_path`
(`tools/analysis/__init__.py`'s `resolve_output_path`, read by `train_pretrain.py`/
`train_finetune.py` and every `resolve_output_dir`/`resolve_finetune_analysis_dir` call).
When unset it defaults to `"<model_name>/pretrain"` for a pretrain run and plain
`model_name` for a finetune run. `model_name` itself stays a clean identity string (what
gets logged at startup, e.g. `"mesae_v10_small"`) and never carries a path segment —
`output_path` is the only field allowed to.

`output/<backbone>/` holds one backbone -- e.g. `output/mesae_v10_small/`; there is no
wrapping `output/pretrain/` layer. Its pretrain artifacts (`checkpoint/`, `artifacts/`,
`visualization/`, `analysis/`, `feature_cache/`) always live under
`output/<backbone>/pretrain/`, from the first pretrain run on, so later finetune runs
(`output/<backbone>/finetune/`, below) never share a level with them. Backbones trained
before this default (e.g. `mesae_v11_small`) may still be flat at the top level until
moved.
`output/archive/` holds the earlier finetune experiments (`experiment_b`, `experiment_c`, `loso_phase1`, `phase2`): superseded
by the restart on the corrected pipeline, kept as the record of why the head was chosen (their stamp-head numbers ran without
MNE coordinates or with the old pipeline, see ADR 0014). New finetune runs write to `output/<output_path>/` (`output_path` may contain a
subfolder, e.g. `mesae_v10_small/finetune/learned/BNCI2014001_intra` — finetune runs nest under the backbone's own
`output/<backbone>/` dir rather than top-level, one subfolder per head, since a second backbone finetuning the
same baseline matrix would otherwise collide at the same paths; see ADR 0017).

`output/<output_path>/` (pretrain runs; a finetune run instead writes `finetune/run_<name>/head.pth` (head checkpoint), `artifacts/group_eval.json` (per-subject tail/last balanced accuracy), `artifacts/config_<timestamp>.json` (config plus `env` stamp) and `visualization/run_<name>/training_dashboard.png`)
- `checkpoint/best.pth` — best val-loss checkpoint (reset at the phase boundary; during
  the mask curriculum it locks onto the easiest epoch, so prefer `last.pth`)
- `checkpoint/last.pth` — every epoch
- `artifacts/config.json` — run snapshot
- `visualization/` — loss plots, topomap reconstructions
- `feature_cache/<dataset>/<key>/<subject>.npz` — regenerable stamp-amplitude cache built by `cache_feature.py`, safe to delete

### Dataset metadata

Each dataset under `datas/<name>/metadata.json` uses a unified schema:
- `data_metadata.acquisition.sample_frequency` — used for compiling (`cache_dataset.py`)
- `data_metadata.channels` — 1-indexed dict with `label` + `coordinates` (polar angle/radius, converted to 3D for spatial embedding — see `IO/loader.py`'s `load_coords_from_metadata`)
- `data_structure` — per-subject file references, `raw/`-prefixed (relative to `datas/<name>/`)

Pretrain's subject-level train/val split is done by shuffling subject IDs (seed 42) at `train_val_split` ratio — **data never leaks between subjects**.

### Multi-dataset training

`build_dataset_from_config` supports multiple entries in `dataset_params`. Channels are unified from the first dataset; other datasets are mapped onto that channel space (missing channels zero-padded).

## Agent skills

### Issue tracker

Issues tracked in GitHub Issues (IanHuangOwO/EEG_Tokenizer) via `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout — `CONTEXT.md` + `docs/adr/` at repo root. See `docs/agents/domain.md`.

### Adding a model

Step-by-step protocol (files to write, contracts to match, verification commands) for
wiring a new tokenizer model into the shared plugin architecture (see
`docs/adr/0004-model-plugin-base-classes.md`). See `docs/agents/adding-a-model.md`.

### Adding a dataset

Step-by-step protocol for converting a raw EEG dataset into the standard `datas/<name>/`
layout (`loader.py`, `gen_metadata.py`, `raw/`) and compiling it into the per-subject
cache `cache_dataset.py` reads — no registry to edit, directory presence is the
registration. See `docs/agents/adding-a-dataset.md`. Datasets MOABB covers use
`IO/loader.py`'s `MoabbLoader` instead of hand-written parsing. `datas/DATASETS.md` lists every
dataset (paradigm, subjects, compiled hours, status); regenerate it with
`python -m tools.misc.dataset_inventory` whenever a dataset is added, compiled or migrated.

### Adding a montage

`preprocess_params.canonical_channels` (cross-dataset channel unification) takes either a
named montage (`configs/montages.json`, e.g. `"10-10"`) or an inline custom channel list.
Step-by-step protocol for adding a new standard (MNE-sourced) or custom montage. See
`docs/agents/adding-a-montage.md`.

### Adding a tool (panel / analysis / viz function)

`tools/` is `analysis/` (calculation), `viz/` (pure rendering), `panels/` (thin
`panel_<name>.py` CLI entrypoints, discovered by filename glob, no registry). Protocol
for which subpackage new code belongs in, the panel contract
(`STAGES`/`NEEDS_CHECKPOINT`/`NEEDS_DATASET`/`run(ctx)`), and CLI wiring. See
`docs/agents/adding-a-tool.md`. `tools/misc/` holds one-off ad-hoc analysis scripts
(run directly, `python tools/misc/<script>.py`, no panel contract/CLI wiring) -- graduate
a script into `analysis/`+`panels/` if it becomes routine.

### Reshape/view pitfalls

`.reshape(`/`.view(` silently scrambles data (no error) if it merges or reorders
axes that aren't already adjacent in the tensor's current dimension order — three
real instances of this hit training data and the MeSAE reconstruction loss in the
same session (see `IO/preprocessing.py` `window_continuous_signal`, `model/MeSAE/MeSAE.py`
`_patch_pyramid_levels`). Check any new `.reshape(`/`.view(` against
`docs/agents/reshape-pitfalls.md` before assuming it's correct just because
shapes match.
