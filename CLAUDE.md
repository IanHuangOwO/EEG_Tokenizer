# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Environment:** run everything in the `eeg_fm` conda env (`/home/mamechin/anaconda3/envs/eeg_fm/bin/python`: torch 2.14, mne 1.12.1, mne-icalabel 0.9.0), not `base`. `base` has no `mne`, so `IO/loader.py`'s `get_standard_coords` silently falls back to flat polar coordinates from `metadata.json` (z=0, radius up to ~1) instead of MNE 3-D positions in meters. The Phase 1 LOSO runs and the 100-epoch BCICIV2a rerun ran in `base`, so they used the fallback coordinates; the torch version difference does not matter, the coordinates might. Check the env before comparing any finetune number against them. Phase 2 (2026-09-21 on) runs in `eeg_fm`.

```bash
# Install dependencies (CUDA 11.8)
pip install -r requirements.txt

# Pretrain: one run, two phases -- unmasked tokenizer phase for
# training_params.pretrain.tokenizer_epochs, then masked phase (docs/adr/0013)
python train_pretrain.py --config config/config.json

# Profile model
python profile_model.py

# Run Finetune stage (loads a pretrain checkpoint, trains a classification head)
python train_finetune.py --config config/config.json

# Post-training checker (checkpoint -> topo/PSD/attn snapshot per subject;
# base config auto-derived from the checkpoint's output/<model>/artifacts/config.json,
# config/analysis.json is a small overlay; viz/extract.py, panels.py, timeseries.py,
# topomap.py are shared primitives it and model/base_checker.py both call — not run directly)
python check_model.py --config config/analysis.json --checkpoint <path>

# Compile raw datasets into per-subject bandpass+resample-baked .npz caches (run once, or
# after changing sample_freq/bandpass_filter — see config/compile.json, docs/agents/adding-a-dataset.md)
python cache_compile.py --config config/compile.json

# Sanity-check a compiled cache (shape/labels/dead-channels/bandpass-rolloff, --deep for a
# raw-vs-cache diff)
python cache_verify.py --config config/compile.json
```

No test suite exists. Validation runs during training.

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
  └─ cache_compile.py          # bandpass filter → resample, baked once into datas/<Name>/cache/*.npz
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
  `PerChannelHeadAttn` (finetune head).
- **`model/MeSAE/MeSAE.py`**: `MeSAEPretrain` (phases, `get_loss`, `freeze_stamps`,
  `encode_post_stamp_expert`), `MeSAEFinetune`.
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

### Config (`config/config.json`)

Key fields:
- `model_params.MeSAE.pretrain`: the one architecture block — `patch_len`, `embed_dim`, `enc_depth`, `pool_after_blocks` (also the tokenizer-phase block set), `moe_ffn`, `stamp_bank`, `loss`. `model_params.MeSAE.finetune`: head params
- `preprocess_params`: `window_length`, `window_pad_threshold`, `patch_length`, `patch_stride` (patch step in samples within a Window; equal to `patch_length` for non-overlapping patches, smaller for overlapping — see `IO/preprocessing.py`'s `slice_patches`), `sample_freq`, `bandpass_filter` (`l_freq`/`h_freq`), `normalization_type`, `masking_strategy` (random/complementary/random_to_complementary — the last ramps random into complementary over a curriculum, see `IO/masking.py`)
- `dataset_params.pretrain`: dataset name → `dataset_path`, `subject_to_use` (`["all"]` or list), `channels_to_use` — used by `train_pretrain.py` (masking applied only in the masked phase)
- `training_params.pretrain`: `model_name` (output dir), `epochs` (total), `tokenizer_epochs` (unmasked phase length), `freeze_stamps`, `warmup_epochs`, `batch_size`, `device`, LR fields
- `training_params.visualize_params`: diagnostic/plotting-only params, no effect on training data — `cmap` (matplotlib colormap for topomap/PSD panels), `fft_resolution` (Hz/bin for check_model.py's diagnostic PSD panels — `model/base_checker.py`/`model/MeSAE/plugin.py`'s `n_fft = round(sample_freq / fft_resolution)`; the dead train-time `fft_patches` path in `IO/dataset.py` is unrelated and stays unwired), `psd_freq_range` (`[l, h]` or `null` — overrides the PSD panel's plotted frequency range independent of `bandpass_filter`; `null` falls back to `bandpass_filter`'s `l_freq`/`h_freq`), `bands` (Delta/Theta/Alpha/Beta/Gamma `[lo, hi]` edges for the band-filtered reconstruction time-series panel, `viz/timeseries.py`'s `_canonical_bands` — each band is still clipped to `bandpass_filter`'s range), plus per-mode `pretrain`/`finetune` sub-keys (`targets`, `every_n_epochs`)

### Outputs

`output/<model_name>/`
- `checkpoint/best.pth` — best val-loss checkpoint (reset at the phase boundary; during
  the mask curriculum it locks onto the easiest epoch, so prefer `last.pth`)
- `checkpoint/last.pth` — every epoch
- `artifacts/config.json` — run snapshot
- `visualization/` — loss plots, topomap reconstructions

### Dataset metadata

Each dataset under `datas/<name>/metadata.json` uses a unified schema:
- `data_metadata.acquisition.sample_frequency` — used for compiling (`cache_compile.py`)
- `data_metadata.channels` — 1-indexed dict with `label` + `coordinates` (polar angle/radius, converted to 3D for spatial embedding — see `IO/loader.py`'s `load_coords_from_metadata`)
- `data_structure` — per-subject file references, `raw/`-prefixed (relative to `datas/<name>/`)

Subject-level train/val split is done by shuffling subject IDs (seed 42) at `train_val_split` ratio — **data never leaks between subjects**.

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
cache `cache_compile.py` reads — no registry to edit, directory presence is the
registration. See `docs/agents/adding-a-dataset.md`.

### Adding a montage

`preprocess_params.canonical_channels` (cross-dataset channel unification) takes either a
named montage (`config/montages.json`, e.g. `"10-10"`) or an inline custom channel list.
Step-by-step protocol for adding a new standard (MNE-sourced) or custom montage. See
`docs/agents/adding-a-montage.md`.

### Reshape/view pitfalls

`.reshape(`/`.view(` silently scrambles data (no error) if it merges or reorders
axes that aren't already adjacent in the tensor's current dimension order — three
real instances of this hit training data and the MeSAE reconstruction loss in the
same session (see `IO/preprocessing.py` `window_continuous_signal`, `model/MeSAE/MeSAE.py`
`_patch_pyramid_levels`). Check any new `.reshape(`/`.view(` against
`docs/agents/reshape-pitfalls.md` before assuming it's correct just because
shapes match.
