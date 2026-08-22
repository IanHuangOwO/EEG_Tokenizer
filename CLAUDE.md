# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (CUDA 11.8)
pip install -r requirements.txt

# Run Tokenizer stage (unmasked, encoder+VQ/SAE trained jointly, spatial/temporal mixing enabled)
python train_tokenizer.py --config config/config.json

# Run Pretrain stage (masked reconstruction; loads training_params.pretrain.tokenizer_checkpoint,
# freezes VQ/SAE, trains only the transformer)
python train_pretrain.py --config config/config.json

# Profile model
python profile_model.py

# Run Finetune stage (loads a Pretrain-stage checkpoint, trains a classification head)
python train_finetune.py --config config/config.json

# Post-training checker (checkpoint -> topo/PSD/attn snapshot per subject, MeFSQ or MeSAE;
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

**MeFSQ** (Multi-head Finite Scalar Quantization) is an EEG tokenizer trained via masked reconstruction pretraining. Quantization is DeepSeekMoE-style: routed + shared Expert pools, each Expert forming its own content-based-attention-pooled **Expert View** over all channels (not one independent code per channel, and not a single shared concatenated Token — that design is retired, see `docs/adr/0002-per-expert-channel-attention.md`). See `CONTEXT.md` for canonical terms (Trial/Window/Patch/Expert View/Expert/Codebook/etc.) and `docs/adr/0001-moe-style-quantization.md` for the original MoE-quantization rationale.

### Data flow

```
EEG signals (raw dataset files)
  └─ datas/<Name>/loader.py    # dataset-specific loader, compile-time only (never runs at train time)
  └─ cache_compile.py          # bandpass filter → resample, baked once into datas/<Name>/cache/*.npz
  └─ IO/dataset.py             # EEGDataset reads the compiled cache directly, channel-maps/pads,
  │                            # applies IO/preprocessing.py's Normalizer (zscore/robust/fixed)
  │    └─ EEGDataset → TokenizerDataset / PretrainDataset / FinetuneDataset
  │         ├─ assemble_trials=True (tokenizer/pretrain): flattens Trials into continuous
  │         │  signal, cuts Windows (IO/preprocessing.py's window_continuous_signal)
  │         ├─ IO/preprocessing.py's slice_patches: Window → Patches, patch_stride for overlap
  │         └─ IO/masking.py: random / complementary / random_to_complementary masking
  │            strategies (block strategy removed) — PretrainDataset only, TokenizerDataset
  │            is always unmasked
  └─ train_tokenizer.py        # Tokenizer stage: unmasked, encoder+VQ/SAE joint, AdamW + cosine LR
  └─ train_pretrain.py         # Pretrain stage: masked reconstruction, loads Tokenizer checkpoint
```

`build_dataset_from_config` runs `sanity_check_base`/`sanity_check_wrapper` (`IO/dataset.py`)
automatically on every call — verifies the per-trial parallel arrays (labels/coords/subject_id/
dataset_name) stayed index-aligned through loading/padding/windowing, and that the wrapper's
`__getitem__` produces finite tensors, before training starts.

Pretraining is split into two sequential scripts/checkpoints (not two phases of one run):
- **`train_tokenizer.py`**: builds the model, calls `trainer.on_tokenizer_start()` (enables
  spatial + temporal mixing immediately — no waiting on masked pretrain), trains
  encoder+VQ/SAE jointly with `bool_masked_pos=None` throughout, saves
  `output/<tokenizer_model_name>/tokenizer/best_tokenizer.pth`.
- **`train_pretrain.py`**: builds the same architecture, loads that tokenizer checkpoint
  (`training_params.pretrain.tokenizer_checkpoint`), re-enables spatial/temporal (plain
  flags, not persisted in the state dict), calls `trainer.on_pretrain_start()` to freeze
  the VQ/SAE apparatus, then trains only the transformer against masked reconstruction.

Each model plugs in via `model/<Name>/plugin.py`, bundling a `Trainer`/`Checker`/`Plotter`
(subclassing `model/base_trainer.py`/`base_checker.py`/`base_plotter.py`) into a
`BasePlugin` (`model/base_plugin.py`) registered in `model/factory.py`'s
`MODEL_REGISTRY` — see `docs/adr/0004-model-plugin-base-classes.md` and
`docs/agents/adding-a-model.md`. See `model/base_trainer.py`
(`BaseTrainer.on_tokenizer_start`/`on_pretrain_start`) for the per-model training-hook
contract.

### Model (`model/MeFSQ/`)

- **`MeFSQ_modules.py`**:
  - `SpatialTemporalEmbeddings`: linear patch proj + sinusoidal time pos + 3D coord MLP → `[B, C, N, D]`
  - `TSABlock`: temporal ConvAdditiveAttn → spatial MHA (cross-channel) → ConvFFN
  - `TSAEncoder`: stack of TSABlocks
  - `Router`: top-k softmax router over the routed Expert pool; scaled dot-product gate scores, Switch-Transformer-style load-balance loss
  - `MeFSQ`: projects `[M, H, D]` through per-head learnable matrix `A` → per-head sigmoid quantization → STE; tracks codebook health (perplexity, STE gap, head diversity) via EMA buffers
  - `MultiHeadDecoder`: per-Expert nonlinear decode (down-proj → activation → up-proj), summed after decode, not before
  - `PerChannelHeadAttn`: finetune head, three-stage learnable-query attention pooling (temporal → head → channel) — replaces the earlier per-channel concat that overparameterized the classifier and caused val-chance memorization (see `docs/agents/` / memory: finetune val-chance bug)
- **`MeFSQ.py`** (`MeFSQPretrain`):
  - `stage_features` runs the full `TSAEncoder` stack and returns only the last block's output `[B, C, N, D]` — no per-head multi-stage fusion (retired; superseded by `TSAEncoder`'s UNet-style `pool_after_blocks` temporal down/up skip mechanism, which gives the encoder multi-resolution access a different way)
  - Per patch position, each Expert forms its own **Expert View** by attention-pooling all C channels' D-dim embeddings with its own learnable query (`ExpertChannelPool`) — not a single shared concatenation, so different Experts can weight channels differently for the same patch
  - VQ: each Expert's View routed through `Router` → **Routed pool** (top-k gated Experts, specialization) and **Shared pool** (always-on Experts, down-weighted 0.2x baseline) → per-Expert `vq_proj` → `MultiHeadDecoder`
  - `encode_pre_vq`: returns the continuous **Pre-VQ feature** per channel (broadcast to every Expert, not concatenated) — diagnostics only now; the finetune bypass mode that used to read this was retired (every Expert sees the identical broadcast vector, so the finetune head had nothing to differentiate)
  - `encode_post_vq`: returns the **Post-VQ feature** per Expert, per channel (each Expert's own decoded output, split back per channel, before the cross-Expert sum) — genuinely Expert-differentiated; this is what `MeFSQFinetune` reads
  - Loss: MSE split masked/unmasked; `masked_mse_weight`/`unmasked_mse_weight` scale each term explicitly; plus router load-balance loss
  - `freeze_vq_and_decoder()` locks both Expert pools + router + fusion params — called by `MeFSQTrainer.on_pretrain_start` once `train_pretrain.py` loads the Tokenizer-stage checkpoint, leaving only the transformer trainable

- **`model/factory.py`**: `build_pretrain_from_config(config, mode=...)` — dispatches on `training_params[mode].model_type` (`MeFSQ` or `MeSAE`)

### Config (`config/config.json`)

Key fields:
- `model_params.MeFSQ.pretrain`: architecture hyperparams — `patch_len`, `embed_dim`, `enc_depth`, `pool_after_blocks`, `upsample_residual_add`, `n_routed_experts`, `n_shared_experts`, `top_k`, `routed_r`/`shared_r` (codebook size), `routed_num_discrete`/`shared_num_discrete`
- `preprocess_params`: `window_length`, `window_pad_threshold`, `patch_length`, `patch_stride` (patch step in samples within a Window; equal to `patch_length` for non-overlapping patches, smaller for overlapping — see `IO/preprocessing.py`'s `slice_patches`), `sample_freq`, `bandpass_filter` (`l_freq`/`h_freq`), `fft_resolution` (Hz/bin for check_model.py's diagnostic PSD panels — `model/base_checker.py`/`model/MeSAE/plugin.py`'s `n_fft = round(sample_freq / fft_resolution)`; the dead train-time `fft_patches` path in `IO/dataset.py` is unrelated and stays unwired), `normalization_type`, `masking_strategy` (random/complementary/random_to_complementary — the last ramps random into complementary over a curriculum, see `IO/masking.py`)
- `dataset_params.pretrain`: dataset name → `dataset_path`, `subject_to_use` (`["all"]` or list), `channels_to_use` — shared by both `train_tokenizer.py` and `train_pretrain.py` (same raw data, masking applied only in the Pretrain stage)
- `training_params.tokenizer`: `model_name` (determines output dir), `epochs`, `batch_size`, `device`
- `training_params.pretrain`: same fields plus `tokenizer_checkpoint` (path to the Tokenizer stage's `best_tokenizer.pth`)

### Outputs

`output/<tokenizer_model_name>/`
- `tokenizer/best_tokenizer.pth` — best val-loss Tokenizer-stage checkpoint

`output/<pretrain_model_name>/`
- `pretrain/best_pretrain.pth` — best val-loss Pretrain-stage checkpoint
- `artifacts/config.json` — run snapshot
- `visualization/` — loss plots, topomap reconstructions (generated every 10 epochs)

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

Issues tracked in GitHub Issues (IanHuangOwO/CNE_Lab-NeuroRVQ) via `gh` CLI. See `docs/agents/issue-tracker.md`.

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
