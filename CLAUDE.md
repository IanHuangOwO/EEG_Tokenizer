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
```

No test suite exists. Validation runs during training.

## Architecture

**MeFSQ** (Multi-head Finite Scalar Quantization) is an EEG tokenizer trained via masked reconstruction pretraining. Quantization is DeepSeekMoE-style: routed + shared Expert pools, each Expert forming its own content-based-attention-pooled **Expert View** over all channels (not one independent code per channel, and not a single shared concatenated Token — that design is retired, see `docs/adr/0002-per-expert-channel-attention.md`). See `CONTEXT.md` for canonical terms (Trial/Window/Patch/Expert View/Expert/Codebook/etc.) and `docs/adr/0001-moe-style-quantization.md` for the original MoE-quantization rationale.

### Data flow

```
EEG signals
  └─ IO/loader.py          # dataset-specific loaders (BETA, BCICIV, Inria, EEGMMIdb, Dial)
  └─ IO/preprocessing.py   # bandpass filter → resample → normalize (zscore/robust/fixed)
  └─ IO/dataset.py         # EEGDataset → MaskedPretrainDataset / FinetuneDataset
       ├─ assemble_trials=True (pretrain): flattens Trials into continuous signal, cuts Windows
       └─ IO/masking.py: random / complementary masking strategies (block strategy removed)
  └─ train_tokenizer.py    # Tokenizer stage: unmasked, encoder+VQ/SAE joint, AdamW + cosine LR
  └─ train_pretrain.py     # Pretrain stage: masked reconstruction, loads Tokenizer checkpoint
```

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
- `preprocess_params`: `trial_length`, `target_freq`, bandpass `l_freq`/`h_freq`, `normalization_type`, `masking_strategy` (random/complementary)
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
- `data_metadata.acquisition.sample_frequency` — used for preprocessing
- `data_metadata.channels` — 1-indexed dict with `label` + `position` (3D coords for spatial embedding)
- `data_structure` — per-subject file references

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
layout and wiring it into `IO/dataset.py`. See `docs/agents/adding-a-dataset.md`.
