# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (CUDA 11.8)
pip install -r requirements.txt

# Run pretraining
python train_pretrain.py --config config/config_pretrain.json

# Profile model
python profile_model.py

# Visualization / analysis scripts
python run_analysis.py
python viz/check_recon.py
python viz/check_codebook.py
python viz/check_epoch.py
```

No test suite exists. Validation runs during training.

## Architecture

**MeFSQ** (Multi-head Finite Scalar Quantization) is an EEG tokenizer trained via masked reconstruction pretraining. Quantization is DeepSeekMoE-style: routed + shared Expert pools operating on a Token fused across all channels, not one independent code per channel. See `CONTEXT.md` for canonical terms (Trial/Window/Patch/Token/Expert/Codebook/etc.) and `docs/adr/0001-moe-style-quantization.md` for why.

### Data flow

```
EEG signals
  └─ IO/loader.py          # dataset-specific loaders (BETA, BCICIV, Inria, EEGMMIdb, Dial)
  └─ IO/preprocessing.py   # bandpass filter → resample → normalize (zscore/robust/fixed)
  └─ IO/dataset.py         # EEGDataset → MaskedPretrainDataset / FinetuneDataset
       ├─ assemble_trials=True (pretrain): flattens Trials into continuous signal, cuts Windows
       └─ IO/masking.py: random / complementary masking strategies (block strategy removed)
  └─ train_pretrain.py     # training loop, AdamW + cosine LR with linear warmup
```

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
  - Per patch position, all C channels' D-dim embeddings are concatenated into one C\*D-wide **Token** (`z_cat`) — one shared Code must explain all channels jointly
  - VQ: Token routed through `Router` → **Routed pool** (top-k gated Experts, specialization) and **Shared pool** (always-on Experts, down-weighted 0.2x baseline) → per-Expert `vq_proj` → `MultiHeadDecoder`
  - `encode_pre_vq`: returns the continuous **Pre-VQ feature** per channel (broadcast to every Expert, not concatenated) — diagnostics only now; the finetune bypass mode that used to read this was retired (every Expert sees the identical broadcast vector, so the finetune head had nothing to differentiate)
  - `encode_post_vq`: returns the **Post-VQ feature** per Expert, per channel (each Expert's own decoded output, split back per channel, before the cross-Expert sum) — genuinely Expert-differentiated; this is what `MeFSQFinetune` reads
  - Loss: MSE split masked/unmasked; `mask_weight` scales masked term; plus router load-balance loss
  - VQ warmup phase: first `vq_warmup_epochs` epochs run without masking; `freeze_vq_and_decoder()` locks both Expert pools + router + fusion params after warmup, leaving only the transformer trainable

- **`model/factory.py`**: `build_model_from_config(config)` — dispatches on `training_params.pretrain.model_type` (`MeFSQ` or `MeSAE`); copies the matching source file to `output/<model_name>/artifacts/` for reproducibility

### Config (`config/config_pretrain.json`)

Key fields:
- `model_params.MeFSQ.pretrain`: architecture hyperparams — `patch_len`, `embed_dim`, `enc_depth`, `pool_after_blocks`, `upsample_residual_add`, `n_routed_experts`, `n_shared_experts`, `top_k`, `routed_r`/`shared_r` (codebook size), `routed_num_discrete`/`shared_num_discrete`
- `preprocess_params`: `trial_length`, `target_freq`, bandpass `l_freq`/`h_freq`, `normalization_type`, `masking_strategy` (random/complementary)
- `dataset_params`: dataset name → `dataset_path`, `subject_to_use` (`["all"]` or list), `channels_to_use`
- `training_params`: `model_name` (determines output dir), `epochs`, `vq_warmup_epochs`, `batch_size`, `device`

### Outputs

`output/<model_name>/`
- `pretrain/best_pretrain.pth` — best val-loss checkpoint
- `artifacts/config.json`, `artifacts/MeFSQ.py` — run snapshot
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
