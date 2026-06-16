# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (CUDA 11.8)
pip install -r requirements.txt

# Run pretraining
python train_pretrain.py --config config/config.json

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

**MeFSQ** (Multi-head Finite Scalar Quantization) is an EEG tokenizer trained via masked reconstruction pretraining.

### Data flow

```
EEG signals
  └─ IO/loader.py          # dataset-specific loaders (BETA, BCICIV, Inria, EEGMMIdb, Dial)
  └─ IO/preprocessing.py   # bandpass filter → resample → normalize (zscore/robust/fixed)
  └─ IO/dataset.py         # EEGDataset → MaskedPretrainDataset / FinetuneDataset
       ├─ assemble_trials=True (pretrain): flattens trials into continuous signal, cuts windows
       └─ masking: random / block / complementary strategies
  └─ train_pretrain.py     # training loop, AdamW + cosine LR with linear warmup
```

### Model (`model/MeFSQ/`)

- **`MeFSQ_modules.py`**:
  - `SpatialTemporalEmbeddings`: linear patch proj + sinusoidal time pos + 3D coord MLP → `[B, C, N, D]`
  - `TSABlock`: temporal ConvAdditiveAttn → spatial MHA (cross-channel) → ConvFFN
  - `TSAEncoder`: stack of TSABlocks
  - `MeFSQ`: projects `[B*C, N, D]` through learnable matrix `A` → per-head sigmoid quantization → STE; tracks codebook health (perplexity, STE gap, head diversity) via EMA buffers
- **`MeFSQ.py`** (`MeFSQPretrain`):
  - Encoder with **multi-stage feature extraction**: concatenates outputs from `stage_indices` blocks
  - VQ: `MeFSQ` module → `vq_proj` linear → decoder `nn.Linear(embed_dim, patch_len)`
  - Loss: MSE split masked/unmasked; `mask_weight` scales masked term
  - VQ warmup phase: first `vq_warmup_epochs` epochs run without masking

- **`model/factory.py`**: `build_model_from_config(config)` — only `MeFSQ` type supported; copies `MeFSQ.py` to `output/<model_name>/artifacts/` for reproducibility

### Config (`config/config.json`)

Key fields:
- `model_params.MeFSQ.pretrain`: architecture hyperparams (`patch_len`, `embed_dim`, `enc_depth`, `stage_indices`, `vq_head_num`, `vq_head_vocab_size`, `vq_num_discrete`)
- `preprocess_params`: `trial_length`, `target_freq`, bandpass `l_freq`/`h_freq`, `normalization_type`, `masking_strategy` (random/block/complementary)
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
