# Plan: Refactor Tokenizer and Backbone for SSVEP (1s Patches, 4s Context)

This plan refactors the EEG Tokenizer and Backbone training pipeline to align with Foundation Model architectures. The Tokenizer focuses on local 1s spatial-spectral patches, while the Backbone models temporal dependencies across trials of varying lengths.

## Objective
- **Tokenizer:** Independent 1s patch reconstruction.
- **Backbone:** Sequence modeling of consecutive 1s patches.
- **Siamese Pretraining:** Contextual distillation across trials.

## Implementation Steps

### 1. Token Scaling & Transformer Complexity
- **Current Scaling:** The number of tokens in the Backbone scales linearly with the number of patches (`P`) and channels (`C`): `Tokens = C * P`. 
- **Analysis:** 
    - 4s window (64 ch) = 256 tokens.
    - 10s window (64 ch) = 640 tokens.
- **Future-Proofing Efficiency:**
    - To prevent the transformer from becoming too slow with very long windows, we can implement **Spatial Bottlenecks** in the future (e.g., compressing 64 channel-tokens into 4-8 latent "spatial experts" per patch).
    - For this refactor, we will maintain the **Full Detail (C * P)** approach as it is manageable for the current SSVEP targets, but we will ensure the architecture is modular enough to swap in a bottleneck later.

### 2. Length-Agnostic Design
- **Constant Patch Size:** Fixed at 1s.
- **Variable Sequence Length:** Transformer handles dynamic `P`.
- **Absolute Temporal Embeddings:** Sinusoidal embeddings based on `time_idx` (0s, 1s, 2s...) ensure consistent time-of-day encoding across different trial lengths.

### 3. Data Consistency (`IO/dataset.py`)
- `MaskedPretrainDataset` yields `x_patches` as `(C, P, T_patch)` and `time_indices` as `[0, ..., P-1]`.

### 4. Model Logic
- **Tokenizer:** Strictly 1s patches.
- **Backbone:** Processes `B x C * P` tokens. Absolute sinusoidal embeddings are applied at the patch level.

## Verification & Testing
- **Shape Verification:**
    - Tokenizer Input: `(B, 64, 200)`
    - Backbone Input: `(B, 64, P, 200)`
- **Run Training:**
    - Execute `train_tokenizer.py`.
    - Execute `train_pretrain.py`.
