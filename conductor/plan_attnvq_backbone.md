# Plan: AttnVQ Backbone & Masked Pretraining Pipeline

## Objective
Implement a complete pretraining pipeline for the `AttnVQBackbone` using **Soft-Target Distillation** from a pretrained `AttnVQTokenizer`. This includes a new dataset class for masked trials, an updated backbone architecture, and a dedicated pretraining script.

## Key Components

### 1. `model/AttnVQ/modeling_backbone.py` (The Student)
- **Architecture Updates:**
    - Prediction Head: Use a single `nn.Linear(embed_dim, in_scales * vq_head_num * vq_head_vocab_size)`.
    - Forward Pass: Reshape logits to `(S, H, B, Tokens, r)` for vectorized loss calculation.
    - Masking: Properly apply `bool_masked_pos` in `forward`.
- **Patch Embedding:** Ensure `AttnVQPatchEmbed` returns `(B, C*P, D)` where `Tokens = C*P`.

### 2. `IO/dataset.py` (The Data)
- **New Class `MaskedPretrainDataset`:**
    - Loads full trials: `(Channels, Time)`.
    - Reshapes into patches: `(Channels, Patches, PatchTime)`.
    - **Masking Strategy:** Generates a random boolean mask of shape `(Channels * Patches)`.
    - Default mask ratio: 75% (MAE standard).
    - Yields: `(x_patches, coords, mask)`.
- **Factory Update:** Update `build_dataset_from_config` to support `mode='pretrain'`.

### 3. `train_pretrain.py` (The Pipeline)
- **Teacher/Student Interaction:**
    - Load `AttnVQTokenizer` from a specified checkpoint. Freeze it (`eval()`, `no_grad()`).
    - Initialize `AttnVQBackbone`.
- **Training Logic:**
    - For each batch `(x_patches, coords, mask)`:
        1. Reshape `x_patches` for Teacher: `(B, C, P, T) -> (B*P, C, T)`.
        2. Teacher Forward: Get `weights` (shape `(S, B*P, C, H, r)`) and `gate_weights` (shape `(S, H)`).
        3. Student Forward: Pass `x_patches` and `mask`. Get `logits` (shape `(S, H, B, C*P, r)`).
        4. **Target Alignment:** Reshape Teacher `weights` to `(S, H, B, C*P, r)`.
        5. **Loss:** Compute Weighted KL-Divergence:
           `Loss = Sum(GateWeight[s,h] * KL(LogSoftmax(StudentLogits[s,h]), TeacherWeights[s,h]))`
        6. Apply loss only to masked tokens (or all tokens, configurable).
- **Logging & Visualization:**
    - Log KL loss per scale/head.
    - Monitor "Prediction Sharpness" (how well the student matches the teacher's confidence).

## Implementation Steps

### Phase 1: Backbone & Dataset (Structural)
1. Update `modeling_backbone.py` with the reshaped head and masking logic.
2. Implement `MaskedPretrainDataset` in `IO/dataset.py`.

### Phase 2: Pretraining Script (Logic)
1. Create `train_pretrain.py` based on `train_tokenizer.py` structure.
2. Implement the `WeightedKLLoss` function.
3. Add logic to load the teacher's checkpoint.

### Phase 3: Verification & Hyperparameters
1. Verify dimension alignment between Teacher weights and Student logits.
2. Test with a small dataset to ensure loss decreases.
3. Tune mask ratio and learning rate.

## Verification & Testing
- **Dimension Check:** `Teacher Weights (S, H, B, C*P, r) == Student Logits (S, H, B, C*P, r)`.
- **Masking Check:** Ensure `mask_token` is correctly placed and that the encoder ignores masked content.
- **Teacher Freeze:** Explicitly check that `teacher.parameters()` have `requires_grad=False`.
