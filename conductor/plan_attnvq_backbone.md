# Plan: Mutual Contextual Distillation (Encoder-Only)

## Objective
Implement a **Mutual Contextual Distillation** strategy for the `AttnVQBackbone`. We will remove the direct "Local Tokenizer Target" and instead focus on a Siamese-style consistency task. The goal is for two differently masked views of the same EEG trial to predict each other's contextualized representations.

## Key Components

### 1. `IO/dataset.py` (The Data)
Modify `MaskedPretrainDataset` to generate **two disjoint masks**.
- **Split:** 50/50 partition of all patches into Set A and Set B.
- **`mask1`:** Hides Set A (Student sees B).
- **`mask2`:** Hides Set B (Student sees A).
- **Goal:** View 1 must use the context of B to "hallucinate" the features of A, and vice-versa.

### 2. `train_pretrain.py` (The Pipeline)
The training loop will now focus entirely on **Cross-View Consistency**.

#### A. Dual Forward Pass
1.  **Pass 1 (Student View 1):** Student processes trial with `mask1` $\to$ outputs `z1` (contextualized features of B).
2.  **Pass 2 (Teacher View 2):** Student processes trial with `mask2` (where A is visible) $\to$ outputs `z2` (contextualized features of A).

#### B. The Mutual Loss (Cross-Distillation)
We want View 1's "guess" for the missing Set A to match what View 2 "actually saw" for Set A.
- **The "Teacher" projection:** Since we are using the **Zero-Head** approach, we project both `z1` and `z2` through the **frozen $A$ matrix** of the tokenizer to get probability distributions $p_1$ and $p_2$ over the codebooks.
- **Symmetric KL Loss:**
  - `Loss_A = KL(log_p1[A] || p2[A].detach())` (View 1 predicts what View 2 saw in A).
  - `Loss_B = KL(log_p2[B] || p1[B].detach())` (View 2 predicts what View 1 saw in B).
- **Why this is better:** It removes the reliance on the "local" tokenizer distribution (which has no context) and instead forces the backbone to learn how a patch's representation *changes* when surrounded by other patches.

## Implementation Steps

### Phase 1: Dataset Partitioning
1. Update `MaskedPretrainDataset` to ensure `mask1` and `mask2` are perfectly disjoint and cover the entire trial.

### Phase 2: Cross-Distillation Logic
1. Update `train_pretrain.py` to:
   - Perform the two passes.
   - Project outputs through frozen $A$ matrix.
   - Apply the masked KL loss symmetrically between the two views.

### Phase 3: Alignment Monitoring
1. Monitor the similarity between the two views. If they converge, the backbone has learned a robust global context.

## Verification & Testing
- **Self-Supervision Check:** Verify that the model can successfully predict visible features from another view.
- **Feature Robustness:** Check if the resulting `z` features are more discriminative than those from a standard MAE.
