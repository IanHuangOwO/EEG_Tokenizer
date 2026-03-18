# Plan: Low-Rank Subspace Expert Architecture

This plan details the transition of the `AttnVQ` model to a "Subspace Expert" architecture. This approach replaces the learning codebook with a fixed orthogonal basis and uses low-rank projection matrices to force heads to specialize in strictly orthogonal feature subspaces, combined using a weighted summation.

## Core Architectural Changes

### 1. The Fixed Orthogonal Codebook
-   **Eliminate EMA:** Remove all EMA tracking buffers. The codebook will no longer be updated during training.
-   **The Codebook:** Replace the learnable `self.embedding` with a fixed **Identity Matrix** of shape `(r, r)`.
-   **Why Identity?** An Identity matrix provides perfectly orthogonal "basis vectors." Matching against an Identity matrix is mathematically equivalent to selecting the most prominent dimensions of the projected signal.
-   **Dimension ($r$):** Set $r$ to `vq_head_vocab_size`.

### 2. Low-Rank Decomposition ($A$ and $B$)
-   Replace the full-rank projections ($W_q$ and $W_o$) with two low-rank factors:
    -   **$A$ (Filter):** Projects the full 128-D signal down to the $r$-dimensional expert subspace.
    -   **$B$ (Synthesizer):** Projects the chosen $r$-dimensional codes back to the full 128-D space.
-   **Initialization:** Xavier Uniform for both factors.

### 3. Performance Optimization: Grouped Linear Projections
To ensure high speed as the number of heads ($H$) increases without devastating performance, we will use a **Grouped Linear** strategy instead of looping or splitting tensors:
-   **Flattened Parameters:** Instead of storing $H$ separate matrices, we flatten them into one large projection. We store `self.A` as a single matrix of shape `(D, H * r)` and `self.B` as a single matrix of shape `(H * r, D)`.
-   **Batch Matrix Multiplication / Linear Projection:** We project the entire batch into all $H$ subspaces simultaneously in a single highly optimized matrix multiplication pass. This is mathematically identical to separate projections per head but maximizes GPU throughput.
-   **Gating in Subspace:** Apply the `head_weights` (Softmax-normalized) directly to the subspace coordinates *before* the synthesis step.

### 4. Forward Pass Flow (Optimized)
1.  **Flatten:** $z_{flat} = \text{reshape}(z, (S \cdot B \cdot C, D))$.
2.  **Filter (Linear):** $q = z_{flat} \cdot A$. The result contains coordinates for all $H$ experts.
3.  **Reshape & Normalize:** Reshape to `(S \cdot B \cdot C, H, r)` and perform a simple Softmax over the $r$ dimensions of each expert (no Top-K, no temperature).
4.  **Gate:** $w_{gated} = w \times \text{Softmax}(\text{head\_weights})$.
5.  **Flatten & Synthesize (Linear):** Flatten back to `(S \cdot B \cdot C, H \cdot r)` and project $z_{q\_soft} = w_{gated} \cdot B$.
6.  **Unflatten:** Reshape $z_{q\_soft}$ back to `(S, B, C, D)`.

### 5. Subspace Orthogonality Loss (Diversification)
To ensure the heads learn completely distinct features, we penalize the overlap of their projection subspaces.
-   **Calculation:** Reshape the optimized $A$ and $B$ matrices back to `(H, D, r)` and calculate pairwise cross-correlations between different heads.
-   **Loss:** Penalize the squared Frobenius norm of off-diagonal head-to-head correlations.

## Modifications & Exclusions
-   **Do not edit `check_codebook.py` yet.** This will be handled in a separate, later phase.
-   **Individual Head Weights:** We will track and return the exact weight of *each individual head* rather than just the aggregate min/max/mean. This will allow us to plot and see exactly which head is dominating.

## Execution Steps for `modeling_tokenizer.py`

1.  **Refactor `__init__`**:
    -   Remove old EMA and Top-K logic.
    -   Implement the flattened `self.A` and `self.B` parameters for optimized projections.
    -   Initialize `self.head_weights` to zeros (for initial uniform distribution).
2.  **Update `get_current_metrics`**:
    -   Unroll the optimized matrices for health monitoring (Orthogonality).
    -   **Update:** Return a tensor or list containing each individual head's normalized weight for precise dominance tracking.
3.  **Optimize `forward`**:
    -   Implement the optimized Linear projection and synthesis.
    -   Implement the Softmax-based expert matching.
    -   Implement the Subspace Orthogonality Loss.
