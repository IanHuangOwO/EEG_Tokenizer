import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from model.factory import build_model_from_config

def linear_cka(X, Y):
    """
    Computes Linear Centered Kernel Alignment (CKA).
    X, Y: (N, D) matrices (codebooks)
    """
    # Center the matrices
    def center(K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=K.device)
        I = torch.eye(n, device=K.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    # Linear kernels
    K = torch.matmul(X, X.t())
    L = torch.matmul(Y, Y.t())

    K_c = center(K)
    L_c = center(L)

    # Frobenius norm based similarity
    hsic = torch.sum(K_c * L_c)
    norm_k = torch.sqrt(torch.sum(K_c * K_c))
    norm_l = torch.sqrt(torch.sum(L_c * L_c))
    
    return (hsic / (norm_k * norm_l)).item()

def subspace_overlap(X, Y, threshold=0.95):
    """
    Calculates how much of the variance in X is explained by the principal components of Y.
    """
    # Get principal components of Y
    U, S, V = torch.pca_lowrank(Y, q=min(Y.shape))
    
    # Cumulative variance to find effective rank
    sum_s = torch.cumsum(S**2, dim=0)
    total_s = torch.sum(S**2)
    n_components = torch.where(sum_s / total_s >= threshold)[0][0].item() + 1
    
    # Projection matrix onto Y's subspace
    basis_y = V[:, :n_components] # (D, k)
    
    # Project X onto Y's basis
    X_projected = torch.matmul(X, torch.matmul(basis_y, basis_y.t()))
    
    # Overlap = Variance of projection / Original variance
    overlap = torch.norm(X_projected)**2 / torch.norm(X)**2
    return overlap.item(), n_components

def main():
    # 1. Load Config
    config_path = 'config/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    m_params = config['model_params']['NeuroRVQ']
    train_params = config['training_params']
    model_name = train_params.get('model_name', 'neurorvq_v1')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Initialize Model and Load Checkpoint via Factory
    model = build_model_from_config(config).to(device)

    ckpt_path = f'output/checkpoints/tokenizer/{model_name}/tokenizer_best.pth'
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    else:
        print("No checkpoint found. Analyzing RANDOM weights.")

    # 3. Extract Normalized Codebooks
    n_layers = m_params['num_codebooks']
    codebooks = []
    for i in range(n_layers):
        cb = model.rvqs[0].layers[i].embedding.weight.data
        codebooks.append(cb - cb.mean(dim=0)) # Zero-center for structural analysis

    # 4. Compute Matrices
    cka_matrix = np.zeros((n_layers, n_layers))
    overlap_matrix = np.zeros((n_layers, n_layers))
    
    print("Computing CKA and Subspace Overlap...")
    for i in range(n_layers):
        for j in range(n_layers):
            cka_matrix[i, j] = linear_cka(codebooks[i], codebooks[j])
            overlap, _ = subspace_overlap(codebooks[i], codebooks[j])
            overlap_matrix[i, j] = overlap

    # 5. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # CKA Plot
    im1 = ax1.imshow(cka_matrix, cmap='magma', vmin=0, vmax=1)
    plt.colorbar(im1, ax=ax1, label='CKA Score')
    ax1.set_title("Linear CKA (Relational Similarity)\n1.0 = Identical Structure (even if rotated)")
    
    # Subspace Overlap Plot
    im2 = ax2.imshow(overlap_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax2, label='Overlap Ratio')
    ax2.set_title("Subspace Overlap\nHow much of L(row) is inside the basis of L(col)?")

    for ax in [ax1, ax2]:
        ax.set_xticks(range(n_layers))
        ax.set_yticks(range(n_layers))
        ax.set_xticklabels(range(1, n_layers + 1))
        ax.set_yticklabels(range(1, n_layers + 1))
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Layer Index")

    output_dir = f'output/visualization/{model_name}'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'structural_similarity.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Structural analysis saved to {save_path}")

if __name__ == "__main__":
    main()
