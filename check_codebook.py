import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import csv
import re
from tqdm import tqdm
from torch.utils.data import DataLoader

from model.factory import build_model_from_config, build_preprocessing_from_config
from IO.dataset import build_dataset_from_config

# =============================================================================
# 1. Analysis Metrics
# =============================================================================

def calc_effective_rank(matrix):
    """
    Computes the effective rank of a matrix using singular value distribution.
    Ref: Roy & Vetterli, 'The effective rank: A measure of effective dimensionality'
    """
    if matrix.numel() == 0:
        return 0.0
    
    try:
        _, S, _ = torch.linalg.svd(matrix.float(), full_matrices=False)
    except:
        return 0.0
        
    s_sum = torch.sum(S)
    if s_sum == 0:
        return 0.0
        
    p = S / s_sum
    p = p[p > 0] 
    
    entropy = -torch.sum(p * torch.log(p))
    return torch.exp(entropy).item()

def compute_cosine_sim_matrix(tensor):
    """
    Computes pairwise cosine similarity matrix for a set of vectors.
    tensor: (N, D) or (D, N) -> we want to compare the N vectors.
    Returns (N, N) similarity matrix.
    """
    # Ensure vectors are in rows for F.cosine_similarity or manual matmul
    # Let's assume input is (N, D)
    normed = F.normalize(tensor, p=2, dim=-1)
    sim = torch.matmul(normed, normed.t())
    return sim

# =============================================================================
# 2. Main Analysis Logic
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Config & Model
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    model_name = config['training_params']['model_name']
    model_type = config['training_params']['model_type']
    
    if model_type != "AttnVQ":
        print(f"Error: This refactored script is specifically for AttnVQ models. Found: {model_type}")
        return

    model = build_model_from_config(config).to(device)
    
    # Load Weights
    # Check multiple possible checkpoint locations
    ckpt_paths = [
        f"output/{model_name}/tokenizer/best_tokenizer.pth",
        f"output/{model_name}/backbone/best_backbone.pth",
        f"output/{model_name}/checkpoints/best_model.pth"
    ]
    
    sd = None
    for path in ckpt_paths:
        if os.path.exists(path):
            print(f"Loading checkpoint: {path}")
            ckpt = torch.load(path, map_location=device)
            sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            break
            
    if sd is not None:
        msg = model.load_state_dict(sd, strict=False)
        print(f"Loaded weights. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
    else:
        print("WARNING: Using random weights (no checkpoint found in expected locations).")
    
    model.eval()
    
    # Extract Matrix A
    if not hasattr(model, 'attnvq') or not hasattr(model.attnvq, 'A'):
        print("Error: Could not find AttnVQ.A in the model.")
        return
    
    A = model.attnvq.A.detach().cpu() # (D, Hr)
    D, Hr = A.shape
    H = model.attnvq.num_heads
    r = Hr // H
    
    print(f"\n=== Matrix A Dimensions ===")
    print(f"D (Embedding Dim): {D}")
    print(f"H (Heads):         {H}")
    print(f"r (Vocab per Head): {r}")
    print(f"Total Vectors:     {Hr}")

    # Setup Directories
    viz_dir = f"output/{model_name}/visualization/matrix_a_health"
    csv_dir = os.path.join(viz_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Inter-Head Mean Similarity
    # -------------------------------------------------------------------------
    print("\nCalculating Inter-Head Mean Similarity...")
    # Reshape A to (D, H, r)
    A_reshaped = A.view(D, H, r)
    # Average over r (dim 2) -> (D, H)
    A_means = A_reshaped.mean(dim=2) 
    # Transpose to (H, D) to compare the H vectors
    head_sim_matrix = compute_cosine_sim_matrix(A_means.t())
    
    plt.figure(figsize=(10, 8))
    plt.imshow(head_sim_matrix.numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title(f"Inter-Head Mean Similarity (H={H})")
    plt.xlabel("Head Index")
    plt.ylabel("Head Index")
    plt.savefig(os.path.join(viz_dir, "1_inter_head_similarity.png"))
    plt.close()

    # -------------------------------------------------------------------------
    # 2. Global Vector Similarity
    # -------------------------------------------------------------------------
    print("Calculating Global Vector Similarity...")
    # A is (D, Hr), we want to compare the Hr vectors. Transpose to (Hr, D).
    global_sim_matrix = compute_cosine_sim_matrix(A.t())
    
    plt.figure(figsize=(12, 10))
    plt.imshow(global_sim_matrix.numpy(), cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title(f"Global Vector Similarity (Total Vectors={Hr})")
    plt.xlabel("Vector Index (H*r)")
    plt.ylabel("Vector Index (H*r)")
    
    # Add grid lines to demarcate heads
    for i in range(1, H):
        plt.axhline(i * r - 0.5, color='white', linewidth=0.5, alpha=0.5)
        plt.axvline(i * r - 0.5, color='white', linewidth=0.5, alpha=0.5)
        
    plt.savefig(os.path.join(viz_dir, "2_global_vector_similarity.png"))
    plt.close()

    # -------------------------------------------------------------------------
    # 3. Per-Head Singular Values
    # -------------------------------------------------------------------------
    print("Calculating Per-Head Singular Values...")
    plt.figure(figsize=(10, 6))
    
    sv_data = []
    for h in range(H):
        # Extract head slice (D, r)
        A_h = A_reshaped[:, h, :] 
        U, S, Vh = torch.linalg.svd(A_h, full_matrices=False)
        plt.plot(S.numpy(), alpha=0.5, label=f'H{h}' if H <= 8 else None)
        sv_data.append(S.numpy())
    
    plt.yscale('log')
    plt.title(f"Singular Value Decay per Head (r={r})")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Value (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    if H <= 8: plt.legend()
    plt.savefig(os.path.join(viz_dir, "3_per_head_singular_values.png"))
    plt.close()

    # -------------------------------------------------------------------------
    # 4. Per-Head Low-Rank Perplexity
    # -------------------------------------------------------------------------
    print("Calculating Per-Head Low-Rank Perplexity...")
    # Extract avg_probs (H, r, N)
    if hasattr(model.attnvq, 'avg_probs'):
        probs = model.attnvq.avg_probs.detach().cpu() # (H, r, N)
        # Compute entropy per expert: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1) # (H, r)
        perplexity = torch.exp(entropy) # (H, r)
        
        # Theoretical range for N discrete choices: [1, N]
        num_discrete = config['model_params'][model_type]['tokenizer']['vq_num_discrete']
        
        plt.figure(figsize=(12, 8))
        plt.imshow(perplexity.numpy(), cmap='magma', aspect='auto', vmin=1.0, vmax=float(num_discrete))
        plt.colorbar(label='Perplexity')
        plt.title("Per-Expert Perplexity (Low-Rank Selection)")
        plt.xlabel("Rank Index (r)")
        plt.ylabel("Head Index (H)")
        plt.savefig(os.path.join(viz_dir, "4_per_head_perplexity.png"))
        plt.close()
        
        # Save to CSV
        perp_csv = os.path.join(csv_dir, "per_head_perplexity.csv")
        np.savetxt(perp_csv, perplexity.numpy(), delimiter=",")
    else:
        print("Warning: avg_probs not found in model, skipping perplexity check.")

    # -------------------------------------------------------------------------
    # 5. Consolidation
    # -------------------------------------------------------------------------
    # Save some summary stats
    summary_path = os.path.join(csv_dir, "matrix_a_summary.csv")
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Global_Eff_Rank', calc_effective_rank(A)])
        writer.writerow(['Avg_Inter_Head_Sim', (head_sim_matrix.sum() - H).item() / (H * (H - 1))])
        writer.writerow(['Avg_Global_Sim', (global_sim_matrix.sum() - Hr).item() / (Hr * (Hr - 1))])

    print(f"\nRefactor complete. Plots and CSVs saved to: {viz_dir}")

if __name__ == "__main__":
    main()
