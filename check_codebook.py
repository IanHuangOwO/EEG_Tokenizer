import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import re
from tqdm import tqdm
from torch.utils.data import DataLoader

from model.factory import build_model_from_config, build_preprocessing_from_config
from IO.dataset import build_dataset_from_config

# =============================================================================
# 1. Extraction (Polymorphic)
# =============================================================================

def extract_codebooks(model):
    """
    Returns list of (tensor, name).
    Example name: "L0_S1_H2" or "Shared" or "Scale1"
    """
    if hasattr(model, 'get_codebooks'):
        return model.get_codebooks()
    print(f"Model {type(model).__name__} missing get_codebooks().")
    return []

def collect_indices(model, loader, device, max_batches=None):
    """
    Returns indices tensor or None.
    Expected to return: (Total_Samples, ...)
    """
    if not hasattr(model, 'get_indices'):
        return None

    indices_acc = []
    print(f"Collecting usage indices from {type(model).__name__} (Full Dataset)...")
    
    total_steps = max_batches if max_batches is not None else len(loader)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=total_steps)):
            if max_batches is not None and i >= max_batches: break
            x, coords, _ = [t.to(device) for t in batch]
            idx, _ = model.get_indices(x, coords) # (B*N, ...)
            if idx is not None:
                indices_acc.append(idx.cpu())
                
    if indices_acc:
        return torch.cat(indices_acc, dim=0)
    return None

# =============================================================================
# 2. Analysis Metrics
# =============================================================================

def analyze_cka(cb1, cb2):
    "Linear CKA between two codebooks." 
    X = cb1.float() - cb1.float().mean(dim=0, keepdim=True)
    Y = cb2.float() - cb2.float().mean(dim=0, keepdim=True)
    
    dot_product = torch.norm(torch.matmul(X.t(), Y))**2
    norm_x = torch.norm(torch.matmul(X.t(), X))
    norm_y = torch.norm(torch.matmul(Y.t(), Y))
    
    return (dot_product / (norm_x * norm_y + 1e-8)).item()

def compute_subspace_overlap(X, Y, threshold=0.95):
    """How much of X is explained by Y's basis."""
    X_c = X.float() - X.float().mean(dim=0, keepdim=True)
    Y_c = Y.float() - Y.float().mean(dim=0, keepdim=True)
    
    # PCA on Y to find basis
    try:
        U, S, V = torch.pca_lowrank(Y_c, q=min(Y.shape))
    except:
        return 0.0
        
    sum_s = torch.cumsum(S**2, dim=0)
    total_s = torch.sum(S**2)
    
    # Determine n_comp for threshold variance
    n_comp_indices = torch.where(sum_s / (total_s + 1e-8) >= threshold)[0]
    n_comp = n_comp_indices[0].item() + 1 if len(n_comp_indices) > 0 else min(Y.shape)
    
    basis_y = V[:, :n_comp]
    
    # Project X onto Y's basis
    X_proj_coords = torch.matmul(X_c, basis_y)
    
    # Ratio of projected energy to total energy
    overlap = torch.norm(X_proj_coords)**2 / (torch.norm(X_c)**2 + 1e-8)
    return overlap.item()

def compute_cross_scale_metrics(codebooks):
    """Computes pairwise CKA and Overlap matrices."""
    n = len(codebooks)
    if n < 2: return None
    
    cka = np.zeros((n, n))
    ovl = np.zeros((n, n))
    labels = [cb[1] for cb in codebooks]
    
    print(f"\nComputing pairwise metrics for {n}x{n} matrix...")
    for i in range(n):
        for j in range(n):
            cka[i, j] = analyze_cka(codebooks[i][0], codebooks[j][0])
            ovl[i, j] = compute_subspace_overlap(codebooks[i][0], codebooks[j][0])
            
    return {'cka': cka, 'overlap': ovl, 'labels': labels}

def calc_perplexity_and_usage(indices, vocab_size):
    """
    indices: flat tensor of code ids
    """
    if indices.numel() == 0:
        return 0.0, 0.0
        
    indices = indices.flatten().long()
    counts = torch.bincount(indices, minlength=vocab_size).float()
    probs = counts / (counts.sum() + 1e-10)
    
    # Perplexity
    p = probs[probs > 0]
    entropy = -torch.sum(p * torch.log(p))
    perplexity = torch.exp(entropy).item()
    
    # Usage %
    used = (counts > 0).sum().item()
    usage_pct = (used / vocab_size) * 100.0
    
    return perplexity, usage_pct

def calc_structure_stats(codebook):
    """
    codebook: (Vocab, Dim)
    """
    cb = codebook.float()
    
    # Orthogonality (Avg Off-Diagonal Cosine Sim)
    normed = F.normalize(cb, dim=1)
    sim_matrix = torch.matmul(normed, normed.t())
    n = sim_matrix.shape[0]
    mask = torch.eye(n, device=cb.device).bool()
    avg_sim = sim_matrix[~mask].abs().mean().item()
    
    # Norm consistency
    norms = torch.norm(cb, dim=1)
    avg_norm = norms.mean().item()
    std_norm = norms.std().item()
    
    return avg_sim, avg_norm, std_norm

def calc_effective_rank(codebook):
    """
    Computes the effective rank of the codebook using singular value distribution.
    Ref: Roy & Vetterli, 'The effective rank: A measure of effective dimensionality'
    """
    if codebook.numel() == 0:
        return 0.0
    
    # SVD
    try:
        # Compute singular values (S)
        _, S, _ = torch.linalg.svd(codebook.float(), full_matrices=False)
    except:
        return 0.0
        
    # Normalize to probability distribution
    # We use L1 normalization of singular values
    s_sum = torch.sum(S)
    if s_sum == 0:
        return 0.0
        
    p = S / s_sum
    p = p[p > 0] # Avoid log(0)
    
    entropy = -torch.sum(p * torch.log(p))
    return torch.exp(entropy).item()

# =============================================================================
# 3. Aggregation & Reporting
# =============================================================================

def parse_codebook_name(name):
    """
    Parses S{}_L{}_H{} or L{}_S{}_H{} into dictionary {{L: int, S: int, H: int}}
    Returns None if pattern doesn't match.
    """
    # Regex for S#_L#_H# (New Primary)
    match = re.match(r"S(\d+)_L(\d+)_H(\d+)", name)
    if match:
        return {'S': int(match.group(1)), 'L': int(match.group(2)), 'H': int(match.group(3))}

    # Regex for L#_S#_H# (Old format support)
    match = re.match(r"L(\d+)_S(\d+)_H(\d+)", name)
    if match:
        return {'L': int(match.group(1)), 'S': int(match.group(2)), 'H': int(match.group(3))}
    
    # Regex for S#_H# (AttnVQ Multi-Head)
    match = re.match(r"S(\d+)_H(\d+)", name)
    if match:
        return {'L': 0, 'S': int(match.group(1)), 'H': int(match.group(2))}

    # Fallback: Maybe just S# (Old AttnVQ)
    match = re.match(r"S(\d+)", name)
    if match:
        return {'L': 0, 'S': int(match.group(1)), 'H': 0}
        
    return None

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Config & Model
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    model_name = config['training_params']['model_name']
    model_type = config['training_params']['model_type']
    vq_head_vocab_size = config['model_params'].get(model_type, {}).get('vq_head_vocab_size', 512)
    embed_dim = config['model_params'].get(model_type, {}).get('embed_dim', 200)
    
    # Determine max effective rank based on head structure
    vq_head_num = config['model_params'].get(model_type, {}).get('vq_head_num', 1)
    # Check if 'enc_heads' is used as fallback (typical for some models, but usually VQ is 1 head unless specified)
    # For AttnVQ, we explicitly added vq_heads. For others, it might be 1.
    eff_rank_max = embed_dim // vq_head_num
    
    # Dataset
    dataset_path = config['dataset_params']['dataset_path']
    with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
        config['data_metadata'] = json.load(f)['data_metadata']
    
    transform = build_preprocessing_from_config(config)
    dataset = build_dataset_from_config(config, transform=transform, mode='tokenizer')
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    model = build_model_from_config(config).to(device)
    
    # Load Weights
    ckpt_path = f"output/{model_name}/checkpoints/best_model.pth"
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        msg = model.load_state_dict(sd, strict=False)
        print(f"Loaded {len(sd)} keys. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
    else:
        print("WARNING: Using random weights (no checkpoint found).")
        
    model.eval()
    
    # 2. Collect Data
    codebooks = extract_codebooks(model) # List of (cb_tensor, name)
    indices_tensor = collect_indices(model, loader, device) # Limit for speed
    
    # 3. Analyze & Correlate
    # We want to create a unified table of: Name | Usage% | Perplex | Ortho | Norm
    
    stats_list = []
    
    print(f"\n=== Codebook Analysis: {model_name} ===")
    print(f"{'Codebook Name':<15} | {'Usage%':<7} | {'Perplex':<8} | {'Ortho':<8} | {'Norm':<6} | {'EffRank':<7}")
    print("-" * 75)
    
    # Prepare grids for heatmaps if structured
    # Find max dims
    max_l, max_s, max_h = 0, 0, 0
    is_structured = True
    
    for cb, name in codebooks:
        meta = parse_codebook_name(name)
        
        # Calculate Structure Stats
        ortho, avg_norm, _ = calc_structure_stats(cb)
        erank = calc_effective_rank(cb)
        
        # Calculate Usage Stats (if indices available)
        usage_pct, perplex = 0.0, 0.0
        
        if indices_tensor is not None and meta:
            # Try to map meta {L, S, H} to indices tensor dims
            # AttnVQ Indices: (T, D, S, H, K)
            try:
                # Select specific slice
                # Note: indices_tensor dimensions assumed [T, L, S, H, K]
                idx_slice = indices_tensor[:, meta['L'], meta['S'], meta['H'], :]
                perplex, usage_pct = calc_perplexity_and_usage(idx_slice, vq_head_vocab_size)
            except Exception:
                pass # Indices structure might not match name structure perfectly
        
        stats_list.append({
            'name': name,
            'meta': meta,
            'usage': usage_pct,
            'perplex': perplex,
            'ortho': ortho,
            'norm': avg_norm,
            'erank': erank
        })
        
        print(f"{name:<15} | {usage_pct:<6.1f}% | {perplex:<8.1f} | {ortho:<8.3f} | {avg_norm:<6.2f} | {erank:<7.2f}")
        
        if meta:
            max_l = max(max_l, meta['L'])
            max_s = max(max_s, meta['S'])
            max_h = max(max_h, meta['H'])
        else:
            is_structured = False

    # 4. Visualization (Heatmaps)
    viz_dir = f"output/{model_name}/visualization"
    os.makedirs(viz_dir, exist_ok=True)
    
    if is_structured:
        D, S, H = max_l + 1, max_s + 1, max_h + 1
        
        # Create Grids: (S, D, H) -> Scale-Major order
        usage_grid = np.zeros((S, D, H))
        perp_grid = np.zeros((S, D, H))
        ortho_grid = np.zeros((S, D, H))
        erank_grid = np.zeros((S, D, H))
        
        for s in stats_list:
            m = s['meta']
            if m:
                # Store as [S, L, H]
                usage_grid[m['S'], m['L'], m['H']] = s['usage']
                perp_grid[m['S'], m['L'], m['H']] = s['perplex']
                ortho_grid[m['S'], m['L'], m['H']] = s['ortho']
                erank_grid[m['S'], m['L'], m['H']] = s['erank']
        
        # --- Visualization: Unrolled Heatmap (S*L vs H) ---
        # Reshape (S, D, H) -> (S*D, H)
        usage_unrolled = usage_grid.reshape(S * D, H)
        perp_unrolled = perp_grid.reshape(S * D, H)
        ortho_unrolled = ortho_grid.reshape(S * D, H)
        erank_unrolled = erank_grid.reshape(S * D, H)
        
        # Create Row Labels: S0.L0, S0.L1...
        row_labels = []
        for s in range(S):
            for d in range(D):
                row_labels.append(f"S{s}.L{d}")
        
        # Increased width for 4 plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, S * D + 4)) 
        
        # 1. Plot Usage
        im1 = ax1.imshow(usage_unrolled, cmap='viridis', vmin=0, vmax=100, aspect='auto')
        plt.colorbar(im1, ax=ax1, label='Usage %')
        ax1.set_title("Codebook Usage (%)")
        ax1.set_xlabel("Head")
        ax1.set_ylabel("Scale . Layer")
        ax1.set_yticks(range(len(row_labels)))
        ax1.set_yticklabels(row_labels)
        ax1.set_xticks(range(H))
        
        # 2. Plot Perplexity
        # Use theoretical max (vq_head_vocab_size)
        im2 = ax2.imshow(perp_unrolled, cmap='magma', vmin=0, vmax=vq_head_vocab_size, aspect='auto')
        plt.colorbar(im2, ax=ax2, label='Perplexity')
        ax2.set_title("Codebook Perplexity")
        ax2.set_xlabel("Head")
        ax2.set_ylabel("Scale . Layer")
        ax2.set_yticks(range(len(row_labels)))
        ax2.set_yticklabels(row_labels)
        ax2.set_xticks(range(H))

        # 3. Plot Orthogonality
        # Use theoretical max cosine similarity (1.0)
        im3 = ax3.imshow(ortho_unrolled, cmap='plasma', vmin=0, vmax=1.0, aspect='auto')
        plt.colorbar(im3, ax=ax3, label='Cos Sim')
        ax3.set_title("Avg Abs Cosine Sim (0=Ortho)")
        ax3.set_xlabel("Head")
        ax3.set_ylabel("Scale . Layer")
        ax3.set_yticks(range(len(row_labels)))
        ax3.set_yticklabels(row_labels)
        ax3.set_xticks(range(H))
        
        # 4. Plot Effective Rank
        im4 = ax4.imshow(erank_unrolled, cmap='cividis', vmin=0, vmax=eff_rank_max, aspect='auto')
        plt.colorbar(im4, ax=ax4, label='Eff. Rank')
        ax4.set_title("Effective Rank")
        ax4.set_xlabel("Head")
        ax4.set_ylabel("Scale . Layer")
        ax4.set_yticks(range(len(row_labels)))
        ax4.set_yticklabels(row_labels)
        ax4.set_xticks(range(H))
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "codebook_comprehensive_matrix.png"))
        print(f"\nComprehensive matrix plots saved to {viz_dir}/codebook_comprehensive_matrix.png")

    # 5. Cross-Scale Analysis (The Big Matrix)
    cross_metrics = compute_cross_scale_metrics(codebooks)
    if cross_metrics:
        n = len(cross_metrics['labels'])
        labels = cross_metrics['labels']
        
        # Create large figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # CKA
        im1 = ax1.imshow(cross_metrics['cka'], cmap='inferno', vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax1, label='CKA')
        ax1.set_title(f"Cross-Codebook CKA ({n}x{n})")
        # Only label every Nth tick if too dense
        step = max(1, n // 30)
        ax1.set_xticks(range(0, n, step))
        ax1.set_yticks(range(0, n, step))
        ax1.set_xticklabels(labels[::step], rotation=90, fontsize=8)
        ax1.set_yticklabels(labels[::step], fontsize=8)
        
        # Overlap
        im2 = ax2.imshow(cross_metrics['overlap'], cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im2, ax=ax2, label='Overlap')
        ax2.set_title(f"Subspace Overlap ({n}x{n})")
        ax2.set_xticks(range(0, n, step))
        ax2.set_yticks(range(0, n, step))
        ax2.set_xticklabels(labels[::step], rotation=90, fontsize=8)
        ax2.set_yticklabels(labels[::step], fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "codebook_cross_correlation.png"))
        print(f"Cross-Correlation plots saved to {viz_dir}/codebook_cross_correlation.png")

if __name__ == "__main__":
    main()
