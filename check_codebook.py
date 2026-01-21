import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.factory import build_model_from_config
from IO.dataset import build_dataset_from_config
from model.NeuroRVQ.preprocessing import NeuroRVQProcessing
from model.RecurrentVQ.preprocessing import RecurrentVQProcessing

class CodebookAnalyzer:
    def __init__(self, model, config, device="cuda"):
        self.model = model
        self.config = config
        self.device = device
        
        self.model_type = config['training_params'].get('model_type', 'NeuroRVQ')
        if self.model_type == 'RecurrentVQ':
            self.m_params = config['model_params']['RecurrentVQ']
            self.n_layers = self.m_params['num_recurrent_steps']
            self.is_shared_codebook = True
        else:
            self.m_params = config['model_params']['NeuroRVQ']
            self.n_layers = self.m_params['num_codebooks']
            self.is_shared_codebook = False
            
        self.embed_dim = self.m_params['embed_dim']
        self.vocab_size = self.m_params['vocab_size']
        self.model.eval()

    # --- 1. Structural Analysis Methods ---

    def compute_cka(self, X, Y):
        """Linear CKA: Similarity of the relational 'maps'."""
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        dot_product = torch.norm(torch.matmul(X.t(), Y))**2
        norm_x = torch.norm(torch.matmul(X.t(), X))
        norm_y = torch.norm(torch.matmul(Y.t(), Y))
        return (dot_product / (norm_x * norm_y + 1e-8)).item()

    def compute_subspace_overlap(self, X, Y, threshold=0.95):
        """How much of X is explained by Y's basis."""
        X_c = X - X.mean(dim=0, keepdim=True)
        Y_c = Y - Y.mean(dim=0, keepdim=True)
        U, S, V = torch.pca_lowrank(Y_c, q=min(Y.shape))
        sum_s = torch.cumsum(S**2, dim=0)
        total_s = torch.sum(S**2)
        n_comp = torch.where(sum_s / (total_s + 1e-8) >= threshold)[0]
        n_comp = n_comp[0].item() + 1 if len(n_comp) > 0 else min(Y.shape)
        basis_y = V[:, :n_comp]
        X_proj_coords = torch.matmul(X_c, basis_y)
        overlap = torch.norm(X_proj_coords)**2 / (torch.norm(X_c)**2 + 1e-8)
        return overlap.item()

    def compute_effective_rank(self, X):
        """Participation Ratio: Dimensional expressivity."""
        X_c = X - X.mean(dim=0, keepdim=True)
        _, S, _ = torch.svd(X_c)
        lambdas = S**2
        pr = (torch.sum(lambdas)**2) / (torch.sum(lambdas**2) + 1e-8)
        return pr.item()

    def compute_atom_similarity(self, codebook):
        """Computes Cosine Similarity matrix between atoms (vocab x vocab)."""
        # codebook: (Vocab, Dim)
        # Normalize rows
        norms = torch.norm(codebook, dim=1, keepdim=True)
        normed_cb = codebook / (norms + 1e-8)
        # Cosine Sim = (A . B) / (|A|*|B|) -> (Normed . Normed^T)
        sim_matrix = torch.matmul(normed_cb, normed_cb.t())
        return sim_matrix.cpu().numpy()

    # --- 2. Usage Analysis Methods ---

    def collect_usage_stats(self, data_loader, scale_idx=0, max_batches=50):
        """Inference pass to see which codewords are actually used."""
        usage_counts = [torch.zeros(self.vocab_size, device=self.device) for _ in range(self.n_layers)]
        indices_buffer = []

        def hook_fn(module, input, output):
            indices_buffer.append(output[2].detach())

        handle = self.model.rvqs[scale_idx].register_forward_hook(hook_fn)
        coords = torch.from_numpy(data_loader.dataset.coords).float().to(self.device)

        print(f"Collecting usage statistics for Scale {scale_idx}...")
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(tqdm(data_loader, total=max_batches)):
                if i >= max_batches: break
                patch = batch_x[..., :200].to(self.device)
                self.model(patch, coords)
                if indices_buffer:
                    idx = indices_buffer.pop()
                    for l in range(self.n_layers):
                        l_idx = idx[..., l].flatten()
                        usage_counts[l].put_(l_idx, torch.ones_like(l_idx, dtype=torch.float), accumulate=True)
        
        handle.remove()
        
        perplexities = []
        for counts in usage_counts:
            probs = counts / (counts.sum() + 1e-8)
            probs = probs[probs > 0]
            entropy = -torch.sum(probs * torch.log(probs))
            perplexities.append(torch.exp(entropy).item())
        
        return perplexities, usage_counts

    # --- 3. Multi-Scale Comparison ---

    def run_multiscale_analysis(self, save_dir="output/visualization"):
        n_scales = len(self.model.rvqs)
        cross_cka = np.zeros((n_scales, n_scales))
        cross_ovl = np.zeros((n_scales, n_scales))
        
        # We compare the FIRST layer of each scale (the primary representation)
        print(f"Comparing {n_scales} scales...")
        scale_codebooks = []
        for s in range(n_scales):
            if hasattr(self.model.rvqs[s], 'layers'):
                scale_codebooks.append(self.model.rvqs[s].layers[0].embedding.weight.data)
            else:
                scale_codebooks.append(self.model.rvqs[s].vq.embedding.weight.data)
        
        for i in range(n_scales):
            for j in range(n_scales):
                cross_cka[i, j] = self.compute_cka(scale_codebooks[i], scale_codebooks[j])
                cross_ovl[i, j] = self.compute_subspace_overlap(scale_codebooks[i], scale_codebooks[j])
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        im1 = ax1.imshow(cross_cka, cmap='plasma', vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax1, label='CKA')
        ax1.set_title("Cross-Scale CKA\n(Similarity of Logic between Branches)")
        
        im2 = ax2.imshow(cross_ovl, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im2, ax=ax2, label='Overlap')
        ax2.set_title("Cross-Scale Subspace Overlap\n(Shared vs. Unique Feature Spaces)")
        
        for ax in [ax1, ax2]:
            ax.set_xticks(range(n_scales))
            ax.set_yticks(range(n_scales))
            ax.set_xticklabels([f"Scale {i}" for i in range(n_scales)])
            ax.set_yticklabels([f"Scale {i}" for i in range(n_scales)])
            
        plt.tight_layout()
        save_path = os.path.join(save_dir, "multiscale_comparison.png")
        plt.savefig(save_path)
        print(f"Multi-scale analysis saved to {save_path}")

    # --- 4. Full Report Generation ---

    def plot_atom_similarities(self, codebooks, save_dir):
        """Generates a grid of Atom-Atom Similarity matrices for all codebooks."""
        n = len(codebooks)
        # Determine grid size dynamically to avoid empty subplots
        cols = min(n, 4)
        rows = int(np.ceil(n / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if n == 1: axes = [axes] # Handle single plot case (returns Axes object, not array)
        axes = np.array(axes).flatten()
        
        print("Computing atom-wise similarities...")
        for i, cb in enumerate(codebooks):
            sim = self.compute_atom_similarity(cb)
            # Use interpolation='nearest' to prevent blurring of diagonal values
            im = axes[i].imshow(sim, cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest')
            axes[i].set_title(f"Layer {i+1} Self-Sim")
            axes[i].axis('off')
            
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        # Add common colorbar only if there's space/need, or per plot. 
        # For simplicity in grid, adding to figure edge is fine.
        cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
        fig.colorbar(im, cax=cb_ax, label='Cosine Similarity')
        
        save_path = os.path.join(save_dir, "atom_similarity.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Atom similarity report saved to {save_path}")

    # --- 4. Full Report Generation ---

    def run_full_analysis(self, data_loader, scale_idx=0, save_dir="output/visualization"):
        os.makedirs(save_dir, exist_ok=True)
        
        # A. Usage
        perplexities, _ = self.collect_usage_stats(data_loader, scale_idx)
        
        # B. Structure
        unique_codebooks = [] # For atom sim plot
        if self.is_shared_codebook:
            # RecurrentVQ case - single shared codebook
            cb = self.model.rvqs[scale_idx].vq.embedding.weight.data
            codebooks = [cb]
            unique_codebooks = [cb]
        else:
            # NeuroRVQ case - list of layers
            if hasattr(self.model.rvqs[scale_idx], 'layers'):
                 codebooks = [self.model.rvqs[scale_idx].layers[i].embedding.weight.data for i in range(self.n_layers)]
                 unique_codebooks = codebooks
            else:
                 # Fallback
                 cb = self.model.rvqs[scale_idx].vq.embedding.weight.data
                 codebooks = [cb]
                 unique_codebooks = [cb]

        ranks = [self.compute_effective_rank(cb) for cb in codebooks]
        
        # NEW: Generate Separate Atom Similarity Report
        self.plot_atom_similarities(unique_codebooks, save_dir)

        if not self.is_shared_codebook:
            cka_mtx = np.zeros((self.n_layers, self.n_layers))
            ovl_mtx = np.zeros((self.n_layers, self.n_layers))

            print("Computing structural metrics...")
            for i in range(self.n_layers):
                for j in range(self.n_layers):
                    cka_mtx[i, j] = self.compute_cka(codebooks[i], codebooks[j])
                    ovl_mtx[i, j] = self.compute_subspace_overlap(codebooks[i], codebooks[j])

        # C. Visualization (Summary Report)
        if self.is_shared_codebook:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # Revert to 1x2 for summary
            ax_perp, ax_rank = axes[0], axes[1]
            plot_ranks = ranks * self.n_layers 
        else:
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            ax_perp, ax_rank = axes[0,0], axes[0,1]
            plot_ranks = ranks

        # 1. Perplexity
        ax_perp.bar(range(1, self.n_layers+1), perplexities, color='teal')
        ax_perp.axhline(y=self.vocab_size, color='r', linestyle='--', label='Max')
        ax_perp.set_title("Perplexity (Usage per Step)")
        ax_perp.set_ylabel("Effective Vocab Size")
        
        # 2. Effective Rank
        ax_rank.bar(range(1, len(plot_ranks)+1), plot_ranks, color='salmon')
        ax_rank.axhline(y=self.embed_dim, color='gray', linestyle='--', label='Max Dim')
        ax_rank.set_title("Effective Rank (Expressivity)")
        ax_rank.set_ylabel("Participating Dimensions")
        
        if self.is_shared_codebook:
             ax_rank.set_title("Effective Rank (Shared Codebook)")
        else:
            # 3. CKA Matrix
            im3 = axes[1,0].imshow(cka_mtx, cmap='magma', vmin=0, vmax=1)
            plt.colorbar(im3, ax=axes[1,0], label='CKA')
            axes[1,0].set_title("Relational Similarity (CKA)")

            # 4. Overlap Matrix
            im4 = axes[1,1].imshow(ovl_mtx, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(im4, ax=axes[1,1], label='Overlap')
            axes[1,1].set_title("Subspace Overlap")

            for ax in axes[1]:
                ax.set_xticks(range(self.n_layers))
                ax.set_yticks(range(self.n_layers))
                ax.set_xticklabels(range(1, self.n_layers+1))
                ax.set_yticklabels(range(1, self.n_layers+1))

        plt.tight_layout()
        save_path = os.path.join(save_dir, "codebook_report.png")
        plt.savefig(save_path)
        print(f"\nReport saved to {save_path}")

        # Console Summary
        print("\n" + "="*50)
        print("CODEBOOK ANALYSIS SUMMARY")
        print("="*50)
        
        if self.is_shared_codebook:
             print(f"Shared Codebook Rank: {ranks[0]:.1f}")
             print(f"{'Step':<8} | {'Perplexity':<12}")
             print("-" * 25)
             for i in range(self.n_layers):
                print(f"S{i+1:<7} | {perplexities[i]:<12.1f}")
        else:
            print(f"{'Layer':<8} | {'Perplexity':<12} | {'Eff. Rank':<10}")
            print("-" * 35)
            for i in range(self.n_layers):
                print(f"L{i+1:<7} | {perplexities[i]:<12.1f} | {ranks[i]:<10.1f}")

            def print_mtx(mtx, name):
                print(f"\n{name}:")
                header = "      " + "".join([f"L{i+1:<5}" for i in range(self.n_layers)])
                print(header)
                for i in range(self.n_layers):
                    row = f"L{i+1:<4} | " + "".join([f"{mtx[i,j]:.3f} " for j in range(self.n_layers)])
                    print(row)

            print_mtx(cka_mtx, "Linear CKA Matrix (Relational Similarity)")
            print_mtx(ovl_mtx, "Subspace Overlap Matrix (Directional Alignment)")
            
        print("\n" + "="*50)

def main():
    config_path = 'config/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    train_params = config['training_params']
    model_name = train_params.get('model_name', 'neurorvq_v0')
    model_type = train_params.get('model_type', 'NeuroRVQ')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Model
    model = build_model_from_config(config).to(device)
    ckpt_path = f'output/checkpoints/tokenizer/{model_name}/tokenizer_best.pth'
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        print(f"Loaded {ckpt_path}")
    
    # Load Data (Small subset)
    dataset_path = config['dataset_params']['dataset_path']
    with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    config['data_metadata'] = meta['data_metadata']
    
    if model_type == 'RecurrentVQ':
        transform = RecurrentVQProcessing(meta['data_metadata']['Sample_Frequency'], 200)
    else:
        transform = NeuroRVQProcessing(meta['data_metadata']['Sample_Frequency'], 200)
        
    dataset = build_dataset_from_config(config, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Analyze
    analyzer = CodebookAnalyzer(model, config, device)
    save_path = f"output/visualization/{model_name}"
    analyzer.run_full_analysis(loader, scale_idx=0, save_dir=save_path)
    analyzer.run_multiscale_analysis(save_dir=save_path)

if __name__ == "__main__":
    main()
