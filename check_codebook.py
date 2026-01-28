import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.factory import build_model_from_config, build_preprocessing_from_config
from IO.dataset import build_dataset_from_config

class CodebookAnalyzer:
    def __init__(self, model, config, device="cuda"):
        self.model = model
        self.config = config
        self.device = device
        
        self.model_type = config['training_params'].get('model_type', 'NeuroRVQ')
        
        # Check for vectorized architecture (RVQ or FSQ)
        self.is_vectorized = hasattr(self.model, 'rvq') or hasattr(self.model, 'fsq')
        self.is_fsq = (self.model_type == 'RecurrentFSQ')
        
        if self.model_type == 'RecurrentVQ':
            self.m_params = config['model_params']['RecurrentVQ']
            self.n_layers = self.m_params['num_recurrent_steps']
            self.is_shared_codebook = True 
            self.vocab_size = self.m_params['vocab_size']
        elif self.model_type == 'RecurrentFSQ':
            self.m_params = config['model_params']['RecurrentFSQ']
            self.n_layers = self.m_params['num_recurrent_steps']
            self.is_shared_codebook = True
            # FSQ vocab size is product of levels
            levels = self.m_params.get('fsq_levels', [8, 5, 5, 5])
            self.vocab_size = np.prod(levels)
        else:
            self.m_params = config['model_params']['NeuroRVQ']
            self.n_layers = self.m_params['num_codebooks']
            self.is_shared_codebook = False
            self.vocab_size = self.m_params['vocab_size']
            
        self.embed_dim = self.m_params['embed_dim']
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
            # output signature varies:
            # FSQ module: (z_out, indices) -> Index 1
            # RVQ module: (z_out, loss, indices) -> Index 2
            idx_pos = 1 if self.is_fsq else 2
            indices_buffer.append(output[idx_pos].detach())

        if self.is_vectorized:
            if self.is_fsq:
                # FSQ has separate modules per scale in self.model.fsq.fsqs
                target_module = self.model.fsq.fsqs[scale_idx]
            else:
                target_module = self.model.rvq
        else:
            target_module = self.model.rvqs[scale_idx]
            
        handle = target_module.register_forward_hook(hook_fn)
        
        print(f"Collecting usage statistics for Scale {scale_idx}...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, total=max_batches)):
                if i >= max_batches: break
                
                # Unpack batch: (patch, coords, label)
                batch_x, batch_coords, _ = [t.to(self.device) for t in batch]
                
                # Clear buffer before forward pass to ensure clean state
                indices_buffer.clear()
                
                # Forward pass
                # model expects x: (B, N, T), coords: (B, N, 3)
                self.model(batch_x, batch_coords)
                
                if self.is_fsq:
                    # FSQ: Hook fires once per recurrent step. Buffer has [Step1, Step2, ...]
                    # We iterate through the buffer directly.
                    if len(indices_buffer) != self.n_layers:
                        # Safety check: might happen if hook logic is wrong or partial
                        # For now, just take what we have or skip
                        pass
                    
                    for l, idx in enumerate(indices_buffer):
                        if l < self.n_layers:
                            # idx shape: (B*N, 1) or (B*N,)
                            l_idx = idx.flatten()
                            usage_counts[l].put_(l_idx.long(), torch.ones_like(l_idx, dtype=torch.float), accumulate=True)
                    
                    indices_buffer.clear()
                    
                elif indices_buffer:
                    # RVQ: Hook fires once at end. Buffer has [Stacked_Indices]
                    all_inds = indices_buffer.pop() 
                    
                    if self.is_vectorized:
                        # RVQ case: (S, B, N, Steps)
                        scale_inds = all_inds[scale_idx] # (B, N, Steps)
                        for l in range(self.n_layers):
                            l_idx = scale_inds[..., l].flatten()
                            usage_counts[l].put_(l_idx, torch.ones_like(l_idx, dtype=torch.float), accumulate=True)
                    else:
                        # Old NeuroRVQ case
                        idx = all_inds
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
        if self.is_fsq:
            print("Skipping Multi-Scale Codebook Analysis for FSQ (No learned codebooks).")
            return

        if self.is_vectorized:
             n_scales = self.model.rvq.num_scales
        else:
             n_scales = len(self.model.rvqs)
             
        cross_cka = np.zeros((n_scales, n_scales))
        cross_ovl = np.zeros((n_scales, n_scales))
        
        # We compare the FIRST layer of each scale (the primary representation)
        print(f"Comparing {n_scales} scales...")
        scale_codebooks = []
        
        if self.is_vectorized:
            # self.model.rvq.embedding: (S, N_E, D) or (N_E, D) if shared
            if self.model.rvq.embedding.dim() == 2:
                # Shared codebook across all scales
                cb = self.model.rvq.embedding.data
                for s in range(n_scales):
                    scale_codebooks.append(cb)
            else:
                for s in range(n_scales):
                    scale_codebooks.append(self.model.rvq.embedding.data[s])
        else:
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
        
        if self.is_fsq:
            print(f"Skipping structural analysis for FSQ (Implicit Codebook).")
            # Minimal plotting for FSQ
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.bar(range(1, self.n_layers+1), perplexities, color='teal')
            ax.axhline(y=self.vocab_size, color='r', linestyle='--', label='Max (Total Levels)')
            ax.set_title("FSQ Perplexity (Effective Vocab Usage)")
            ax.set_ylabel("Effective Vocab Size")
            ax.set_xlabel("Recurrent Step")
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(save_dir, "codebook_report.png")
            plt.savefig(save_path)
            print(f"Report saved to {save_path}")
            return

        # B. Structure
        unique_codebooks = [] # For atom sim plot
        if self.is_shared_codebook:
            # RecurrentVQ case - single shared codebook (per scale)
            if self.is_vectorized:
                 if self.model.rvq.embedding.dim() == 2:
                     cb = self.model.rvq.embedding.data
                 else:
                     cb = self.model.rvq.embedding.data[scale_idx]
            else:
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
    
    # New structured path: ./output/{model_name}/checkpoints/best_model.pth
    ckpt_path = f'output/{model_name}/checkpoints/best_model.pth'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Support both old direct state_dict and new dict format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Handle shape mismatch for shared codebook transition
        if 'rvq.embedding' in state_dict:
            ckpt_emb = state_dict['rvq.embedding']
            if hasattr(model, 'rvq') and model.rvq.embedding.dim() == 2 and ckpt_emb.dim() == 3:
                print("Detected checkpoint with separate codebooks. Averaging for shared initialization...")
                state_dict['rvq.embedding'] = ckpt_emb.mean(dim=0)
                
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded {ckpt_path}")
    
    # Load Data (Small subset)
    dataset_path = config['dataset_params']['dataset_path']
    with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    config['data_metadata'] = meta['data_metadata']
    
    transform = build_preprocessing_from_config(config)
    dataset = build_dataset_from_config(config, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Analyze
    analyzer = CodebookAnalyzer(model, config, device)
    # New structured path: ./output/{model_name}/visualization
    viz_dir = f"output/{model_name}/visualization"
    analyzer.run_full_analysis(loader, scale_idx=0, save_dir=viz_dir)
    analyzer.run_multiscale_analysis(save_dir=viz_dir)

if __name__ == "__main__":
    main()
