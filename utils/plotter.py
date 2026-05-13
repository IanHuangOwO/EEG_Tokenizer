
import os
import matplotlib.pyplot as plt
import csv

class Plotter:
    """A helper class to track and plot losses during training."""
    def __init__(self, output_dir='output/visualization/training_curves'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.history = {
            'train': {},
            'val': {}
        }

    def update(self, train_metrics, val_metrics=None):
        """
        Adds a new set of loss values for the current epoch.
        metrics should be a dictionary: {'loss': ..., 'vq': ..., ...}
        """
        # Update train history
        for k, v in train_metrics.items():
            if k not in self.history['train']:
                self.history['train'][k] = []
            self.history['train'][k].append(v)
        
        # Update val history
        if val_metrics:
            for k, v in val_metrics.items():
                if k not in self.history['val']:
                    self.history['val'][k] = []
                self.history['val'][k].append(v)

    def plot(self, filename='training_curves.png'):
        """Saves a plot of the training curves (losses)."""
        if 'loss' not in self.history['train'] or not self.history['train']['loss']:
            return
            
        epochs = range(1, len(self.history['train']['loss']) + 1)
        has_val = 'loss' in self.history['val'] and len(self.history['val']['loss']) > 0

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 16), sharex=True)
        # Plot Main Losses
        ax1.plot(epochs, self.history['train']['loss'], 'b-', label='Train Total')
        if has_val:
            ax1.plot(epochs, self.history['val']['loss'], 'b--', alpha=0.7, label='Val Total')

        ax1.set_title('Global Losses')
        ax1.set_ylabel('Loss')
        ax1.legend(); ax1.grid(True)
        
        # Plot Components (Real/Imag/Sub)
        if 'real' in self.history['train']:
            ax2.plot(epochs, self.history['train']['real'], color='orange', linestyle='-', label='Train Real')
        if 'imag' in self.history['train']:
            ax2.plot(epochs, self.history['train']['imag'], color='purple', linestyle='-', label='Train Imag')
        if 'sub' in self.history['train']:
            ax2.plot(epochs, self.history['train']['sub'], color='red', linestyle='-', label='Train Sub')
        
        if has_val:
             if 'real' in self.history['val']:
                ax2.plot(epochs, self.history['val']['real'], color='orange', linestyle='--', alpha=0.7, label='Val Real')
             if 'imag' in self.history['val']:
                ax2.plot(epochs, self.history['val']['imag'], color='purple', linestyle='--', alpha=0.7, label='Val Imag')
             if 'sub' in self.history['val']:
                ax2.plot(epochs, self.history['val']['sub'], color='red', linestyle='--', alpha=0.7, label='Val Sub')
             
        ax2.set_title('Loss Components')
        ax2.set_ylabel('Loss')
        ax2.legend(); ax2.grid(True)

        # Plot Temporal
        if 'temp' in self.history['train']:
            ax3.plot(epochs, self.history['train']['temp'], 'm-', label='Train Temp (MSE)')
        if has_val and 'temp' in self.history['val']:
            ax3.plot(epochs, self.history['val']['temp'], 'm--', alpha=0.7, label='Val Temp (MSE)')
            
        ax3.set_title('Temporal Loss (Global MSE)')
        ax3.set_xlabel('Epoch'); ax3.set_ylabel('Loss')
        ax3.legend(); ax3.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close(fig)

    def plot_metrics(self, filename='training_metrics.png'):
        """Saves a plot of specific training metrics organized by layer/head."""
        
        # 1. Identify all unique AttnVQ heads (layers) present in the metrics
        # Keys look like: subspace_ortho_head_0, codebook_perplexity_head_1, etc.
        all_keys = self.history['train'].keys()
        head_indices = sorted(list(set([
            int(parts[1]) for k in all_keys
            if '_head_' in k and (parts := k.rsplit('_head_', 1)) and parts[1].isdigit()
        ])))
        
        if not head_indices:
            return # Fallback or handle pretrain if needed

        n_heads = len(head_indices)

        # 2. Create Grid: Rows = Heads, Columns = (Codebook+Matrix, Subspace, Pooler A, Pooler B)
        fig, axes = plt.subplots(n_heads, 4, figsize=(28, 5 * n_heads))
        if n_heads == 1:
            axes = [axes]

        for i, h_idx in enumerate(head_indices):
            suffix = f"_head_{h_idx}"
            ax_cb, ax_sub, ax_pool_a, ax_pool_b = axes[i]

            # --- Column 1: Codebook Health + Matrix Health ---
            p_key  = f"codebook_perplexity{suffix}"
            s_key  = f"codebook_sharpness{suffix}"
            sv_key = f"codebook_avg_singular_value{suffix}"
            cd_key = f"codebook_condition_number{suffix}"
            if p_key in self.history['train']:
                ep = range(1, len(self.history['train'][p_key]) + 1)
                ax_cb.plot(ep, self.history['train'][p_key], 'g-',  label='Perplexity')
                ax_cb.plot(ep, self.history['train'][sv_key], 'b-', label='Avg SV', alpha=0.7)
                ax_cb_twin = ax_cb.twinx()
                ax_cb_twin.plot(ep, self.history['train'][s_key],  'm-',  label='Sharpness', alpha=0.8)
                ax_cb_twin.plot(ep, self.history['train'][cd_key], 'r--', label='Cond #',    alpha=0.6)
                ax_cb_twin.set_yscale('log')
                ax_cb.set_ylabel('Perplexity / Avg SV')
                ax_cb_twin.set_ylabel('Sharpness / Cond #')
                ax_cb.set_title(f'Layer {h_idx}: Codebook + Matrix Health')
                lines,  labels  = ax_cb.get_legend_handles_labels()
                lines2, labels2 = ax_cb_twin.get_legend_handles_labels()
                ax_cb.legend(lines + lines2, labels + labels2, loc='upper left', fontsize='x-small')
                ax_cb.grid(True)

            # --- Column 2: Subspace Diversity ---
            o_key  = f"codebook_total_orthogonality{suffix}"
            c_key  = f"codebook_head_orthogonality{suffix}"
            ar_key = f"codebook_active_rank{suffix}"
            if o_key in self.history['train']:
                ep = range(1, len(self.history['train'][o_key]) + 1)
                ax_sub.plot(ep, self.history['train'][o_key], 'b-', label='Total Ortho')
                ax_sub_twin = ax_sub.twinx()
                ax_sub_twin.plot(ep, self.history['train'][c_key], 'r-', label='Head Ortho', alpha=0.7)
                if ar_key in self.history['train']:
                    ar_ep = range(1, len(self.history['train'][ar_key]) + 1)
                    ax_sub_twin.plot(ar_ep, self.history['train'][ar_key], 'g--', label='Active Rank', alpha=0.7)
                ax_sub.set_ylabel('Total Ortho'); ax_sub_twin.set_ylabel('Head Ortho / Active Rank')
                ax_sub.set_yscale('log')
                ax_sub.set_title(f'Layer {h_idx}: Subspace Diversity')
                lines,  labels  = ax_sub.get_legend_handles_labels()
                lines2, labels2 = ax_sub_twin.get_legend_handles_labels()
                ax_sub.legend(lines + lines2, labels + labels2, loc='upper right', fontsize='x-small')
                ax_sub.grid(True, which="both", ls="-", alpha=0.3)

            # --- Column 3: Pooler — Spectral Gap + Eff Rank (val only) ---
            gap_key  = f"pool_spectral_gap{suffix}"
            util_key = f"pool_filter_eff_rank{suffix}"
            estd_key = f"pool_graph_edge_std{suffix}"
            cont_key = f"pool_filter_contrast{suffix}"
            src = self.history['val']
            if gap_key in src:
                ep = range(1, len(src[gap_key]) + 1)
                ax_pool_a.plot(ep, src[gap_key],  color='steelblue',  linestyle='-', label='Spectral Gap')
                ax_pool_a_twin = ax_pool_a.twinx()
                ax_pool_a_twin.plot(ep, src[util_key], color='darkorange', linestyle='-', label='Eff Rank')
                ax_pool_a_twin.set_ylim(bottom=0)
                ax_pool_a.set_ylabel('Spectral Gap', color='steelblue')
                ax_pool_a_twin.set_ylabel('Eff Rank', color='darkorange')
                ax_pool_a.set_title(f'Layer {h_idx}: Pooler — Gap & Rank (val)')
                lines,  labels  = ax_pool_a.get_legend_handles_labels()
                lines2, labels2 = ax_pool_a_twin.get_legend_handles_labels()
                ax_pool_a.legend(lines + lines2, labels + labels2, loc='upper left', fontsize='x-small')
                ax_pool_a.grid(True)

                # --- Column 4: Pooler — Edge Std + Contrast (val only) ---
                ax_pool_b.plot(ep, src[estd_key], color='steelblue',  linestyle='-', label='Edge Std')
                ax_pool_b_twin = ax_pool_b.twinx()
                ax_pool_b_twin.plot(ep, src[cont_key], color='darkorange', linestyle='-', label='Contrast')
                ax_pool_b_twin.set_ylim(bottom=0)
                ax_pool_b.set_ylabel('Edge Std', color='steelblue')
                ax_pool_b_twin.set_ylabel('Contrast', color='darkorange')
                ax_pool_b.set_title(f'Layer {h_idx}: Pooler — Std & Contrast (val)')
                lines,  labels  = ax_pool_b.get_legend_handles_labels()
                lines2, labels2 = ax_pool_b_twin.get_legend_handles_labels()
                ax_pool_b.legend(lines + lines2, labels + labels2, loc='upper left', fontsize='x-small')
                ax_pool_b.grid(True)
            else:
                ax_pool_a.set_title(f'Layer {h_idx}: Pooler (no data)'); ax_pool_a.axis('off')
                ax_pool_b.axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close(fig)
