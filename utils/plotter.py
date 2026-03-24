
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

    def save_csv(self):
        """Saves the entire training history to a CSV file."""
        csv_dir = os.path.join(self.output_dir, 'csv')
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, 'training_history.csv')
        
        # Collect all unique keys
        all_keys = set(self.history['train'].keys()) | set(self.history['val'].keys())
        # Sort keys: epoch first, then others
        sorted_keys = sorted(list(all_keys))
        
        n_epochs = len(next(iter(self.history['train'].values()))) if self.history['train'] else 0
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            header = ['epoch']
            for k in sorted_keys:
                header.append(f'train_{k}')
                header.append(f'val_{k}')
            writer.writerow(header)
            
            # Rows
            for i in range(n_epochs):
                row = [i + 1]
                for k in sorted_keys:
                    row.append(self.history['train'].get(k, [None])[i] if i < len(self.history['train'].get(k, [])) else None)
                    row.append(self.history['val'].get(k, [None])[i] if i < len(self.history['val'].get(k, [])) else None)
                writer.writerow(row)
        # print(f"Training history saved to {csv_path}")

    def plot(self, filename='training_curves.png'):
        """Saves a plot of the training curves (losses)."""
        if 'loss' not in self.history['train'] or not self.history['train']['loss']:
            return
            
        epochs = range(1, len(self.history['train']['loss']) + 1)
        has_val = 'loss' in self.history['val'] and len(self.history['val']['loss']) > 0

        # Determine if we are in pretrain mode
        is_pretrain = 'unmasked_kl' in self.history['train'] or 'distill_masked' in self.history['train']

        if is_pretrain:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            # Support both old and new pretrain keys
            m_key = 'distill_masked' if 'distill_masked' in self.history['train'] else 'loss'
            v_key = 'distill_visible' if 'distill_visible' in self.history['train'] else 'unmasked_kl'
            
            ax1.plot(epochs, self.history['train'][m_key], 'b-', label='Train Masked (M)')
            if v_key in self.history['train']:
                ax1.plot(epochs, self.history['train'][v_key], 'g-', label='Train Visible (V)')
            
            if has_val:
                ax1.plot(epochs, self.history['val'][m_key], 'b--', alpha=0.7, label='Val Masked (M)')
                if v_key in self.history['val']:
                    ax1.plot(epochs, self.history['val'][v_key], 'g--', alpha=0.7, label='Val Visible (V)')
            
            ax1.set_title('Pretraining Distillation Loss')
            ax1.set_ylabel('Loss (KL Divergence)')
            ax1.set_xlabel('Epoch')
            ax1.legend()
            ax1.grid(True)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 16), sharex=True)
            # Plot Main Losses
            ax1.plot(epochs, self.history['train']['loss'], 'b-', label='Train Total')
            
            if has_val:
                ax1.plot(epochs, self.history['val']['loss'], 'b--', alpha=0.7, label='Val Total')

            ax1.set_title('Global Losses')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot Components (Amp/Phase/Sub)
            if 'amp' in self.history['train']:
                ax2.plot(epochs, self.history['train']['amp'], color='orange', linestyle='-', label='Train Amp')
            if 'phase' in self.history['train']:
                ax2.plot(epochs, self.history['train']['phase'], color='purple', linestyle='-', label='Train Phase')
            if 'sub' in self.history['train']:
                ax2.plot(epochs, self.history['train']['sub'], color='red', linestyle='-', label='Train Sub')
            
            if has_val:
                 if 'amp' in self.history['val']:
                    ax2.plot(epochs, self.history['val']['amp'], color='orange', linestyle='--', alpha=0.7, label='Val Amp')
                 if 'phase' in self.history['val']:
                    ax2.plot(epochs, self.history['val']['phase'], color='purple', linestyle='--', alpha=0.7, label='Val Phase')
                 if 'sub' in self.history['val']:
                    ax2.plot(epochs, self.history['val']['sub'], color='red', linestyle='--', alpha=0.7, label='Val Sub')
                 
            ax2.set_title('Loss Components')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)

            # Plot Temporal
            if 'temp' in self.history['train']:
                ax3.plot(epochs, self.history['train']['temp'], 'm-', label='Train Temp (MSE)')
            if has_val and 'temp' in self.history['val']:
                ax3.plot(epochs, self.history['val']['temp'], 'm--', alpha=0.7, label='Val Temp (MSE)')
                
            ax3.set_title('Temporal Loss (Time Domain)')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.legend()
            ax3.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close(fig)

    def plot_metrics(self, filename='training_metrics.png'):
        """Saves a plot of specific training metrics."""
        self.save_csv()
        # If we don't have temp_mse but we have unmasked_kl or distill_masked, we are in pretrain mode
        is_pretrain = 'unmasked_kl' in self.history['train'] or 'distill_masked' in self.history['train']
        
        if is_pretrain:
            # For pretrain, we plot per-head KL if available
            kl_m_keys = sorted([k for k in self.history['train'].keys() if k.startswith('kl_masked_h')],
                             key=lambda x: int(x.split('_h')[1]))
            kl_v_keys = sorted([k for k in self.history['train'].keys() if k.startswith('kl_visible_h')],
                             key=lambda x: int(x.split('_h')[1]))
            
            if not kl_m_keys:
                # Fallback to total KL if separate keys not found
                kl_keys = sorted([k for k in self.history['train'].keys() if k.startswith('kl_h')],
                               key=lambda x: int(x.split('_h')[1]))
                if not kl_keys: return
                
                epochs = range(1, len(self.history['train'][kl_keys[0]]) + 1)
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                for k in kl_keys:
                    label = k.replace('kl_h', 'Head ')
                    ax.plot(epochs, self.history['train'][k], label=label)
                ax.set_title('Per-Head Distillation KL Divergence')
                ax.set_ylabel('KL Div'); ax.set_xlabel('Epoch'); ax.legend(loc='upper right', fontsize='x-small', ncol=4); ax.grid(True)
            else:
                epochs = range(1, len(self.history['train'][kl_m_keys[0]]) + 1)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
                
                for k in kl_m_keys:
                    label = k.replace('kl_masked_h', 'Head ')
                    ax1.plot(epochs, self.history['train'][k], label=label)
                ax1.set_title('Masked Patches: Per-Head KL Divergence')
                ax1.set_ylabel('KL Div'); ax1.legend(loc='upper right', fontsize='x-small', ncol=4); ax1.grid(True)
                
                for k in kl_v_keys:
                    label = k.replace('kl_visible_h', 'Head ')
                    ax2.plot(epochs, self.history['train'][k], label=label)
                ax2.set_title('Visible Patches: Per-Head KL Divergence')
                ax2.set_ylabel('KL Div'); ax2.set_xlabel('Epoch'); ax2.legend(loc='upper right', fontsize='x-small', ncol=4); ax2.grid(True)
            
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path)
            plt.close(fig)
            return
            
        if 'temp_mse' not in self.history['train'] or not self.history['train']['temp_mse']:
            return
            
        epochs = range(1, len(self.history['train']['temp_mse']) + 1)
        has_val = 'temp_mse' in self.history['val'] and len(self.history['val']['temp_mse']) > 0

        # Identify scales and heads
        head_keys = [k for k in self.history['train'].keys() if k.startswith('head_weight_')]
        fusion_keys = sorted([k for k in self.history['train'].keys() if k.startswith('fusion_weight_s')],
                            key=lambda x: int(x.split('_s')[1]))
        
        # Determine how to group heads
        # If we have head_weight_s0_h0, we group by scale
        # If we have head_weight_h0, we have one group
        s_head_keys = [k for k in head_keys if '_s' in k]
        if s_head_keys:
            scales = sorted(list(set([int(k.split('_s')[1].split('_h')[0]) for k in s_head_keys])))
            n_scales = len(scales)
            groups = []
            for s in scales:
                heads_in_scale = sorted([k for k in s_head_keys if f'head_weight_s{s}_h' in k], 
                                        key=lambda x: int(x.split('_h')[1]))
                groups.append((f'Heads (Scale {s})', heads_in_scale))
        else:
            h_only_keys = sorted([k for k in head_keys if '_h' in k], key=lambda x: int(x.split('_h')[1]))
            groups = [('Head Weights', h_only_keys)] if h_only_keys else []
            
        # Add fusion weights as a group if they exist
        if fusion_keys:
            groups.insert(0, ('Scale Fusion Weights', fusion_keys))

        # 2x3 Grid Layout
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        ax_mse = axes[0]
        ax_cb = axes[1]
        ax_sub = axes[2]
        ax_mat = axes[3]
        gating_axes = axes[4:]

        # 1. Temporal MSE
        ax_mse.plot(epochs, self.history['train']['temp_mse'], 'b-', label='Train MSE')
        if has_val:
            ax_mse.plot(epochs, self.history['val']['temp_mse'], 'b--', alpha=0.7, label='Val MSE')
            
        ax_mse.set_title('Reconstruction Quality (MSE)')
        ax_mse.set_ylabel('MSE (Log Scale)')
        ax_mse.set_yscale('log')
        ax_mse.legend()
        ax_mse.grid(True, which="both", ls="-", alpha=0.5)

        # 2. Codebook Health (Perplexity and Sharpness)
        if 'codebook_perplexity' in self.history['train']:
            ax_cb.plot(epochs, self.history['train']['codebook_perplexity'], 'g-', label='Perplexity (Usage)')
            ax_cb_twin = ax_cb.twinx()
            ax_cb_twin.plot(epochs, self.history['train']['codebook_sharpness'], 'm-', label='Sharpness (Max Prob)')
            
            ax_cb.set_title('Codebook Health (Diversity vs Sharpness)')
            ax_cb.set_ylabel('Perplexity (Unique Codes)')
            ax_cb_twin.set_ylabel('Sharpness (0 to 1)')
            
            # Combine legends
            lines, labels = ax_cb.get_legend_handles_labels()
            lines2, labels2 = ax_cb_twin.get_legend_handles_labels()
            ax_cb.legend(lines + lines2, labels + labels2, loc='upper left')
            ax_cb.grid(True)

        # 3. Subspace Health (Symmetry & Diversity)
        if 'subspace_loss' in self.history['train']:
            ax_sub.plot(epochs, self.history['train']['subspace_loss'], 'k-', label='Total Subspace Loss', linewidth=2)
            ax_sub.plot(epochs, self.history['train']['subspace_symmetry_err'], 'r--', alpha=0.7, label='Symmetry Error')
            ax_sub.plot(epochs, self.history['train']['subspace_cross_head_corr'], 'b--', alpha=0.7, label='Cross-Head Corr')
            
            ax_sub.set_title('Subspace Health (Joint Gram Matrix)')
            ax_sub.set_ylabel('Loss/Error')
            ax_sub.set_yscale('log')
            ax_sub.legend(loc='upper right', fontsize='x-small')
            ax_sub.grid(True, which="both", ls="-", alpha=0.5)

        # 4. Matrix Health (Singular Values & Condition Number)
        if 'A_sing_val_avg' in self.history['train']:
            ax_mat.plot(epochs, self.history['train']['A_sing_val_avg'], 'b-', label='A Avg SV')
            ax_mat.plot(epochs, self.history['train']['B_sing_val_avg'], 'g-', label='B Avg SV')
            ax_mat_twin = ax_mat.twinx()
            ax_mat_twin.plot(epochs, self.history['train']['A_cond'], 'b--', alpha=0.5, label='A Cond #')
            ax_mat_twin.plot(epochs, self.history['train']['B_cond'], 'g--', alpha=0.5, label='B Cond #')
            
            ax_mat.set_title('Matrix Health (Singular Values & Rank)')
            ax_mat.set_ylabel('Avg Singular Value')
            ax_mat_twin.set_ylabel('Condition Number (Log)')
            ax_mat_twin.set_yscale('log')
            
            lines, labels = ax_mat.get_legend_handles_labels()
            lines2, labels2 = ax_mat_twin.get_legend_handles_labels()
            ax_mat.legend(lines + lines2, labels + labels2, loc='upper left', fontsize='x-small')
            ax_mat.grid(True)

        # 5. Gating / Weight Mechanisms (Up to 2 plots)
        for i in range(2): # Fill exactly 2 slots for gating (or empty)
            ax = gating_axes[i]
            if i < len(groups):
                title, keys = groups[i]
                for k in keys:
                    label = k.replace('head_weight_h', 'H').replace('head_weight_', 'H').replace('fusion_weight_s', 'S')
                    ax.plot(epochs, self.history['train'][k], label=label)
                
                ax.set_title(title)
                ax.set_ylabel('Weight Value')
                ax.set_xlabel('Epoch')
                ax.legend(loc='upper left', fontsize='x-small', ncol=min(len(keys), 4))
                ax.grid(True)
            else:
                ax.axis('off') # Hide unused slots
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close(fig)
