
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
            
            ax1.set_title('Pretraining Distillation Loss (CE)')
            ax1.set_ylabel('Loss (Cross-Entropy)')
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
            
            # Reconstruction MSE plotting for pretrain
            has_mse_train = 'mse' in self.history['train'] and self.history['train']['mse']
            has_mse_val = 'mse' in self.history['val'] and self.history['val']['mse']
            has_mse = has_mse_train or has_mse_val
            has_acc = 'acc' in self.history['train'] and self.history['train']['acc']
            
            n_rows = 0
            if has_mse: n_rows += 1
            if has_acc: n_rows += 1
            if kl_m_keys: n_rows += 2
            elif not kl_m_keys and any(k.startswith('kl_h') for k in self.history['train'].keys()): n_rows += 1
            
            if n_rows == 0: return

            fig, axes = plt.subplots(n_rows, 1, figsize=(12, 6 * n_rows))
            if n_rows == 1: axes = [axes]
            curr_ax = 0

            # 1. Plot Accuracy & F1 if available
            if has_acc:
                ax = axes[curr_ax]
                epochs = range(1, len(self.history['train']['acc']) + 1)
                
                # Plot Total
                ax.plot(epochs, self.history['train']['acc'], 'k-', label='Train Total', alpha=0.3)
                
                # Plot Masked vs Visible
                if 'acc_m' in self.history['train']:
                    ax.plot(epochs, self.history['train']['acc_m'], 'r-', label='Train Masked')
                if 'acc_v' in self.history['train']:
                    ax.plot(epochs, self.history['train']['acc_v'], 'g-', label='Train Visible')
                
                if 'acc_m' in self.history['val']:
                    ax.plot(epochs, self.history['val']['acc_m'], 'r--', alpha=0.7, label='Val Masked')
                if 'acc_v' in self.history['val']:
                    ax.plot(epochs, self.history['val']['acc_v'], 'g--', alpha=0.7, label='Val Visible')
                
                ax.set_title('Classification Performance (Subspace Indices)')
                ax.set_ylabel('Accuracy (0-1)'); ax.legend(ncol=3, fontsize='small'); ax.grid(True)
                curr_ax += 1

            # 2. Plot MSE if available
            if has_mse:
                ax = axes[curr_ax]
                epochs = range(1, len(self.history['val']['mse']) + 1) if has_mse_val else range(1, len(self.history['train']['mse']) + 1)
                
                # Check for split MSE (using keys from train_pretrain.py)
                has_split_mse = 'mse_m' in self.history['val']
                
                if has_split_mse:
                    ax.plot(epochs, self.history['val']['mse'], 'k-', label='Val Total', linewidth=2, alpha=0.5)
                    ax.plot(epochs, self.history['val']['mse_m'], 'r--', label='Val Masked')
                    ax.plot(epochs, self.history['val']['mse_v'], 'g--', label='Val Visible')
                else:
                    if has_mse_train:
                        ax.plot(epochs, self.history['train']['mse'], 'b-', label='Train MSE')
                    if has_mse_val:
                        ax.plot(epochs, self.history['val']['mse'], 'b--', alpha=0.7, label='Val MSE')
                
                ax.set_title('Reconstruction Quality (MSE)')
                ax.set_ylabel('MSE'); ax.legend(ncol=3, fontsize='small'); ax.grid(True)
                curr_ax += 1

            if not kl_m_keys:
                # Fallback to total KL if separate keys not found
                kl_keys = sorted([k for k in self.history['train'].keys() if k.startswith('kl_h')],
                               key=lambda x: int(x.split('_h')[1]))
                if kl_keys:
                    ax = axes[curr_ax]
                    epochs = range(1, len(self.history['train'][kl_keys[0]]) + 1)
                    for k in kl_keys:
                        label = k.replace('kl_h', 'Head ')
                        ax.plot(epochs, self.history['train'][k], label=label)
                    ax.set_title('Per-Head Distillation Loss (CE)')
                    ax.set_ylabel('Loss (CE)'); ax.set_xlabel('Epoch'); ax.legend(loc='upper right', fontsize='x-small', ncol=4); ax.grid(True)
            else:
                epochs = range(1, len(self.history['train'][kl_m_keys[0]]) + 1)
                
                ax1 = axes[curr_ax]
                for k in kl_m_keys:
                    label = k.replace('kl_masked_h', 'Head ')
                    ax1.plot(epochs, self.history['train'][k], label=label)
                ax1.set_title('Masked Patches: Per-Head Distillation Loss (CE)')
                ax1.set_ylabel('Loss (CE)'); ax1.legend(loc='upper right', fontsize='x-small', ncol=4); ax1.grid(True)
                curr_ax += 1
                
                ax2 = axes[curr_ax]
                for k in kl_v_keys:
                    label = k.replace('kl_visible_h', 'Head ')
                    ax2.plot(epochs, self.history['train'][k], label=label)
                ax2.set_title('Visible Patches: Per-Head Distillation Loss (CE)')
                ax2.set_ylabel('Loss (CE)'); ax2.set_xlabel('Epoch'); ax2.legend(loc='upper right', fontsize='x-small', ncol=4); ax2.grid(True)
            
            plt.tight_layout()
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

        # 3. Subspace Health (Inter / Intra / Ortho)
        if 'subspace_ortho' in self.history['train']:
            ax_sub.plot(epochs, self.history['train']['subspace_ortho'], 'b-', label='Ortho Loss (Gram)')
            if 'head_cross_corr' in self.history['train']:
                ax_sub_twin = ax_sub.twinx()
                ax_sub_twin.plot(epochs, self.history['train']['head_cross_corr'], 'r-', label='Head Cross-Corr')
                ax_sub_twin.set_ylabel('Head Cross-Corr')
            
            ax_sub.set_title('Subspace Diversity & Orthogonality')
            ax_sub.set_ylabel('Ortho MSE')
            ax_sub.set_yscale('log')
            
            lines, labels = ax_sub.get_legend_handles_labels()
            if 'head_cross_corr' in self.history['train']:
                lines2, labels2 = ax_sub_twin.get_legend_handles_labels()
                ax_sub.legend(lines + lines2, labels + labels2, loc='upper right', fontsize='x-small')
            else:
                ax_sub.legend(loc='upper right', fontsize='x-small')
            ax_sub.grid(True, which="both", ls="-", alpha=0.5)
        elif 'loss_inter' in self.history['train']:
            ax_sub.plot(epochs, self.history['train']['loss_inter'], 'b-', label='Inter-Head Loss')
            ax_sub.plot(epochs, self.history['train']['loss_intra'], 'r-', label='Intra-Head Loss')
            
            ax_sub.legend(loc='upper right', fontsize='x-small')
            ax_sub.set_title('Subspace Health (Inter/Intra Diversity)')
            ax_sub.set_ylabel('Diversity Loss')
            ax_sub.set_yscale('log')
            ax_sub.grid(True, which="both", ls="-", alpha=0.5)
        elif 'subspace_loss' in self.history['train']:
            # Fallback for older checkpoints/training scripts
            ax_sub.plot(epochs, self.history['train']['subspace_loss'], 'k-', label='Total Subspace Loss', linewidth=2)
            ax_sub.set_title('Subspace Health (Total)')
            ax_sub.legend()
            ax_sub.grid(True)
            
        # 4. Matrix Health (Singular Values & Rank)
        if 'A_sing_val_avg' in self.history['train']:
            ax_mat.plot(epochs, self.history['train']['A_sing_val_avg'], 'b-', label='A Avg SV')
            ax_mat_twin = ax_mat.twinx()
            ax_mat_twin.plot(epochs, self.history['train']['A_cond'], 'b--', alpha=0.5, label='A Cond #')
            
            if 'active_rank_ratio' in self.history['train']:
                ax_mat.plot(epochs, self.history['train']['active_rank_ratio'], 'g-', label='Active Rank Ratio')
            
            ax_mat.set_title('Matrix Health (Singular Values & Rank)')
            ax_mat.set_ylabel('Avg SV / Rank Ratio')
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
