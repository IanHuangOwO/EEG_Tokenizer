
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

    def plot(self):
        """Saves a plot of the training curves (losses)."""
        if 'loss' not in self.history['train'] or not self.history['train']['loss']:
            return
            
        epochs = range(1, len(self.history['train']['loss']) + 1)
        has_val = 'loss' in self.history['val'] and len(self.history['val']['loss']) > 0

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 16), sharex=True)

        # Plot Main Losses
        ax1.plot(epochs, self.history['train']['loss'], 'b-', label='Train Total')
        if 'recon' in self.history['train']:
            ax1.plot(epochs, self.history['train']['recon'], 'g-', label='Train Recon')
        if 'vq' in self.history['train']:
            ax1.plot(epochs, self.history['train']['vq'], 'r-', label='Train VQ')
        
        if has_val:
            ax1.plot(epochs, self.history['val']['loss'], 'b--', alpha=0.7, label='Val Total')
            if 'recon' in self.history['val']:
                ax1.plot(epochs, self.history['val']['recon'], 'g--', alpha=0.7, label='Val Recon')
            if 'vq' in self.history['val']:
                ax1.plot(epochs, self.history['val']['vq'], 'r--', alpha=0.7, label='Val VQ')

        ax1.set_title('Global Losses')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot Components (Amp/Phase)
        if 'amp' in self.history['train']:
            ax2.plot(epochs, self.history['train']['amp'], color='orange', linestyle='-', label='Train Amp')
        if 'phase' in self.history['train']:
            ax2.plot(epochs, self.history['train']['phase'], color='purple', linestyle='-', label='Train Phase')
        
        if has_val:
             if 'amp' in self.history['val']:
                ax2.plot(epochs, self.history['val']['amp'], color='orange', linestyle='--', alpha=0.7, label='Val Amp')
             if 'phase' in self.history['val']:
                ax2.plot(epochs, self.history['val']['phase'], color='purple', linestyle='--', alpha=0.7, label='Val Phase')
             
        ax2.set_title('Reconstruction Components')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        # Plot Temporal
        if 'temp' in self.history['train']:
            ax3.plot(epochs, self.history['train']['temp'], 'm-', label='Train Temp (L1)')
        if has_val and 'temp' in self.history['val']:
            ax3.plot(epochs, self.history['val']['temp'], 'm--', alpha=0.7, label='Val Temp (L1)')
            
        ax3.set_title('Temporal Loss (Time Domain)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(save_path)
        plt.close(fig)

    def plot_metrics(self):
        """Saves a plot of specific training metrics (e.g., MSE, Codebook Health)."""
        self.save_csv()
        if 'temp_mse' not in self.history['train'] or not self.history['train']['temp_mse']:
            return
            
        epochs = range(1, len(self.history['train']['temp_mse']) + 1)
        has_val = 'temp_mse' in self.history['val'] and len(self.history['val']['temp_mse']) > 0

        # Create three subplots: MSE, Rank, and Health
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))

        # 1. Temporal MSE
        ax1.plot(epochs, self.history['train']['temp_mse'], 'b-', label='Train MSE')
        if has_val:
            ax1.plot(epochs, self.history['val']['temp_mse'], 'b--', alpha=0.7, label='Val MSE')
            
        ax1.set_title('Reconstruction Quality (MSE)')
        ax1.set_ylabel('MSE (Log Scale)')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, which="both", ls="-", alpha=0.5)

        # 2. Ranks (Effective Rank)
        if 'cb_erank' in self.history['train']:
            ax2.plot(epochs, self.history['train']['cb_erank'], 'r-', label='CB Rank')
            if 'wq_erank' in self.history['train']:
                ax2.plot(epochs, self.history['train']['wq_erank'], 'g-', label='Wq Rank')
            if 'wo_erank' in self.history['train']:
                ax2.plot(epochs, self.history['train']['wo_erank'], 'm-', label='Wo Rank')
            
            ax2.set_title('Geometric Rank Tracking')
            ax2.set_ylabel('Effective Rank')
            ax2.legend()
            ax2.grid(True)

        # 3. Codebook Health (Perplexity and Similarity)
        if 'perplexity' in self.history['train']:
            ax3.plot(epochs, self.history['train']['perplexity'], 'k-', label='Perplexity')
            ax3.set_ylabel('Perplexity', color='k')
            ax3.tick_params(axis='y', labelcolor='k')
            
            if 'cb_sim' in self.history['train']:
                ax3_twin = ax3.twinx()
                ax3_twin.plot(epochs, self.history['train']['cb_sim'], 'b-', label='Head Sim')
                ax3_twin.set_ylabel('Avg Head Similarity (0=Independent)', color='b')
                ax3_twin.tick_params(axis='y', labelcolor='b')
                ax3_twin.set_ylim(0, 1)

        ax3.set_title('Bottleneck Health Tracking')
        ax3.set_xlabel('Epoch')
        ax3.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'training_metrics.png')
        plt.savefig(save_path)
        plt.close(fig)
