
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

        # 2. Subspace Overlap (Orthogonality)
        if 'A_overlap' in self.history['train']:
            ax2.plot(epochs, self.history['train']['A_overlap'], 'r-', label='A Overlap (Filter)')
            if 'B_overlap' in self.history['train']:
                ax2.plot(epochs, self.history['train']['B_overlap'], 'b-', label='B Overlap (Synth)')
            
            ax2.set_title('Subspace Diversification (Lower is better)')
            ax2.set_ylabel('Overlap Score')
            ax2.legend()
            ax2.grid(True)

        # 3. Gating Mechanism (Head Weights)
        if 'head_weight_mean' in self.history['train']:
            ax3.plot(epochs, self.history['train']['head_weight_mean'], 'k-', label='Mean Weight')
            if 'head_weight_max' in self.history['train']:
                ax3.plot(epochs, self.history['train']['head_weight_max'], 'g--', label='Max Weight')
            if 'head_weight_min' in self.history['train']:
                ax3.plot(epochs, self.history['train']['head_weight_min'], 'r--', label='Min Weight')
            
            ax3.set_ylabel('Weight Value')
            ax3.legend(loc='upper left')

        ax3.set_title('Expert Gating Mechanism')
        ax3.set_xlabel('Epoch')
        ax3.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'training_metrics.png')
        plt.savefig(save_path)
        plt.close(fig)
