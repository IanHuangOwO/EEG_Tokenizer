
import os
import matplotlib.pyplot as plt

class Plotter:
    """A helper class to track and plot losses during training."""
    def __init__(self, output_dir='output/visualization/training_curves'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.history = {
            'train': {'total_loss': [], 'recon_loss': [], 'vq_loss': [], 'mse': []},
            'val': {'total_loss': [], 'recon_loss': [], 'vq_loss': [], 'mse': []}
        }

    def update(self, train_metrics, val_metrics=None):
        """
        Adds a new set of loss values for the current epoch.
        metrics should be a dict or tuple: (total, recon, vq, mse)
        """
        # Unpack train tuple
        t_total, t_recon, t_vq, t_mse = train_metrics
        self.history['train']['total_loss'].append(t_total)
        self.history['train']['recon_loss'].append(t_recon)
        self.history['train']['vq_loss'].append(t_vq)
        self.history['train']['mse'].append(t_mse)
        
        if val_metrics:
            v_total, v_recon, v_vq, v_mse = val_metrics
            self.history['val']['total_loss'].append(v_total)
            self.history['val']['recon_loss'].append(v_recon)
            self.history['val']['vq_loss'].append(v_vq)
            self.history['val']['mse'].append(v_mse)

    def plot(self):
        """Saves a plot of the training curves."""
        epochs = range(1, len(self.history['train']['total_loss']) + 1)
        has_val = len(self.history['val']['total_loss']) > 0

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

        # Plot Main Losses
        ax1.plot(epochs, self.history['train']['total_loss'], 'b-', label='Train Total')
        ax1.plot(epochs, self.history['train']['recon_loss'], 'g-', label='Train Recon')
        ax1.plot(epochs, self.history['train']['vq_loss'], 'r-', label='Train VQ')
        
        if has_val:
            ax1.plot(epochs, self.history['val']['total_loss'], 'b--', alpha=0.7, label='Val Total')
            ax1.plot(epochs, self.history['val']['recon_loss'], 'g--', alpha=0.7, label='Val Recon')
            # VQ loss is usually 0 for FSQ/LFQ or tracked differently, but plotting for completeness
            ax1.plot(epochs, self.history['val']['vq_loss'], 'r--', alpha=0.7, label='Val VQ')

        ax1.set_title('Losses (Solid: Train, Dashed: Val)')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot MSE
        ax2.plot(epochs, self.history['train']['mse'], 'm-', label='Train MSE')
        if has_val:
            ax2.plot(epochs, self.history['val']['mse'], 'm--', alpha=0.7, label='Val MSE')
            
        ax2.set_title('Reconstruction MSE (Time Domain)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Updated training curve plot saved to {save_path}")
