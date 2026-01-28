
import os
import matplotlib.pyplot as plt

class Plotter:
    """A helper class to track and plot losses during training."""
    def __init__(self, output_dir='output/visualization/training_curves'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.history = {
            'train': {'total_loss': [], 'recon_loss': [], 'vq_loss': [], 'mse': [], 'amp_loss': [], 'phase_loss': []},
            'val': {'total_loss': [], 'recon_loss': [], 'vq_loss': [], 'mse': [], 'amp_loss': [], 'phase_loss': []}
        }

    def update(self, train_metrics, val_metrics=None):
        """
        Adds a new set of loss values for the current epoch.
        metrics should be a tuple: (total, recon, vq, mse, amp, phase)
        """
        # Unpack train tuple
        t_total, t_recon, t_vq, t_mse, t_amp, t_phase = train_metrics
        self.history['train']['total_loss'].append(t_total)
        self.history['train']['recon_loss'].append(t_recon)
        self.history['train']['vq_loss'].append(t_vq)
        self.history['train']['mse'].append(t_mse)
        self.history['train']['amp_loss'].append(t_amp)
        self.history['train']['phase_loss'].append(t_phase)
        
        if val_metrics:
            v_total, v_recon, v_vq, v_mse, v_amp, v_phase = val_metrics
            self.history['val']['total_loss'].append(v_total)
            self.history['val']['recon_loss'].append(v_recon)
            self.history['val']['vq_loss'].append(v_vq)
            self.history['val']['mse'].append(v_mse)
            self.history['val']['amp_loss'].append(v_amp)
            self.history['val']['phase_loss'].append(v_phase)

    def plot(self):
        """Saves a plot of the training curves."""
        epochs = range(1, len(self.history['train']['total_loss']) + 1)
        has_val = len(self.history['val']['total_loss']) > 0

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 16), sharex=True)

        # Plot Main Losses
        ax1.plot(epochs, self.history['train']['total_loss'], 'b-', label='Train Total')
        ax1.plot(epochs, self.history['train']['recon_loss'], 'g-', label='Train Recon')
        ax1.plot(epochs, self.history['train']['vq_loss'], 'r-', label='Train VQ')
        
        if has_val:
            ax1.plot(epochs, self.history['val']['total_loss'], 'b--', alpha=0.7, label='Val Total')
            ax1.plot(epochs, self.history['val']['recon_loss'], 'g--', alpha=0.7, label='Val Recon')
            ax1.plot(epochs, self.history['val']['vq_loss'], 'r--', alpha=0.7, label='Val VQ')

        ax1.set_title('Global Losses')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot Components (Amp/Phase)
        ax2.plot(epochs, self.history['train']['amp_loss'], color='orange', linestyle='-', label='Train Amp')
        ax2.plot(epochs, self.history['train']['phase_loss'], color='purple', linestyle='-', label='Train Phase')
        
        if has_val:
             ax2.plot(epochs, self.history['val']['amp_loss'], color='orange', linestyle='--', alpha=0.7, label='Val Amp')
             ax2.plot(epochs, self.history['val']['phase_loss'], color='purple', linestyle='--', alpha=0.7, label='Val Phase')
             
        ax2.set_title('Reconstruction Components')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        # Plot MSE
        ax3.plot(epochs, self.history['train']['mse'], 'm-', label='Train MSE')
        if has_val:
            ax3.plot(epochs, self.history['val']['mse'], 'm--', alpha=0.7, label='Val MSE')
            
        ax3.set_title('Reconstruction MSE (Time Domain)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('MSE')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(save_path)
        plt.close(fig)
        # print(f"Updated training curve plot saved to {save_path}")
