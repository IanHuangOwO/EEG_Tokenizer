
import os
import matplotlib.pyplot as plt

class Plotter:
    """A helper class to track and plot losses during training."""
    def __init__(self, output_dir='output/visualization/training_curves'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.history = {
            'train': {
                'total_loss': [], 'recon_loss': [], 'vq_loss': [], 
                'temp_loss': [], 'amp_loss': [], 'phase_loss': [],
                'temp_mse': []
            },
            'val': {
                'total_loss': [], 'recon_loss': [], 'vq_loss': [], 
                'temp_loss': [], 'amp_loss': [], 'phase_loss': [],
                'temp_mse': []
            }
        }

    def update(self, train_metrics, val_metrics=None):
        """
        Adds a new set of loss values for the current epoch.
        metrics should be a tuple: (total, vq, recon, temp, amp, phase, temp_mse)
        Note: The indices must match the order in train_tokenizer.py
        """
        # Unpack train tuple
        t_total, t_vq, t_recon, t_temp, t_amp, t_phase, t_mse = train_metrics
        self.history['train']['total_loss'].append(t_total)
        self.history['train']['vq_loss'].append(t_vq)
        self.history['train']['recon_loss'].append(t_recon)
        self.history['train']['temp_loss'].append(t_temp)
        self.history['train']['amp_loss'].append(t_amp)
        self.history['train']['phase_loss'].append(t_phase)
        self.history['train']['temp_mse'].append(t_mse)
        
        if val_metrics:
            v_total, v_vq, v_recon, v_temp, v_amp, v_phase, v_mse = val_metrics
            self.history['val']['total_loss'].append(v_total)
            self.history['val']['vq_loss'].append(v_vq)
            self.history['val']['recon_loss'].append(v_recon)
            self.history['val']['temp_loss'].append(v_temp)
            self.history['val']['amp_loss'].append(v_amp)
            self.history['val']['phase_loss'].append(v_phase)
            self.history['val']['temp_mse'].append(v_mse)

    def plot(self):
        """Saves a plot of the training curves (losses)."""
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

        # Plot Temporal
        ax3.plot(epochs, self.history['train']['temp_loss'], 'm-', label='Train Temp (L1)')
        if has_val:
            ax3.plot(epochs, self.history['val']['temp_loss'], 'm--', alpha=0.7, label='Val Temp (L1)')
            
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
        """Saves a plot of specific training metrics (e.g., MSE)."""
        epochs = range(1, len(self.history['train']['temp_mse']) + 1)
        has_val = len(self.history['val']['temp_mse']) > 0

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot Temporal MSE
        ax.plot(epochs, self.history['train']['temp_mse'], 'b-', label='Train Temp MSE')
        if has_val:
            ax.plot(epochs, self.history['val']['temp_mse'], 'b--', alpha=0.7, label='Val Temp MSE')
            
        ax.set_title('Temporal Reconstruction Quality (MSE)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.set_yscale('log') # Usually helpful for MSE
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.5)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'training_metrics.png')
        plt.savefig(save_path)
        plt.close(fig)
