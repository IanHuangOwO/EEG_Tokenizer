
import os
import matplotlib.pyplot as plt

class Plotter:
    """A helper class to track and plot losses during training."""
    def __init__(self, output_dir='output/visualization/training_curves'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.history = {
            'total_loss': [],
            'recon_loss': [],
            'vq_loss': [],
            'mse': []
        }

    def update(self, total_loss, recon_loss, vq_loss, mse):
        """Adds a new set of loss values for the current epoch."""
        self.history['total_loss'].append(total_loss)
        self.history['recon_loss'].append(recon_loss)
        self.history['vq_loss'].append(vq_loss)
        self.history['mse'].append(mse)

    def plot(self):
        """Saves a plot of the training curves."""
        epochs = range(1, len(self.history['total_loss']) + 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Plot Main Losses on first axis
        ax1.plot(epochs, self.history['total_loss'], 'bo-', label='Total Loss')
        ax1.plot(epochs, self.history['recon_loss'], 'go-', label='Reconstruction Loss')
        ax1.plot(epochs, self.history['vq_loss'], 'ro-', label='VQ Loss')
        ax1.set_title('Training Losses')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot MSE on second axis
        ax2.plot(epochs, self.history['mse'], 'mo-', label='Reconstruction MSE (Time Domain)')
        ax2.set_title('Time Domain Reconstruction MSE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Updated training curve plot saved to {save_path}")
