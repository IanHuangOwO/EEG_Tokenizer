
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import mne

def visualize_reconstruction(original, reconstruction, epoch, step=None, output_dir='output/visualization/reconstruction', ch=0):
    """
    Plots original vs reconstructed signal for a single channel, decomposed into bands.
    original, reconstruction: (B, N, T) tensors
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Take first batch item and specified channel
    orig_np = original[0, ch].detach().cpu().numpy()
    recon_np = reconstruction[0, ch].detach().cpu().numpy()
    
    # Ensure shape (1, T) for MNE
    orig_mne = orig_np.reshape(1, -1).astype(np.float64)
    recon_mne = recon_np.reshape(1, -1).astype(np.float64)
    
    fs = 200.0
    time_vec = np.arange(len(orig_np)) / fs
    
    bands = {
        'Raw': None,
        'Delta (0.5-4 Hz)': (0.5, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
        'Gamma (30-80 Hz)': (30, 80)
    }
    
    fig, axes = plt.subplots(len(bands), 1, figsize=(10, 15), sharex=True)
    
    if step is not None:
        fig.suptitle(f"Reconstruction Analysis - Epoch {epoch}, Step {step}")
    else:
        fig.suptitle(f"Reconstruction Analysis - Epoch {epoch}")
    
    for i, (name, freqs) in enumerate(bands.items()):
        ax = axes[i]
        
        if freqs is None:
            # Plot Raw
            y_orig = orig_np
            y_recon = recon_np
        else:
            # Filter
            l_freq, h_freq = freqs
            # Use IIR for short signals
            try:
                y_orig = mne.filter.filter_data(orig_mne, fs, l_freq, h_freq, method='iir', verbose=False)[0]
                y_recon = mne.filter.filter_data(recon_mne, fs, l_freq, h_freq, method='iir', verbose=False)[0]
            except Exception as e:
                print(f"Filtering failed for {name}: {e}")
                y_orig = np.zeros_like(orig_np)
                y_recon = np.zeros_like(recon_np)

        ax.plot(time_vec, y_orig, label='Original', color='black', alpha=0.7, linewidth=1.0)
        ax.plot(time_vec, y_recon, label='Reconstructed', color='red', alpha=0.7, linestyle='--', linewidth=1.0)
        
        ax.set_title(name)
        ax.set_ylabel("Amp")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize='small')

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust for suptitle
    
    if step is not None:
        filename = f'recon_epoch_{epoch}_step_{step}.png'
    else:
        filename = f'recon_epoch_{epoch}.png'
        
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    return save_path
