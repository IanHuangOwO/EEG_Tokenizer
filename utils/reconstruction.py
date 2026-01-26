
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import mne

def visualize_reconstruction(train_batch, val_batch, epoch, output_dir='output/visualization/reconstruction', ch=0):
    """
    Plots original vs reconstructed signal for a single channel, decomposed into bands.
    Plots Train (Left) and Validation (Right) side-by-side.
    batch: tuple (original, reconstruction) where tensors are (B, N, T)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Unpack
    train_orig, train_recon = train_batch
    val_orig, val_recon = val_batch
    
    # Helper to prep data
    def prep_data(orig_t, recon_t):
        if orig_t is None or recon_t is None: return None, None
        o = orig_t[0, ch].detach().cpu().numpy()
        r = recon_t[0, ch].detach().cpu().numpy()
        return o, r

    t_orig, t_recon = prep_data(train_orig, train_recon)
    v_orig, v_recon = prep_data(val_orig, val_recon)
    
    if t_orig is None or v_orig is None:
        print("Skipping visualization due to missing data")
        return

    fs = 200.0
    time_vec = np.arange(len(t_orig)) / fs
    
    bands = {
        'Raw': None,
        'Delta (0.5-4 Hz)': (0.5, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
        'Gamma (30-80 Hz)': (30, 80)
    }
    
    fig, axes = plt.subplots(len(bands), 2, figsize=(16, 18), sharex=True)
    fig.suptitle(f"Reconstruction Analysis - Epoch {epoch}", fontsize=16)
    
    # Column headers
    axes[0, 0].set_title(f"Training Sample (Ch {ch}) - Raw")
    axes[0, 1].set_title(f"Validation Sample (Ch {ch}) - Raw")

    for i, (band_name, freqs) in enumerate(bands.items()):
        # Process both Train (col 0) and Val (col 1)
        for col, (orig, recon) in enumerate([(t_orig, t_recon), (v_orig, v_recon)]):
            ax = axes[i, col]
            
            # Prepare MNE compatible
            o_mne = orig.reshape(1, -1).astype(np.float64)
            r_mne = recon.reshape(1, -1).astype(np.float64)

            if freqs is None:
                y_o, y_r = orig, recon
            else:
                l_f, h_f = freqs
                try:
                    y_o = mne.filter.filter_data(o_mne, fs, l_f, h_f, method='iir', verbose=False)[0]
                    y_r = mne.filter.filter_data(r_mne, fs, l_f, h_f, method='iir', verbose=False)[0]
                except Exception:
                    y_o, y_r = np.zeros_like(orig), np.zeros_like(recon)
            
            ax.plot(time_vec, y_o, 'k', alpha=0.6, linewidth=1.0, label='Original')
            ax.plot(time_vec, y_r, 'r--', alpha=0.7, linewidth=1.0, label='Recon')
            
            if col == 0:
                ax.set_ylabel(band_name)
            
            ax.grid(True, alpha=0.3)
            if i == 0 and col == 1: # Legend on top right only
                ax.legend(loc='upper right')

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    filename = f'recon_epoch_{epoch}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    return save_path
