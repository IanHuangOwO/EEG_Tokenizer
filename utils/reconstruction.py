
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import mne

def visualize_reconstruction(train_batch, val_batch, epoch, output_dir='output/visualization/reconstruction'):
    """
    Plots original vs reconstructed signal for 1 Train sample and 3 diverse Validation samples.
    Layout: 
    - Rows: Bands (Raw, Delta, Theta, Alpha, Beta, Gamma)
    - Cols: Train_Ch0, Val_Ch0, Val_ChMid, Val_ChLast
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Unpack
    train_orig, train_recon = train_batch
    val_orig, val_recon = val_batch
    
    if train_orig is None or val_orig is None: return

    num_chans = train_orig.shape[1]
    mid_ch = num_chans // 2
    last_ch = num_chans - 1
    
    fs = 200.0
    time_vec = np.arange(train_orig.shape[-1]) / fs
    
    bands = {
        'Raw': None,
        'Delta (0.5-4)': (0.5, 4),
        'Theta (4-8)': (4, 8),
        'Alpha (8-13)': (8, 13),
        'Beta (13-30)': (13, 30),
        'Gamma (30-80)': (30, 80)
    }
    
    # Grid Setup: 6 Bands (Rows), 4 Samples (Cols)
    n_rows = len(bands)
    n_cols = 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 2.5), sharex=True)
    fig.suptitle(f"Reconstruction Analysis - Epoch {epoch}", fontsize=20, fontweight='bold')
    
    # Define Column configurations: (Source_Tensor_Pair, Channel_Index, Label)
    configs = [
        ((train_orig, train_recon), 0, f"Train (Ch 0)"),
        ((val_orig, val_recon), 0, f"Val (Ch 0)"),
        ((val_orig, val_recon), mid_ch, f"Val (Ch {mid_ch})"),
        ((val_orig, val_recon), last_ch, f"Val (Ch {last_ch})")
    ]

    for row_idx, (band_name, freqs) in enumerate(bands.items()):
        for col_idx, ((orig_batch, recon_batch), ch_idx, title) in enumerate(configs):
            ax = axes[row_idx, col_idx]
            
            # Extract single channel
            orig = orig_batch[0, ch_idx].detach().cpu().numpy()
            recon = recon_batch[0, ch_idx].detach().cpu().numpy()
            
            # Filter
            if freqs is None:
                y_o, y_r = orig, recon
            else:
                l_f, h_f = freqs
                o_mne = orig.reshape(1, -1).astype(np.float64)
                r_mne = recon.reshape(1, -1).astype(np.float64)
                try:
                    y_o = mne.filter.filter_data(o_mne, fs, l_f, h_f, method='iir', verbose=False)[0]
                    y_r = mne.filter.filter_data(r_mne, fs, l_f, h_f, method='iir', verbose=False)[0]
                except:
                    y_o, y_r = np.zeros_like(orig), np.zeros_like(recon)

            ax.plot(time_vec, y_o, 'k', alpha=0.6, linewidth=0.8, label='Orig')
            ax.plot(time_vec, y_r, 'r--', alpha=0.7, linewidth=0.8, label='Rec')
            
            # Formatting
            if row_idx == 0:
                ax.set_title(title, fontsize=14, fontweight='bold')
            
            if col_idx == 0:
                ax.set_ylabel(band_name, fontsize=12, rotation=0, labelpad=40, fontweight='bold')
            
            if row_idx == 0 and col_idx == n_cols - 1:
                ax.legend(fontsize=10)
                
            ax.grid(True, alpha=0.2)

    # X-Labels on bottom
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    filename = f'recon_epoch_{epoch}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    return save_path
