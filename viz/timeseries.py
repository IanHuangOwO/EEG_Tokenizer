"""Band-filtered orig-vs-recon time series plot, one row per channel."""

import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_reconstruction(train_batch, val_batch, epoch,
                             output_dir='output/visualization/reconstruction',
                             channel_names=None,
                             subject_id=None, trial_idx=None,
                             mask=None, patch_len=100):
    """
    Band-filtered orig vs recon for all channels of one val sample.
    Rows: channels. Cols: Raw / Delta / Theta / Alpha / Beta / Gamma.
    Masked patches highlighted in red per channel.
    mask: [C, N] bool numpy array or None.
    """
    os.makedirs(output_dir, exist_ok=True)

    val_orig, val_recon = val_batch
    if val_orig is None:
        return

    fs    = 200.0
    orig  = val_orig[0].detach().cpu().numpy()
    recon = val_recon[0].detach().cpu().numpy()
    C = orig.shape[0]
    n = min(orig.shape[-1], recon.shape[-1])
    orig, recon = orig[:, :n], recon[:, :n]
    t = np.arange(n) / fs

    bands = {
        'Raw':            None,
        'Delta (0.5-4)':  (0.5,  4),
        'Theta (4-8)':    (4,    8),
        'Alpha (8-13)':   (8,   13),
        'Beta (13-30)':   (13,  30),
        'Gamma (30-80)':  (30,  80),
    }

    n_rows, n_cols = C, len(bands)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 1.2),
                             sharex=True, constrained_layout=True)
    if C == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Reconstruction (Val) — Epoch {epoch}",
                 fontsize=14, fontweight='bold')

    for col, (band_name, freqs) in enumerate(bands.items()):
        for row in range(C):
            ax = axes[row, col]
            yo, yr = _band_filter(orig[row], recon[row], freqs, fs)
            ax.plot(t, yo, color='#666666', lw=0.5, alpha=0.7)
            ax.plot(t, yr, 'r--', lw=0.5, alpha=0.8)
            # shade masked patches
            if mask is not None:
                ch_mask = mask[row] if mask.ndim == 2 else mask  # [N]
                for p_idx, is_masked in enumerate(ch_mask):
                    if is_masked:
                        t0 = p_idx * patch_len / fs
                        t1 = (p_idx + 1) * patch_len / fs
                        ax.axvspan(t0, t1, color='red', alpha=0.15, linewidth=0)
            ax.set_yticks([])
            ax.grid(True, alpha=0.08)
            if row == 0:
                ax.set_title(band_name, fontsize=8, fontweight='bold')
            if col == 0:
                ch_label = channel_names[row] if channel_names else f'Ch {row}'
                ax.set_ylabel(ch_label, fontsize=5, rotation=0, labelpad=28, va='center')
            if row < C - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Time (s)", fontsize=6)

    prefix   = f"sub{subject_id}_trial{trial_idx}_" if subject_id is not None else ""
    ep_tag   = f"ep{epoch:04d}_" if epoch is not None else ""
    path = os.path.join(output_dir, f"{prefix}{ep_tag}recon_signal.png")
    plt.savefig(path, dpi=80, bbox_inches='tight')
    plt.close()
    return path


def _band_filter(orig, recon, freqs, fs=200.0):
    """Band-filter orig and recon using MNE IIR filter. Returns (orig_filtered, recon_filtered)."""
    if freqs is None:
        return orig, recon
    try:
        import mne
        l_f, h_f = freqs
        yo = mne.filter.filter_data(orig.reshape(1, -1).astype(np.float64),
                                    fs, l_f, h_f, method='iir', verbose=False)[0]
        yr = mne.filter.filter_data(recon.reshape(1, -1).astype(np.float64),
                                    fs, l_f, h_f, method='iir', verbose=False)[0]
        return yo, yr
    except Exception:
        return np.zeros_like(orig), np.zeros_like(recon)
