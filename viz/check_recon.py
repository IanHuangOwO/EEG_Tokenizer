import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch


def _select_channels(total_ch: int, n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n >= total_ch:
        return np.arange(total_ch)
    return np.sort(rng.choice(total_ch, n, replace=False))


def _plot_combined_overview(data, sel_ch, ch_names, subject_id, trial_idx, viz_dir,
                            epoch_tag=''):
    """
    2-panel overview:
      LEFT  — raw vs recon time series (selected channels)
      RIGHT — Welch PSD comparison (selected channels)
    """
    n_ch    = len(sel_ch)
    fs      = data['fs']
    T       = data['T']
    raw     = data['raw']
    recon   = data['recon']
    labels  = [ch_names[i] for i in sel_ch]
    t       = np.arange(T) / fs
    nperseg = min(256, T)

    # PSD
    f_ref, _ = welch(raw[sel_ch[0]], fs=fs, nperseg=nperseg)
    raw_psd   = np.stack([welch(raw[ci],   fs=fs, nperseg=nperseg)[1] for ci in sel_ch])
    recon_psd = np.stack([welch(recon[ci], fs=fs, nperseg=nperseg)[1] for ci in sel_ch])
    raw_psd_db   = 10 * np.log10(raw_psd   + 1e-12)
    recon_psd_db = 10 * np.log10(recon_psd + 1e-12)

    fig = plt.figure(figsize=(24, max(8.0, n_ch * 1.4)), constrained_layout=True)
    fig.suptitle(f"Overview — Sub {subject_id} Trial {trial_idx}",
                 fontsize=13, fontweight='bold')
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[14, 10], wspace=0.06)

    # LEFT: time series
    gs_l = gridspec.GridSpecFromSubplotSpec(n_ch, 1, subplot_spec=outer[0], hspace=0.05)
    for row, ci in enumerate(sel_ch):
        ax = fig.add_subplot(gs_l[row])
        ax.plot(t, raw[ci],   color='#444444', lw=0.6, alpha=0.8, label='Raw')
        ax.plot(t, recon[ci], color='red',     lw=0.6, alpha=0.8, ls='--', label='Recon')
        ax.set_yticks([])
        ax.set_ylabel(labels[row], fontsize=5, rotation=0, labelpad=28, va='center')
        ax.grid(True, alpha=0.1)
        if row == 0:
            ax.legend(fontsize=6, loc='upper right')
        if row < n_ch - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Time (s)', fontsize=7)

    # RIGHT: PSD heatmaps
    gs_r = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], hspace=0.14)
    psd_vmin = min(raw_psd_db.min(), recon_psd_db.min())
    psd_vmax = max(raw_psd_db.max(), recon_psd_db.max())
    err      = (raw_psd_db - recon_psd_db) ** 2
    err_vmax = float(np.percentile(err, 98))

    for r_idx, (mat, title, cmap, vmin, vmax) in enumerate([
        (raw_psd_db,   'Raw PSD (dB/Hz)',   'RdBu_r', psd_vmin, psd_vmax),
        (recon_psd_db, 'Recon PSD (dB/Hz)', 'RdBu_r', psd_vmin, psd_vmax),
        (err,          'PSD MSE (dB^2)',     'magma',  0,        max(err_vmax, 1e-6)),
    ]):
        ax = fig.add_subplot(gs_r[r_idx])
        im = ax.imshow(mat, aspect='auto', origin='lower', cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       extent=[f_ref[0], f_ref[-1], -0.5, n_ch - 0.5])
        ax.set_title(title, fontsize=8, fontweight='bold')
        if r_idx == 0:
            ax.set_yticks(range(n_ch))
            ax.set_yticklabels(labels, fontsize=5)
        else:
            ax.set_yticks([])
        if r_idx < 2:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Frequency (Hz)', fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

    out = os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_overview.png")
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [recon] Overview saved -> {out}")
