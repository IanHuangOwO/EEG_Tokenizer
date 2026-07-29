"""
Shared epoch-snapshot panels (topo_psd_filter, attn_topo) reused by
BaseEpochChecker._render_snapshot (MeFSQ Experts or MeSAE Filters — viz/extract.py's
extract_head_*/extract_filter_* already return the shared PsdResult/SpectraResult
dataclasses). Keeps the two training-phase methods producing the exact same panel format
instead of drifting — see model/base_checker.py and docs/adr/0004-model-plugin-base-classes.md.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from viz.topomap import draw_topomap


def plot_topo_psd_filter(out_path, pos2d, raw_power, recon_power, psd_raw, psd_recon,
                          psd_ch_x, psd_x, freqs, importance, cmap='YlOrRd',
                          subject_id=None, trial_idx=None, epoch_tag='', unit_label='Filter',
                          l_freq=None, h_freq=None):
    """
    Raw / Full-recon / per-unit (Expert or Filter) topo + PSD, side by side, one row each,
    sorted by `importance` below the first two fixed rows (Raw, Full Recon). Raw/recon PSD
    must already be computed with the same n_fft/freq axis as psd_x so every row is directly
    comparable, not just visually similar (see BaseEpochChecker.check_pretrain for the FFT call).

    psd_ch_x: [C, Q] per-unit per-channel decoded activation. psd_x: [Q, C, F] per-unit PSD.
    freqs/psd_x/psd_raw/psd_recon are expected already band-cropped by the caller if desired
    (l_freq/h_freq here are only used for the x-axis label/ticks, not re-cropping).
    """
    Q = psd_ch_x.shape[1]
    sorted_ord = np.argsort(importance)[::-1]
    freq_label = 'Hz' if freqs is not None and len(freqs) else 'cyc/patch'
    freq_ticks = np.linspace(freqs[0], freqs[-1], 5)

    entries = [('Raw', raw_power, psd_raw), ('Full Recon', recon_power, psd_recon)]
    entries += [(f'{unit_label[0]}{q} ({importance[q]:.3f})', psd_ch_x[:, q], psd_x[q]) for q in sorted_ord]
    n_items = len(entries)
    n_rows = math.ceil(n_items / 2)

    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 3.0 * n_rows), squeeze=False, constrained_layout=True)
    fig.suptitle(f"Raw / Recon / {unit_label} Topo + PSD ({unit_label}s sorted by contribution) — "
                 f"Sub {subject_id}, Trial {trial_idx}{epoch_tag}", fontsize=13, fontweight='bold')

    def _psd_row(ax_topo, ax_psd, power, psd_cf, label):
        im_t = draw_topomap(ax_topo, pos2d, power, cmap=cmap, vmin=power.min(), vmax=power.max())
        ax_topo.set_title(f'{label} Topo (power)', fontsize=9, fontweight='bold')
        fig.colorbar(im_t, ax=ax_topo, fraction=0.05, pad=0.02)

        ax_psd.imshow(psd_cf, aspect='auto', cmap=cmap, origin='lower',
                      extent=[freqs[0], freqs[-1], 0, psd_cf.shape[0]])
        ax_psd.set_title(f'{label} PSD (channel x freq)', fontsize=9, fontweight='bold')
        ax_psd.set_xticks(freq_ticks)
        ax_psd.set_xticklabels([f'{f:.0f}' for f in freq_ticks], fontsize=6)
        ax_psd.set_xlabel(freq_label, fontsize=7)
        ax_psd.set_yticks([])

    for i, (label, power, psd_cf) in enumerate(entries):
        row, col_pair = divmod(i, 2)
        _psd_row(axes[row, col_pair * 2], axes[row, col_pair * 2 + 1], power, psd_cf, label)
    for i in range(n_items, n_rows * 2):
        row, col_pair = divmod(i, 2)
        axes[row, col_pair * 2].axis('off')
        axes[row, col_pair * 2 + 1].axis('off')

    fig.text(0.5, 0.005, 'PSD y-axis: Channel (Fp1 -> Iz)', ha='center', fontsize=8)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_attn_topo(out_path, pos2d, attn, importance, channel_names, valid_channels=None,
                    subject_id=None, trial_idx=None, epoch_tag='', unit_label='Filter'):
    """
    Per-unit (Expert/Filter/Head) channel-attention topography, one topomap per unit
    (sorted by `importance`), plus one large Channel x Unit heatmap spanning all rows.
    attn: [Q, C] — each unit's own attention weight over channels (rows sum to 1).
    valid_channels: [C] bool or None — padded/invalid channels hidden from the topomaps only.
    """
    Q, C = attn.shape
    sorted_ord = np.argsort(importance)[::-1]
    attn_masked = attn if valid_channels is None else np.where(valid_channels[None, :], attn, np.nan)

    n_topo_rows = math.ceil(Q / 2)
    fig = plt.figure(figsize=(40.0, 3.0 * n_topo_rows), constrained_layout=True)
    gs = fig.add_gridspec(n_topo_rows, 3, width_ratios=[1.0, 1.0, 3.5], wspace=0.3)

    for i, q_orig in enumerate(sorted_ord):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        valid = ~np.isnan(attn_masked[q_orig])
        im = draw_topomap(ax, pos2d[valid], attn_masked[q_orig][valid], cmap='viridis')
        ax.set_title(f'{unit_label} {q_orig}  ({importance[q_orig]:.3f})', fontsize=9, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    if Q % 2:
        fig.add_subplot(gs[n_topo_rows - 1, 1]).axis('off')

    ax_big = fig.add_subplot(gs[:, 2])
    attn_ordered = attn[sorted_ord].T  # [C, Q], columns ordered to match the topomap panels
    im_big = ax_big.imshow(attn_ordered, aspect='auto', cmap='viridis')
    ax_big.set_yticks(range(C))
    ax_big.set_yticklabels(channel_names, fontsize=5)
    ax_big.set_xticks(range(Q))
    ax_big.set_xticklabels([f'{unit_label[0]}{q}\n{importance[q]:.3f}' for q in sorted_ord], fontsize=6, rotation=90)
    ax_big.set_xlabel(f'{unit_label} (sorted by contribution)', fontsize=9)
    ax_big.set_title(f'Channel x {unit_label} Attention', fontsize=10, fontweight='bold')
    fig.colorbar(im_big, ax=ax_big, fraction=0.03, pad=0.02).set_label('Attention weight', fontsize=8)

    fig.suptitle(f"Per-{unit_label} Channel Attention ({unit_label}s sorted by contribution) — "
                 f"Sub {subject_id}, Trial {trial_idx}{epoch_tag}", fontsize=12, fontweight='bold')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
