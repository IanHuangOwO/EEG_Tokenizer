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

from viz.topomap import draw_topomap, build_triangulation


def plot_topo_psd_filter(out_path, pos2d, raw_power, recon_power, psd_raw, psd_recon,
                          psd_ch_x, psd_x, freqs, importance, cmap='YlOrRd',
                          subject_id=None, trial_idx=None, epoch_tag='', unit_label='Filter',
                          l_freq=None, h_freq=None, unit_colors=None):
    """
    Raw / Full-recon / per-unit (Expert or Filter) topo + PSD, side by side, one row each,
    sorted by `importance` below the first two fixed rows (Raw, Full Recon). Raw/recon PSD
    must already be computed with the same n_fft/freq axis as psd_x so every row is directly
    comparable, not just visually similar (see BaseEpochChecker.check_pretrain for the FFT call).

    psd_ch_x: [C, Q] per-unit per-channel decoded activation. psd_x: [Q, C, F] per-unit PSD.
    freqs/psd_x/psd_raw/psd_recon are expected already band-cropped by the caller if desired
    (l_freq/h_freq here are only used for the x-axis label/ticks, not re-cropping).
    unit_colors: optional [Q] list of title colors (e.g. MeSAE routed-gating: red = shared
    Filter, orange = routed Filter selected for this trial) — None means no color override.
    """
    Q = psd_ch_x.shape[1]
    sorted_ord = np.argsort(importance)[::-1]
    freq_label = 'Hz' if freqs is not None and len(freqs) else 'cyc/patch'
    freq_ticks = np.linspace(freqs[0], freqs[-1], 5)

    entries = [('Raw', raw_power, psd_raw, 'black'), ('Full Recon', recon_power, psd_recon, 'black')]
    entries += [(f'{unit_label[0]}{q} ({importance[q]:.3f})', psd_ch_x[:, q], psd_x[q],
                 unit_colors[q] if unit_colors else 'black') for q in sorted_ord]
    n_items = len(entries)
    # (topo, psd) pairs per row — was fixed at 2 (4 total columns), bumped to 3 to cut
    # canvas height (and therefore savefig render/encode time) by ~1/3 for large Q.
    n_col_pairs = 3
    n_rows = math.ceil(n_items / n_col_pairs)

    fig, axes = plt.subplots(n_rows, n_col_pairs * 2, figsize=(10 * n_col_pairs, 3.0 * n_rows),
                              squeeze=False, constrained_layout=True)
    fig.suptitle(f"Raw / Recon / {unit_label} Topo + PSD ({unit_label}s sorted by contribution) — "
                 f"Sub {subject_id}, Trial {trial_idx}{epoch_tag}", fontsize=13, fontweight='bold')

    # Every row's topomap shares the same electrode layout (pos2d) — triangulate once
    # instead of once per row (n_items rows, e.g. 66 for 64 Filters + Raw/Recon).
    triang = build_triangulation(pos2d)

    def _psd_row(ax_topo, ax_psd, power, psd_cf, label, color):
        im_t = draw_topomap(ax_topo, pos2d, power, cmap=cmap, vmin=power.min(), vmax=power.max(), triang=triang)
        ax_topo.set_title(f'{label} Topo (power)', fontsize=9, fontweight='bold', color=color)
        fig.colorbar(im_t, ax=ax_topo, fraction=0.05, pad=0.02)

        ax_psd.imshow(psd_cf, aspect='auto', cmap=cmap, origin='lower',
                      extent=[freqs[0], freqs[-1], 0, psd_cf.shape[0]])
        ax_psd.set_title(f'{label} PSD (channel x freq)', fontsize=9, fontweight='bold', color=color)
        ax_psd.set_xticks(freq_ticks)
        ax_psd.set_xticklabels([f'{f:.0f}' for f in freq_ticks], fontsize=6)
        ax_psd.set_xlabel(freq_label, fontsize=7)
        ax_psd.set_yticks([])

    for i, (label, power, psd_cf, color) in enumerate(entries):
        row, col_pair = divmod(i, n_col_pairs)
        _psd_row(axes[row, col_pair * 2], axes[row, col_pair * 2 + 1], power, psd_cf, label, color)
    for i in range(n_items, n_rows * n_col_pairs):
        row, col_pair = divmod(i, n_col_pairs)
        axes[row, col_pair * 2].axis('off')
        axes[row, col_pair * 2 + 1].axis('off')

    fig.text(0.5, 0.005, 'PSD y-axis: Channel (Fp1 -> Iz)', ha='center', fontsize=8)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_attn_topo(out_path, pos2d, attn, importance, channel_names, valid_channels=None,
                    subject_id=None, trial_idx=None, epoch_tag='', unit_label='Filter',
                    unit_colors=None):
    """
    Per-unit (Expert/Filter/Head) channel-attention topography, one topomap per unit
    (sorted by `importance`), plus one large Channel x Unit heatmap spanning all rows.
    attn: [Q, C] — each unit's own attention weight over channels (rows sum to 1).
    valid_channels: [C] bool or None — padded/invalid channels hidden from the topomaps only.
    unit_colors: optional [Q] list of title/tick-label colors (e.g. MeSAE routed-gating:
    red = shared Filter, orange = routed Filter selected for this trial) — None means no
    color override (default black).
    """
    Q, C = attn.shape
    sorted_ord = np.argsort(importance)[::-1]
    attn_masked = attn if valid_channels is None else np.where(valid_channels[None, :], attn, np.nan)

    # Topomap columns per row — was fixed at 2, bumped to 4 to roughly halve the number
    # of rows (and therefore canvas height / savefig render+encode time) for large Q.
    # Width scaled to keep each topomap's and the heatmap's on-canvas size unchanged.
    n_topo_cols = 4
    heatmap_ratio = 3.5
    old_width, old_ratio_sum = 40.0, 2 * 1.0 + heatmap_ratio
    per_ratio_width = old_width / old_ratio_sum
    fig_width = per_ratio_width * (n_topo_cols * 1.0 + heatmap_ratio)

    n_topo_rows = math.ceil(Q / n_topo_cols)
    fig = plt.figure(figsize=(fig_width, 3.0 * n_topo_rows), constrained_layout=True)
    gs = fig.add_gridspec(n_topo_rows, n_topo_cols + 1,
                           width_ratios=[1.0] * n_topo_cols + [heatmap_ratio], wspace=0.3)

    # valid_channels masking is the same for every unit (only depends on the channel, not
    # q), so the NaN pattern — and therefore pos2d[valid] — is identical across all Q
    # topomaps; triangulate once instead of once per unit.
    valid = ~np.isnan(attn_masked[sorted_ord[0]]) if Q else None
    triang = build_triangulation(pos2d[valid]) if valid is not None else None

    for i, q_orig in enumerate(sorted_ord):
        row, col = divmod(i, n_topo_cols)
        ax = fig.add_subplot(gs[row, col])
        im = draw_topomap(ax, pos2d[valid], attn_masked[q_orig][valid], cmap='viridis', triang=triang)
        color = unit_colors[q_orig] if unit_colors else 'black'
        ax.set_title(f'{unit_label} {q_orig}  ({importance[q_orig]:.3f})', fontsize=9, fontweight='bold', color=color)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    for i in range(Q, n_topo_rows * n_topo_cols):
        row, col = divmod(i, n_topo_cols)
        fig.add_subplot(gs[row, col]).axis('off')

    ax_big = fig.add_subplot(gs[:, n_topo_cols])
    attn_ordered = attn[sorted_ord].T  # [C, Q], columns ordered to match the topomap panels
    im_big = ax_big.imshow(attn_ordered, aspect='auto', cmap='viridis')
    ax_big.set_yticks(range(C))
    ax_big.set_yticklabels(channel_names, fontsize=5)
    ax_big.set_xticks(range(Q))
    ax_big.set_xticklabels([f'{unit_label[0]}{q}\n{importance[q]:.3f}' for q in sorted_ord], fontsize=6, rotation=90)
    if unit_colors:
        for tick, q in zip(ax_big.get_xticklabels(), sorted_ord):
            tick.set_color(unit_colors[q])
    ax_big.set_xlabel(f'{unit_label} (sorted by contribution)', fontsize=9)
    ax_big.set_title(f'Channel x {unit_label} Attention', fontsize=10, fontweight='bold')
    fig.colorbar(im_big, ax=ax_big, fraction=0.03, pad=0.02).set_label('Attention weight', fontsize=8)

    fig.suptitle(f"Per-{unit_label} Channel Attention ({unit_label}s sorted by contribution) — "
                 f"Sub {subject_id}, Trial {trial_idx}{epoch_tag}", fontsize=12, fontweight='bold')
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
