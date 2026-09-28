"""Rendering for tools/analysis/probe_maps.py: where a linear probe reads from."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_probe_maps(out_path, maps, channel_names, title, t_axis, event_s=None):
    """maps: {row title: (importance [K, N], spatial [K, C])}. Left: virtual channel (ranked by importance)
    x time in seconds (event at 0, dashed); right: each virtual channel's spatial filter over electrodes."""
    fig, axes = plt.subplots(len(maps), 2, figsize=(15, 2.8 * len(maps)), squeeze=False,
                             gridspec_kw={'width_ratios': [1.3, 1]})
    dt = (t_axis[1] - t_axis[0]) if len(t_axis) > 1 else 1.0
    for (ax_t, ax_s), (name, (imp, sp)) in zip(axes, maps.items()):
        imp, sp = np.asarray(imp), np.asarray(sp)
        K = imp.shape[0]
        im = ax_t.imshow(imp, aspect='auto', cmap='viridis', origin='lower',
                         extent=[t_axis[0] - dt / 2, t_axis[-1] + dt / 2, -0.5, K - 0.5])
        if event_s is not None:
            ax_t.axvline(0.0, color='w', ls='--', lw=1.5)
            ax_t.text(0.0, K - 0.6, ' event', color='w', fontsize=8, va='top')
        ax_t.set_yticks(range(K), [f'v{k + 1}' for k in range(K)], fontsize=7)
        ax_t.set_ylabel('virtual channel\n(ranked)')
        ax_t.set_title(f'{name}: decision weight over time', fontsize=9)
        fig.colorbar(im, ax=ax_t, fraction=0.03)
        lim = np.abs(sp).max() or 1.0
        im2 = ax_s.imshow(sp, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim, origin='lower')
        ax_s.set_xticks(range(len(channel_names)), channel_names, rotation=90, fontsize=6)
        ax_s.set_yticks(range(K), [f'v{k + 1}' for k in range(K)], fontsize=7)
        ax_s.set_title('spatial filter per virtual channel (sign-aligned)', fontsize=9)
        fig.colorbar(im2, ax=ax_s, fraction=0.03)
    axes[-1, 0].set_xlabel('time from event (s)' if event_s is not None else 'time in window (s)')
    fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
