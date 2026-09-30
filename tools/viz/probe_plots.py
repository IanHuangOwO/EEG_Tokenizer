"""Rendering for tools/analysis/probe_maps.py: where a linear probe reads from."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_probe_maps(out_path, maps, channel_names, title, event_s=None):
    """maps: {row title: (importance [K, N], spatial [K, C], t_axis [N] seconds of each patch centre)} -- each row
    has its own time axis (groups may differ in patch length / stride); the time columns share one x range.
    Left: virtual channel (ranked by importance) x time (event at 0, dashed); right: each virtual channel's
    spatial filter over electrodes."""
    fig, axes = plt.subplots(len(maps), 2, figsize=(15, 2.8 * len(maps)), squeeze=False,
                             gridspec_kw={'width_ratios': [1.3, 1]})
    for ax_t in axes[1:, 0]:
        ax_t.sharex(axes[0, 0])
    for (ax_t, ax_s), (name, (imp, sp, t_axis)) in zip(axes, maps.items()):
        imp, sp, t_axis = np.asarray(imp), np.asarray(sp), np.asarray(t_axis)
        dt = (t_axis[1] - t_axis[0]) if len(t_axis) > 1 else 1.0
        K = imp.shape[0]
        im = ax_t.imshow(imp, aspect='auto', cmap='viridis', origin='lower',
                         extent=[t_axis[0] - dt / 2, t_axis[-1] + dt / 2, -0.5, K - 0.5])
        if event_s is not None:
            ax_t.axvline(0.0, color='crimson', ls='--', lw=1.5)   # visible on the map and on the blank margin
            ax_t.text(0.0, K - 0.6, ' event', color='crimson', fontsize=8, va='top')
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


def plot_stamp_maps(out_path, tw, imp, chan, t_axis, stamp_labels, xy, channel_names, title, event_s=None, top=8,
                    z_maps=None):
    """Stamp-power half of a head. Left: per-stamp time weights (rows = stamps ranked by importance, label = stamp,
    template peak and importance share); right: channel map of the `top` most important stamps on the scalp.
    z_maps: optional (importance [K, N], spatial [K, C]) of the same head's latent_signed entry, drawn as a top row
    (virtual channel x time, spatial filters) on the same time axis."""
    tw, imp, chan, t_axis, xy = (np.asarray(v) for v in (tw, imp, chan, t_axis, xy))
    order = np.argsort(-imp)
    share = imp / imp.sum()
    ncol = 4
    z0 = 0 if z_maps is None else 1
    fig = plt.figure(figsize=(16, 6 + 2.6 * z0))
    gs = fig.add_gridspec(2 + z0, 2 + ncol, width_ratios=[2.2] + [1] * ncol + [0.08], height_ratios=[1.1] * z0 + [1, 1])
    dt = (t_axis[1] - t_axis[0]) if len(t_axis) > 1 else 1.0
    ax = fig.add_subplot(gs[z0:, 0])
    if z_maps is not None:
        zi, zs = (np.asarray(v) for v in z_maps)
        K = zi.shape[0]
        az = fig.add_subplot(gs[0, 0], sharex=ax)
        zshare = zi.sum(1) / zi.sum()                            # rows come ranked (v1 = most important)
        vlab = [f'v{k + 1} ({zshare[k]:.0%})' for k in range(K)]
        imz = az.imshow(zi, aspect='auto', cmap='viridis', origin='upper',
                        extent=[t_axis[0] - dt / 2, t_axis[-1] + dt / 2, K - 0.5, -0.5])
        az.set_yticks(range(K), vlab, fontsize=7)
        az.set_title('z half (latent_signed): decision weight per virtual channel', fontsize=9)
        if event_s is not None:
            az.axvline(0.0, color='crimson', ls='--', lw=1.5)
        fig.colorbar(imz, ax=az, fraction=0.03)
        asp = fig.add_subplot(gs[0, 1:])
        lim = np.abs(zs).max() or 1.0
        ims = asp.imshow(zs, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim, origin='upper')
        asp.set_xticks(range(len(channel_names)), channel_names, rotation=90, fontsize=6)
        asp.set_yticks(range(K), vlab, fontsize=7)
        asp.set_title('z half: spatial filter per virtual channel (sign-aligned)', fontsize=9)
        fig.colorbar(ims, ax=asp, fraction=0.03)
    im = ax.imshow(tw[order], aspect='auto', cmap='viridis', origin='upper',
                   extent=[t_axis[0] - dt / 2, t_axis[-1] + dt / 2, len(order) - 0.5, -0.5])
    ax.set_yticks(range(len(order)), [f'{stamp_labels[s]} ({share[s]:.0%})' for s in order], fontsize=7)
    if event_s is not None:
        ax.axvline(0.0, color='crimson', ls='--', lw=1.5)
    ax.set_xlabel('time from event (s)' if event_s is not None else 'time in window (s)')
    ax.set_title('stamp half (stamp_power): time weight per stamp, ranked by importance', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03)
    shown = order[:min(top, 2 * ncol)]
    vmax = max(float(chan[shown].max()), 1e-12)                  # one colour scale for every scalp map
    for i, s in enumerate(shown):
        a = fig.add_subplot(gs[z0 + i // ncol, 1 + i % ncol])
        sc = a.scatter(xy[:, 0], xy[:, 1], c=chan[s], cmap='Reds', vmin=0, vmax=vmax, s=90,
                       edgecolors='k', linewidths=0.3)
        if len(channel_names) <= 8:
            for (x, y), n in zip(xy, channel_names):
                a.annotate(n, (x, y), fontsize=6, ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
        a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
        pad = 0.02
        a.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad); a.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
        a.set_title(f'{stamp_labels[s]} ({share[s]:.0%})', fontsize=8)
    fig.colorbar(sc, cax=fig.add_subplot(gs[z0:, -1]), label='channel weight (per head, each stamp max = 1)')
    fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
