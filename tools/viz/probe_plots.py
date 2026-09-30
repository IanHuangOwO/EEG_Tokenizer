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


def plot_stamp_maps(out_path, tw, imp, sp, t_axis, stamp_labels, channel_names, title, event_s=None, z_maps=None):
    """Stamp-power half of a head, laid out like probe_maps. Left: per-stamp time weights (rows = stamps ranked by
    decision importance, label = stamp, template peak and importance share); right: the entry's spatial filter per
    virtual channel (ranked, sign-aligned). z_maps: optional (importance [K, N], spatial [K, C]) of the same head's
    latent_signed entry, drawn as a top row on the same time axis."""
    tw, imp, sp, t_axis = (np.asarray(v) for v in (tw, imp, sp, t_axis))
    order = np.argsort(-imp)
    share = imp / imp.sum()
    rows = [('z', z_maps)] if z_maps is not None else []
    rows.append(('stamp', None))
    fig, axes = plt.subplots(len(rows), 2, figsize=(15, 2.8 + 5.2 * (len(rows) - 1) if len(rows) > 1 else 5.2),
                             squeeze=False, gridspec_kw={'width_ratios': [1.3, 1],
                                                         'height_ratios': [1.1] * (len(rows) - 1) + [2]})
    dt = (t_axis[1] - t_axis[0]) if len(t_axis) > 1 else 1.0
    ext = [t_axis[0] - dt / 2, t_axis[-1] + dt / 2]

    def spatial(ax_s, m, what):
        lim = np.abs(m).max() or 1.0
        im = ax_s.imshow(m, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim, origin='lower')
        ax_s.set_xticks(range(len(channel_names)), channel_names, rotation=90, fontsize=6)
        ax_s.set_yticks(range(m.shape[0]), [f'v{k + 1}' for k in range(m.shape[0])], fontsize=7)
        ax_s.set_title(f'{what}: spatial filter per virtual channel (ranked, sign-aligned)', fontsize=9)
        fig.colorbar(im, ax=ax_s, fraction=0.03)

    for (ax_t, ax_s), (kind, zm) in zip(axes, rows):
        if kind == 'z':
            zi, zs = (np.asarray(v) for v in zm)
            K = zi.shape[0]
            im = ax_t.imshow(zi, aspect='auto', cmap='viridis', origin='lower', extent=ext + [-0.5, K - 0.5])
            ax_t.set_yticks(range(K), [f'v{k + 1}' for k in range(K)], fontsize=7)
            ax_t.set_ylabel('virtual channel\n(ranked)')
            ax_t.set_title('z half (latent_signed): decision weight over time', fontsize=9)
            spatial(ax_s, zs, 'z half')
        else:
            im = ax_t.imshow(tw[order], aspect='auto', cmap='viridis', origin='upper',
                             extent=ext + [len(order) - 0.5, -0.5])
            ax_t.set_yticks(range(len(order)), [f'{stamp_labels[s]} ({share[s]:.0%})' for s in order], fontsize=7)
            ax_t.set_title('stamp half (stamp_power): time weight per stamp, ranked by importance', fontsize=9)
            spatial(ax_s, sp, 'stamp half')
        if event_s is not None:
            ax_t.axvline(0.0, color='crimson', ls='--', lw=1.5)
        fig.colorbar(im, ax=ax_t, fraction=0.03)
    for ax_t in axes[1:, 0]:
        ax_t.sharex(axes[0, 0])
    axes[-1, 0].set_xlabel('time from event (s)' if event_s is not None else 'time in window (s)')
    fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
