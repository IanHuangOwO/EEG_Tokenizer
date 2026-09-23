"""Grouped bar chart rendering for panel_head_dataset_bars.py's cross-head x
cross-dataset accuracy/kappa comparison, and panel_protocol_gap.py's intra-vs-LOSO delta
view. Pure rendering -- tools/analysis/head_dataset_matrix.py does the numbers."""
import numpy as np
import matplotlib.pyplot as plt


def plot_grouped_bars(out_path, group_labels, series_labels, means, stds, ns, ylabel, title='',
                       gap_after=()):
    """One bar cluster per group_label (e.g. a dataset_mode), one color per series_label
    (e.g. a head). means/stds/ns: [n_series][n_groups] -- NaN mean = skip that bar
    (missing data, e.g. a head never run against that dataset), leaving a gap instead of
    drawing a misleading 0. n is annotated above every bar: fold counts vary 5-16x across
    this repo's real datasets (5-fold intra CV vs a subsampled LOSO with as few as 5
    folds), so an error bar alone would let two very differently-powered comparisons
    look equally confident.

    gap_after: series indices after which a wider gap is inserted (e.g. {0} to set the
    first series -- typically raw_signal, see tools/analysis/head_dataset_matrix.py's
    order_heads -- visually apart from the rest of the cluster). Every other adjacent
    pair keeps the normal spacing."""
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    ns = np.asarray(ns)
    n_series, n_groups = means.shape
    x = np.arange(n_groups)
    slot = 0.8 / max(n_series, 1)
    width = slot * 0.5  # bar drawn narrower than its slot -- a visible gap between
    # adjacent heads' bars within one dataset cluster, not just between clusters

    # Cumulative slot centers, with an extra half-slot inserted after each gap_after
    # index -- uniform spacing everywhere else.
    centers = np.zeros(n_series)
    for i in range(1, n_series):
        centers[i] = centers[i - 1] + slot + (slot * 0.5 if (i - 1) in gap_after else 0.0)
    centers -= centers.mean()

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * n_groups), 5))
    for i, label in enumerate(series_labels):
        pos = x + centers[i]
        valid = ~np.isnan(means[i])
        if not valid.any():
            continue
        ax.bar(pos[valid], means[i][valid], width=width, yerr=stds[i][valid],
               capsize=3, label=label)
        for xp, m, n in zip(pos[valid], means[i][valid], ns[i][valid]):
            ax.annotate(f'n={n}', (xp, m), textcoords='offset points', xytext=(0, 4),
                        ha='center', fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
