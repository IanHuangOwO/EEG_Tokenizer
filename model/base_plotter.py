"""BasePlotter: training-curve dashboard, rendered generically from a declarative list
of panel specs each subclass builds in plot_pretrain/plot_finetune.
See docs/adr/0004-model-plugin-base-classes.md.
"""

import os
import math

import numpy as np
import matplotlib.pyplot as plt


def _ep(series):
    return range(1, len(series) + 1)


class BasePlotter:
    """Generic training-curve dashboard renderer. Subclasses build a list of panel
    specs and call self.render(panels, filename, suptitle).

    Panel spec: {title, ylabel, series: [SeriesSpec, ...], twin: {ylabel, series} | None}
    SeriesSpec: {
      key: history key (looked up in both self.history['train'] and ['val']), or None
           if using override_train/override_val directly,
      label: legend label (default = key),
      color, style_train='-', style_val='--',
      band: bool — shade key+'_std' as a +/-1 std fill around the line,
      train_only: bool — skip the val line even if present,
      override_train / override_val: explicit value list, bypasses history lookup
                                      (e.g. a flat zero line when freeze_backbone=True),
    }
    A panel with no plottable series is left blank (axis off) instead of an empty frame.
    """

    def __init__(self, output_dir='output/visualization/training_curves'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.history = {'train': {}, 'val': {}}

    def update(self, train_metrics, val_metrics=None):
        for k, v in train_metrics.items():
            self.history['train'].setdefault(k, []).append(v)
        if val_metrics:
            for k, v in val_metrics.items():
                self.history['val'].setdefault(k, []).append(v)

    # -- helpers for building panel specs --------------------------------------

    def pool_pair_series(self, prefix, colors=('green', 'steelblue'), source='val'):
        """routed/shared metric variants, falling back to a legacy unsuffixed key
        (pre-MoE runs) — used e.g. for codebook_perplexity_routed/_shared."""
        hist = self.history[source]
        routed_key, shared_key = f'{prefix}_routed', f'{prefix}_shared'
        if routed_key in hist or shared_key in hist:
            candidates = [(routed_key, 'routed', colors[0]), (shared_key, 'shared', colors[1])]
        elif prefix in hist:
            candidates = [(prefix, '', colors[0])]
        else:
            return []

        series = []
        for key, label, color in candidates:
            vals = hist.get(key)
            if not vals:
                continue
            entry = dict(key=None, label=label or prefix, color=color, train_only=(source == 'train'))
            entry['override_train' if source == 'train' else 'override_val'] = vals
            series.append(entry)
        return series

    def router_health_series(self, prefix='', entropy_label='Router entropy (higher=balanced)'):
        """MoE router-health panel shape shared by every router in this codebase (Expert,
        Filter, FFN, ...): router_entropy + gate_entropy on the main axis, router_load_std
        + lb_loss on a twin axis. History keys are unprefixed for the bare 'router' (MeFSQ's
        Expert router: 'router_entropy', 'lb_loss', ...) or f'{prefix}_<key>' otherwise
        (MeSAE's 'stamp'/'ffn' routers: 'stamp_router_entropy', 'ffn_lb_loss', ...).
        Returns (main_series, twin_series) — pass twin_series into a panel's twin dict
        only if non-empty, same guard every call site already used by hand."""
        def k(key):
            return f'{prefix}_{key}' if prefix else key

        main = [(k('router_entropy'), 'teal', entropy_label),
                (k('gate_entropy'), 'darkgoldenrod', 'Gate entropy (softmax weight)')]
        twin = [(k('router_load_std'), 'salmon', '--', 'Load std'),
                (k('lb_loss'), 'peru', ':', 'LB loss (1.0=uniform)')]

        main_series = [dict(key=key, color=c, label=l, train_only=True)
                       for key, c, l in main if key in self.history['train']]
        twin_series = [dict(key=key, color=c, style_train=ls, label=l, train_only=True)
                       for key, c, ls, l in twin if key in self.history['train']]
        return main_series, twin_series

    def indexed_series(self, prefix, cmap_name=None, train_only=True):
        """Unbounded family of series sharing a key prefix, sorted by trailing index
        (e.g. skip_gate_0, skip_gate_1, ...). train_only=False also plots the val curve
        for each key, when val history for that key exists."""
        hist = self.history['train']
        keys = sorted((k for k in hist if k.startswith(prefix)),
                      key=lambda k: int(k.rsplit('_', 1)[1]))
        n = len(keys)
        cmap = plt.get_cmap(cmap_name) if cmap_name else None
        series = []
        for i, k in enumerate(keys):
            color = cmap(i / max(n - 1, 1)) if cmap else f'C{i % 10}'
            series.append(dict(key=k, color=color, label=k, train_only=train_only))
        return series

    # -- rendering ---------------------------------------------------------------

    def render(self, panels, filename, suptitle=None, ncols=3):
        if 'loss' not in self.history['train'] or not self.history['train']['loss']:
            return

        n = len(panels)
        ncols = min(ncols, n) or 1
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows), constrained_layout=True)
        if suptitle:
            fig.suptitle(suptitle, fontsize=14, fontweight='bold')
        axes = np.atleast_1d(axes).flatten()

        for ax, panel in zip(axes, panels):
            self._render_panel(ax, panel)
        for ax in axes[n:]:
            ax.axis('off')

        fig.savefig(os.path.join(self.output_dir, filename), dpi=110, bbox_inches='tight')
        plt.close(fig)

    def _render_panel(self, ax, panel):
        plotted = False
        for s in panel.get('series', []):
            plotted |= self._plot_one_series(ax, s)

        twin = panel.get('twin')
        if twin:
            ax2 = ax.twinx()
            twin_plotted = False
            for s in twin.get('series', []):
                twin_plotted |= self._plot_one_series(ax2, s)
            if twin_plotted:
                ax2.set_ylabel(twin.get('ylabel', ''))
                self._merge_legends(ax, ax2)
            plotted = plotted or twin_plotted
        elif plotted:
            ax.legend(fontsize='x-small')

        if plotted:
            ax.set_title(panel.get('title', ''))
            ax.set_ylabel(panel.get('ylabel', ''))
            ax.set_xlabel('Epoch')
            ax.grid(True)
        else:
            ax.axis('off')

    def _plot_one_series(self, ax, s):
        key = s.get('key')
        tr = s.get('override_train') if s.get('override_train') is not None \
            else (self.history['train'].get(key) if key else None)
        va = None if s.get('train_only') else (
            s.get('override_val') if s.get('override_val') is not None
            else (self.history['val'].get(key) if key else None))
        color = s.get('color', 'C0')
        label = s.get('label', key)
        plotted = False

        if tr:
            ax.plot(_ep(tr), tr, color=color, ls=s.get('style_train', '-'),
                    label=f"{label} (train)" if va else label)
            plotted = True
            if s.get('band') and key:
                std = self.history['train'].get(f'{key}_std')
                if std:
                    lo = [m - sd for m, sd in zip(tr, std)]
                    hi = [m + sd for m, sd in zip(tr, std)]
                    ax.fill_between(_ep(tr), lo, hi, color=color, alpha=0.15)

        if va:
            ax.plot(_ep(va), va, color=color, ls=s.get('style_val', '--'), alpha=0.7,
                    label=f"{label} (val)")
            plotted = True
            if s.get('band') and key:
                std = self.history['val'].get(f'{key}_std')
                if std:
                    lo = [m - sd for m, sd in zip(va, std)]
                    hi = [m + sd for m, sd in zip(va, std)]
                    ax.fill_between(_ep(va), lo, hi, color=color, alpha=0.08)

        return plotted

    @staticmethod
    def _merge_legends(ax1, ax2):
        l1, n1 = ax1.get_legend_handles_labels()
        l2, n2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, n1 + n2, loc='upper left', fontsize='x-small')

    def plot_pretrain(self, filename='training_dashboard.png'):
        raise NotImplementedError

    def plot_finetune(self, filename='training_dashboard.png', freeze_backbone=False):
        raise NotImplementedError
