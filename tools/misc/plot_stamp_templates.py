"""One-off: what do a checkpoint's stamp atoms actually look like? Plots each alive
stamp's D_i (free waveform, StampBank.fingerprint()) and its quadrature partner H_i
(_template_tables()[1]) overlaid -- these are the raw dictionary shapes a stamp's (a, b)
amplitude scales/mixes, no reconstruction or real trial involved (see
model/MeSAE/MeSAE_modules.py's StampBank class docstring: amp never touches shape, only
scale/sign). Capped to the top N alive stamps by fire_ema (dead atoms are noise, not
signal) -- pass --all for every stamp instead (large grid).

Usage: python -m tools.misc.plot_stamp_templates --checkpoint <path> [--base-config <path>]
  [--top-n 40] [--all] [--out <path.png>]
(run with -m from the repo root -- it's a nested module, plain `python
tools/misc/plot_stamp_templates.py` can't resolve the `tools.analysis` import)
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from tools.analysis import _deep_merge, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--base-config', dest='base_config', default=None,
                     help='defaults to <checkpoint dir>/../artifacts/config.json')
    ap.add_argument('--top-n', type=int, default=40)
    ap.add_argument('--all', action='store_true', help='plot every stamp, ignore --top-n')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    base_config = args.base_config or os.path.join(
        os.path.dirname(os.path.dirname(args.checkpoint)), 'artifacts', 'config.json')
    with open(base_config) as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(config, args.checkpoint, device, mode='pretrain')
    model.eval()

    stamps = model.stamps
    D_all, H_all = stamps._template_tables()  # each [n_stamps, patch_len], unit-normalized
    D_all, H_all = D_all.detach().cpu().numpy(), H_all.detach().cpu().numpy()
    n_stamps, patch_len = D_all.shape
    n_routed = stamps.n_routed

    fire_ema = stamps.fire_ema.detach().cpu().numpy()  # [n_routed] only, routed stamps' liveness
    alive_routed = np.flatnonzero(fire_ema >= stamps.dead_threshold)
    routed_order = alive_routed[np.argsort(-fire_ema[alive_routed])]
    shared_ids = np.arange(n_routed, n_stamps)  # always-on -- no fire_ema/dead concept applies

    if args.all:
        ids = np.concatenate([shared_ids, routed_order])
    else:
        # Shared stamps ALWAYS included, never counted against --top-n's routed budget --
        # there are only n_shared_stamps of them (4 by default) and they're structurally
        # different (always-on, not competitively selected), not just "high fire_ema"
        # routed atoms that happened to rank first.
        ids = np.concatenate([shared_ids, routed_order[:args.top_n]])

    n = len(ids)
    ncols = 8
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 1.6 * nrows), squeeze=False)
    t = np.arange(patch_len)
    for i, sid in enumerate(ids):
        ax = axes[i // ncols][i % ncols]
        ax.plot(t, D_all[sid], color='steelblue', lw=1.2, label='D_i')
        ax.plot(t, H_all[sid], color='crimson', lw=1.0, ls='--', label='H_i')
        is_shared = sid >= n_routed
        ax.set_title(f'#{sid} ({"shared" if is_shared else "routed"})', fontsize=7,
                     color='darkgreen' if is_shared else 'black', fontweight='bold' if is_shared else 'normal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.axhline(0, color='gray', lw=0.4)
        if is_shared:
            for spine in ax.spines.values():
                spine.set_edgecolor('darkgreen')
                spine.set_linewidth(2)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    axes[0][0].legend(fontsize=6, loc='upper right')
    fig.suptitle(f'Stamp dictionary atoms (D_i solid, H_i quadrature dashed) -- {n}/{n_stamps} shown',
                 fontweight='bold')
    fig.tight_layout()

    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(args.checkpoint)), 'analysis', 'stamp_templates.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  -> {out}")


if __name__ == '__main__':
    main()
