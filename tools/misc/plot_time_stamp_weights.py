"""One-off: LearnedTimePool's learned weight map w[s, n] = softmax_n(sum_r p[r,s] q[r,n])
(model/MeSAE/MeSAE_modules.py's LearnedTimePool, ADR 0014 C1) -- which patch/time
positions each stamp actually pools from, per dataset. Only heads with
time_pool='learned' have this (learned/learned_advance/learned_evoked here); for each
dataset, picks whichever of those has the best mean_tail accuracy
(tools/analysis/head_dataset_matrix.py's build_matrix, same source
panel_head_dataset_bars.py uses), then averages that head's w across every
subject/fold's own head.pth under its finetune/run_*/ dirs -- one weight map per subject
(each fold trains its own LearnedTimePool from scratch), so "average across subjects" is
a literal mean of the post-softmax weight maps, not an average of the raw p/q logits.

Usage: python -m tools.misc.plot_time_stamp_weights --group-eval '<glob>' [--out-dir <dir>]
  glob e.g. 'output/mesae_v10_small/finetune/*/*/artifacts/group_eval.json'
"""
import argparse
import glob as globmod
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from tools.analysis import event_onset_patch
from tools.analysis.group_summary import _locate
from tools.analysis.head_dataset_matrix import build_matrix

LEARNED_HEADS = ('learned', 'learned_advance', 'learned_evoked')


def _time_pool_weights(head_pth):
    ckpt = torch.load(head_pth, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    if 'time.p' not in sd or 'time.q' not in sd:
        return None
    p, q = sd['time.p'], sd['time.q']
    return torch.softmax(torch.einsum('rs,rn->sn', p, q), dim=-1).numpy()  # [S, N]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group-eval', required=True)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    paths = sorted(globmod.glob(args.group_eval))
    if not paths:
        raise SystemExit(f"no group_eval.json matched {args.group_eval!r}")
    backbone_dir = _locate(paths[0])[0]
    out_dir = args.out_dir or os.path.join(backbone_dir, 'finetune', 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    matrix = build_matrix(paths, metric='tail')  # {(head, dataset_mode): (mean, std, n)}
    datasets = sorted({d for h, d in matrix if h in LEARNED_HEADS})

    for dataset_mode in datasets:
        candidates = [(h, matrix[(h, dataset_mode)][0]) for h in LEARNED_HEADS
                      if (h, dataset_mode) in matrix and not np.isnan(matrix[(h, dataset_mode)][0])]
        if not candidates:
            continue
        best_head, best_acc = max(candidates, key=lambda hc: hc[1])

        run_dir = os.path.join(backbone_dir, 'finetune', best_head, dataset_mode, 'finetune')
        head_pths = sorted(globmod.glob(os.path.join(run_dir, 'run_*', 'head.pth')))
        maps = [w for w in (_time_pool_weights(p) for p in head_pths) if w is not None]
        if not maps:
            print(f"  [skip] {dataset_mode}: best head {best_head} has no time.p/time.q "
                  f"(shouldn't happen for a 'learned' head -- check head_config)")
            continue
        shape = maps[0].shape
        maps = [m for m in maps if m.shape == shape]  # guard a stale mismatched checkpoint
        avg = np.mean(maps, axis=0)  # [S, N]

        fig, ax = plt.subplots(figsize=(max(6, avg.shape[1] * 0.2), max(4, avg.shape[0] * 0.15)))
        im = ax.imshow(avg, aspect='auto', cmap='viridis', origin='lower')
        # Event onset line: patch settings from this run's own config snapshot, dataset from
        # the current datas/finetune/<name> (older snapshots carry pre-rename names/paths,
        # e.g. BCICIV2a at datas/BCICIV2a).
        cfgs = sorted(globmod.glob(os.path.join(backbone_dir, 'finetune', best_head, dataset_mode,
                                                'artifacts', 'config_*.json')))
        ds_name = dataset_mode.rsplit('_', 1)[0]
        pp = json.load(open(cfgs[-1])).get('preprocess_params', {}) if cfgs else {}
        ev = event_onset_patch({'preprocess_params': pp, 'dataset_params': {'finetune': {
            ds_name: {'dataset_path': os.path.join('datas', 'finetune', ds_name)}}}}, ds_name)
        if ev is not None:
            ax.axvline(ev, color='w', ls='--', lw=1.2, label='event onset')
            ax.legend(loc='upper right', fontsize=7)
        ax.set_xlabel('Patch (time) position')
        ax.set_ylabel('Stamp index')
        ax.set_title(f'{dataset_mode}: LearnedTimePool weights, {best_head} '
                      f'(acc={best_acc:.3f}, avg over {len(maps)} subject/fold checkpoints)',
                      fontsize=10, fontweight='bold')
        fig.colorbar(im, ax=ax, label='softmax weight')
        fig.tight_layout()

        out_path = os.path.join(out_dir, f'time_stamp_weights_{dataset_mode}.png')
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"  -> {out_path}  (best={best_head}, acc={best_acc:.3f}, n={len(maps)})")


if __name__ == '__main__':
    main()
