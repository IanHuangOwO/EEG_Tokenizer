"""One-off: are a checkpoint's stamps duplicates of each other, up to phase?

Each stamp reconstructs a*D_i + b*H_i (H_i = Hilbert partner of D_i, see
model/MeSAE/MeSAE_modules.py's StampBank._quadrature), so stamp j can present stamp i's
waveform at ANY phase. Plain cosine <D_i, D_j> misses a phase-shifted copy; the right
similarity is how much of D_i lies in stamp j's (D_j, H_j) plane:

    sim(i, j) = sqrt(<D_i, D_j>^2 + <D_i, H_j>^2)      in [0, 1], 1 = same atom at some phase

(symmetrized by max(sim, sim.T) -- DC/Nyquist zeroing makes H only approximately
unitary). Optionally also measures each stamp's mean amplitude sqrt(a^2+b^2) on real
trials (tools/analysis/stamp_dist.py's accumulate_stamp_ab), so a weak copy of a strong
stamp shows up as such.

Writes <checkpoint dir>/../analysis/stamp_duplicates.png (similarity heatmap) and prints
every pair above --threshold, plus each stamp's nearest neighbour.

Usage: python -m tools.misc.stamp_duplicates --checkpoint <path> [--base-config <path>]
  [--threshold 0.9] [--dataset <name in dataset_params.pretrain> --n-subjects 3
  --max-trials 30] [--device cpu] [--out <path.png>]
"""
import argparse
import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from tools.analysis import load_model


def _mean_amp(model, config, dataset_name, n_subjects, max_trials, device, n_stamps):
    from IO.dataset import build_dataset_from_config
    from tools.analysis.stamp_dist import accumulate_stamp_ab

    ds_args = config['dataset_params']['pretrain'][dataset_name]
    meta = json.load(open(os.path.join(ds_args['dataset_path'], 'metadata.json')))
    subjects = sorted(meta['data_structure'], key=lambda s: (len(s), s))[:n_subjects]
    cfg = copy.deepcopy(config)
    cfg['dataset_params']['pretrain'] = {dataset_name: {**ds_args, 'subject_to_use': subjects}}
    dataset = build_dataset_from_config(cfg, mode='pretrain')
    trials = list(range(min(max_trials, len(dataset.base_dataset))))
    ab = accumulate_stamp_ab(model, dataset, trials, device, max_stamps=n_stamps)
    amp = np.full(n_stamps, np.nan)
    for sid, (a, b) in ab.items():
        amp[sid] = float(np.hypot(a, b).mean())
    return amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--base-config', dest='base_config', default=None,
                    help='defaults to <checkpoint dir>/../artifacts/config.json')
    ap.add_argument('--threshold', type=float, default=0.9)
    ap.add_argument('--dataset', default=None, help='pretrain dataset to measure amplitude on (optional)')
    ap.add_argument('--n-subjects', type=int, default=3, dest='n_subjects')
    ap.add_argument('--max-trials', type=int, default=30, dest='max_trials')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    run_dir = os.path.dirname(os.path.dirname(args.checkpoint))
    config = json.load(open(args.base_config or os.path.join(run_dir, 'artifacts', 'config.json')))
    device = torch.device(args.device)
    model = load_model(config, args.checkpoint, device, mode='pretrain').eval()

    stamps = model.stamps
    D, H = (t.detach().cpu().numpy() for t in stamps._template_tables())  # [n_stamps, L] each
    n_stamps, n_routed = D.shape[0], stamps.n_routed
    fire_ema = stamps.fire_ema.detach().cpu().numpy()
    ids = np.concatenate([np.flatnonzero(fire_ema >= stamps.dead_threshold),
                          np.arange(n_routed, n_stamps)])  # alive routed + every shared

    Dk, Hk = D[ids], H[ids]
    sim = np.sqrt((Dk @ Dk.T) ** 2 + (Dk @ Hk.T) ** 2)
    sim = np.maximum(sim, sim.T)
    np.fill_diagonal(sim, np.nan)

    amp = None
    if args.dataset:
        amp = _mean_amp(model, config, args.dataset, args.n_subjects, args.max_trials, device, n_stamps)[ids]

    def label(k):
        kind = 's' if ids[k] >= n_routed else 'r'
        return f"#{ids[k]}{kind}" + (f" (amp {amp[k]:.3g})" if amp is not None else "")

    print(f"{len(ids)} stamps ({n_routed} routed slots, {n_stamps - n_routed} shared), "
          f"threshold {args.threshold}")
    iu = np.triu_indices(len(ids), k=1)
    pairs = sorted(((sim[i, j], i, j) for i, j in zip(*iu) if sim[i, j] >= args.threshold), reverse=True)
    print(f"\n{len(pairs)} pair(s) with sim >= {args.threshold}:")
    for s, i, j in pairs:
        print(f"  {s:.3f}  {label(i)}  <->  {label(j)}")
    print("\nnearest neighbour per stamp:")
    for k in range(len(ids)):
        j = int(np.nanargmax(sim[k]))
        print(f"  {label(k):>24}  ->  {label(j):<24} sim {sim[k, j]:.3f}")

    fig, ax = plt.subplots(figsize=(max(6, 0.3 * len(ids)), max(5, 0.3 * len(ids))))
    im = ax.imshow(sim, vmin=0, vmax=1, cmap='magma')
    ticks = [f"{ids[k]}" + ('s' if ids[k] >= n_routed else '') for k in range(len(ids))]
    ax.set_xticks(range(len(ids)), ticks, rotation=90, fontsize=6)
    ax.set_yticks(range(len(ids)), ticks, fontsize=6)
    ax.set_title(f'Phase-invariant stamp similarity sqrt(<Di,Dj>^2+<Di,Hj>^2)\n'
                 f'{len(pairs)} pair(s) >= {args.threshold}', fontsize=10, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = args.out or os.path.join(run_dir, 'analysis', 'stamp_duplicates.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\n  -> {out}")


if __name__ == '__main__':
    main()
