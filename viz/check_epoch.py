"""
Training-epoch snapshot: raw vs recon topomap + head activation grid.
"""

import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from viz import (
    load_config,
    load_model,
    resolve_output_dir,
    select_subject_dataset,
    filter_config_to_subject,
    pick_trial,
)
from viz.draw import project_coords_2d, draw_topomap, extract_head_psd
from viz.compute import _run_reconstruction
from viz.train import visualize_reconstruction


# ── Module interface ──────────────────────────────────────────────────────────

def add_args(parser):
    parser.add_argument('--recon_max_heads', type=int,  default=None)
    parser.add_argument('--recon_cmap',      type=str,  default='YlOrRd')


def run(config, output_dir, args, model=None, dataset=None,
        trial_idx=None, subject_id=None, epoch=None):

    viz_dir = os.path.join(output_dir, 'recon')
    os.makedirs(viz_dir, exist_ok=True)
    epoch_tag = f'_ep{epoch:04d}' if epoch is not None else ''

    recon_cfg = config.get('check', {}).get('reconstruction', {})
    cmap      = getattr(args, 'recon_cmap', None) or recon_cfg.get('cmap', 'YlOrRd')
    topo_cmap = config.get('check', {}).get('topomap', {}).get('cmap', 'YlOrRd')

    device = next(model.parameters()).device

    x_patches, coords, mask, time_indices, _, _ = dataset[trial_idx]
    x_in = x_patches.unsqueeze(0).to(device)
    c_in = coords.unsqueeze(0).to(device)
    t_in = time_indices.unsqueeze(0).to(device)

    data  = _run_reconstruction(model, dataset, trial_idx, device)
    pos2d = project_coords_2d(coords.numpy())

    # raw vs recon time-series
    raw_t      = torch.from_numpy(data['raw']).unsqueeze(0)
    recon_t    = torch.from_numpy(data['recon']).unsqueeze(0)
    C, N, _ = x_patches.shape
    mask_np   = mask.numpy().reshape(C, N)  # [C*N] → [C, N]
    patch_len = x_patches.shape[-1]         # L
    visualize_reconstruction(
        None, (raw_t, recon_t),
        epoch,
        output_dir=viz_dir,
        channel_names=dataset.base_dataset.channel_names,
        subject_id=subject_id, trial_idx=trial_idx,
        mask=mask_np, patch_len=patch_len,
    )

    # head activation grid
    try:
        stage_psd, head_norms, head_affinity, routing_score = extract_head_psd(model, x_in, c_in, t_in)
        psd_ch_h   = stage_psd[0]   # [C, H]
        H          = psd_ch_h.shape[1]
        sorted_ord = np.argsort(routing_score)[::-1]
        score_ranked = routing_score[sorted_ord]

        n_tc = min(16, H)
        n_tr = math.ceil(H / n_tc)
        fig_w = max(28.0, n_tc * 2.8)
        fig_h = max(9.0, 3.8 + n_tr * 3.0)

        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
        fig.suptitle(
            f"Head Routing — Sub {subject_id}, Trial {trial_idx}",
            fontsize=13, fontweight='bold', y=0.998,
        )
        outer = gridspec.GridSpec(
            2, 1, figure=fig,
            height_ratios=[3.8, n_tr * 3.0],
            hspace=0.32,
            top=0.95, bottom=0.03, left=0.03, right=0.975,
        )

        ax_imp = fig.add_subplot(outer[0])
        rank_x = np.arange(H)
        ax_imp.plot(rank_x, score_ranked, color='steelblue', lw=1.5)
        ax_imp.scatter(rank_x, score_ranked, color='steelblue', s=10)
        ax_imp.set_xlim(-0.5, H - 0.5)
        ax_imp.set_xlabel('Head rank (0 = highest router score)', fontsize=9)
        ax_imp.set_ylabel('Mean gate logit (raw)', fontsize=9)
        ax_imp.set_title(f'Head Routing Importance — {H} heads', fontsize=9, fontweight='bold')
        ax_imp.grid(alpha=0.3)

        gs_topo = gridspec.GridSpecFromSubplotSpec(
            n_tr, n_tc, subplot_spec=outer[1], hspace=0.42, wspace=0.05,
        )
        for pos, h_orig in enumerate(sorted_ord):
            r, c = divmod(pos, n_tc)
            ax_h = fig.add_subplot(gs_topo[r, c])
            p_h  = psd_ch_h[:, h_orig]
            draw_topomap(ax_h, pos2d, p_h, cmap=cmap, vmin=p_h.min(), vmax=p_h.max())
            ax_h.set_title(f'H{h_orig}\n{routing_score[h_orig]:.3f}', fontsize=5.5)

        for pos in range(H, n_tr * n_tc):
            r, c = divmod(pos, n_tc)
            fig.add_subplot(gs_topo[r, c]).axis('off')

        out_path = os.path.join(
            viz_dir,
            f"sub{subject_id}_trial{trial_idx}{epoch_tag}_topo_heads.png",
        )
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  [epoch] -> {out_path}")
    except Exception as e:
        print(f"  [epoch] head topo failed: {e}")

    # Raw vs recon power topomap
    try:
        raw_power   = (data['raw']   ** 2).mean(axis=-1)
        recon_power = (data['recon'] ** 2).mean(axis=-1)
        fig2, axes2 = plt.subplots(1, 2, figsize=(8, 4))
        fig2.suptitle(f"Power Topo — Sub {subject_id}, Trial {trial_idx}{epoch_tag}",
                      fontsize=10, fontweight='bold')
        for ax, power, title in zip(axes2, [raw_power, recon_power], ['Raw', 'Recon']):
            im = draw_topomap(ax, pos2d, power, cmap=topo_cmap)
            ax.set_title(title, fontsize=9)
            fig2.colorbar(im, ax=ax, fraction=0.05, pad=0.02).set_label('Power', fontsize=6)
        out2 = os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_power_topo.png")
        fig2.savefig(out2, dpi=120, bbox_inches='tight')
        plt.close(fig2)
    except Exception as e:
        print(f"  [epoch] power topo failed: {e}")


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    from IO.dataset import build_dataset_from_config

    parser = argparse.ArgumentParser(description='EEG Reconstruction + Topomap')
    parser.add_argument('--config',     default='config/analysis.json')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--subject',    type=int, default=None)
    parser.add_argument('--trial',      type=int, default=None)
    add_args(parser)
    args = parser.parse_args()

    cfg        = load_config(args.config)
    checkpoint = args.checkpoint or cfg.get('checkpoint', '')
    ds_name, subject = select_subject_dataset(cfg, args.subject)
    filtered = filter_config_to_subject(cfg, ds_name, subject)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl      = load_model(filtered, checkpoint, device)
    ds       = build_dataset_from_config(filtered, mode='pretrain')
    trial_cfg = args.trial if args.trial is not None else cfg['dataset_params']['pretrain'][ds_name].get('trial_to_use')
    t_idx, subject_id = pick_trial(ds, subject, trial_cfg)
    out      = resolve_output_dir(filtered, 'check')
    run(filtered, out, args, model=mdl, dataset=ds,
        trial_idx=t_idx, subject_id=subject_id)
