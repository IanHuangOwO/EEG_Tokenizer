"""
Per-head topomap analysis: unweighted PSD, G-weighted PSD, and head interaction matrices.
Drawing primitives live in utils/topomap_helpers.py.
"""

import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from analysis import (
    load_config,
    load_model,
    resolve_output_dir,
    select_subject_dataset,
    filter_config_to_subject,
    pick_trial,
)
from utils.topomap_helpers import (
    project_coords_2d,
    save_topomap_grid,
    extract_head_psd,
)


# ── Module interface ──────────────────────────────────────────────────────────

def add_args(parser):
    parser.add_argument('--topo_max_heads', type=int,   default=None,
                        help='Max heads to display per stage (default from config)')
    parser.add_argument('--topo_n_std',     type=float, default=None,
                        help='Std multiplier for high/low importance bands')
    parser.add_argument('--topo_cmap',      type=str,   default=None,
                        help='Colormap for topomap grids')


def run(config, output_dir, args, model=None, dataset=None, trial_idx=None, subject_id=None):
    viz_dir = os.path.join(output_dir, 'topomap')
    os.makedirs(viz_dir, exist_ok=True)

    topo_cfg  = config.get('check', {}).get('topomap', {})
    max_heads = getattr(args, 'topo_max_heads', None) or topo_cfg.get('max_heads', 128)
    n_std     = getattr(args, 'topo_n_std',     None) or topo_cfg.get('n_std',     1.0)
    cmap      = getattr(args, 'topo_cmap',      None) or topo_cfg.get('cmap',      'YlOrRd')

    device = next(model.parameters()).device

    x_patches, coords, time_indices, _, _ = dataset[trial_idx]
    x_in        = x_patches.unsqueeze(0).to(device)
    coords_in   = coords.unsqueeze(0).to(device)
    time_idx_in = time_indices.unsqueeze(0).to(device)

    pos2d = project_coords_2d(coords.numpy())

    with torch.no_grad():
        _, _, _, _, _, all_head_importances, all_sim, all_G = model(x_in, coords_in, time_idx_in)

    stage_psd = extract_head_psd(model, x_in, coords_in, time_idx_in)

    for s_idx, (psd_ch_h, h_cfg) in enumerate(zip(stage_psd, model.decoder_heads_config)):
        f_min, f_max = h_cfg["freq_range"]
        importance   = psd_ch_h.mean(axis=0)               # [H]
        total_heads  = importance.shape[0]
        n_show       = min(max_heads, total_heads)
        sorted_order = np.argsort(importance)[::-1]
        display_order = sorted_order[:n_show]

        # Unweighted topomap
        topo_path = os.path.join(viz_dir,
                                 f"sub{subject_id}_trial{trial_idx}_stage{s_idx}_topomap.png")
        save_topomap_grid(topo_path, pos2d, psd_ch_h, display_order, importance,
                          subject_id, trial_idx, s_idx, f_min, f_max, cmap, 'unweighted')
        print(f"  [topomap] Stage {s_idx} unweighted → {topo_path}")

        # G-weighted topomap
        G_np          = all_G[s_idx][0].cpu().numpy()
        psd_weighted  = psd_ch_h @ G_np.T
        importance_w  = psd_weighted.mean(axis=0)
        display_order_w = np.argsort(importance_w)[::-1][:n_show]
        topo_w_path   = os.path.join(viz_dir,
                                     f"sub{subject_id}_trial{trial_idx}_stage{s_idx}_topomap_weighted.png")
        save_topomap_grid(topo_w_path, pos2d, psd_weighted, display_order_w, importance_w,
                          subject_id, trial_idx, s_idx, f_min, f_max, cmap, 'G-weighted')
        print(f"  [topomap] Stage {s_idx} G-weighted  → {topo_w_path}")

        # Head interaction figure: importance decay + sim matrix + G matrix
        imp_ranked = importance[sorted_order]
        sim_np     = all_sim[s_idx][0].cpu().numpy()
        reorder    = np.ix_(sorted_order, sorted_order)
        sim_ranked = sim_np[reorder]
        G_ranked   = G_np[reorder]

        mean_imp    = imp_ranked.mean()
        std_imp     = imp_ranked.std()
        high_thresh = mean_imp + n_std * std_imp
        low_thresh  = mean_imp - n_std * std_imp
        n_high      = int(np.sum(imp_ranked > high_thresh))
        n_low       = int(np.sum(imp_ranked < low_thresh))

        fig2 = plt.figure(figsize=(max(18, total_heads * 0.18 + 4), 11))
        gs   = fig2.add_gridspec(2, 2, height_ratios=[0.45, 1],
                                 width_ratios=[1, 1], hspace=0.08, wspace=0.30)
        ax_d  = fig2.add_subplot(gs[0, :])
        ax_s  = fig2.add_subplot(gs[1, 0])
        ax_g  = fig2.add_subplot(gs[1, 1])

        ax_d.plot(range(total_heads), imp_ranked, color='steelblue', linewidth=1.5, zorder=3)
        ax_d.axhline(high_thresh, color='green', linewidth=0.9, linestyle='--', alpha=0.7)
        ax_d.axhline(mean_imp,    color='gray',  linewidth=0.8, linestyle=':',  alpha=0.6)
        if low_thresh > 0:
            ax_d.axhline(low_thresh, color='red', linewidth=0.9, linestyle='--', alpha=0.7)
        if n_high > 0:
            ax_d.axvspan(-0.5, n_high - 0.5, alpha=0.10, color='green',
                         label=f'+{n_std:.1f}σ ({n_high} heads)')
            ax_d.scatter(range(n_high), imp_ranked[:n_high], color='green', s=18, zorder=4)
        if n_low > 0:
            ax_d.axvspan(total_heads - n_low - 0.5, total_heads - 0.5,
                         alpha=0.10, color='red',
                         label=f'−{n_std:.1f}σ ({n_low} heads)')
            ax_d.scatter(range(total_heads - n_low, total_heads),
                         imp_ranked[total_heads - n_low:], color='red', s=18, zorder=4)
        ax_d.set_xlim(-0.5, total_heads - 0.5)
        ax_d.set_ylim(0, imp_ranked.max() * 1.12)
        ax_d.set_ylabel('Importance', fontsize=8); ax_d.set_xticks([])
        ax_d.legend(loc='upper right', fontsize=7); ax_d.grid(alpha=0.3)
        ax_d.set_title(
            f"Stage {s_idx} ({f_min}–{f_max} Hz) — Head Interaction  "
            f"Sub {subject_id}, Trial {trial_idx}  "
            f"({total_heads} heads, μ={mean_imp:.3f} σ={std_imp:.3f})",
            fontsize=9, fontweight='bold')

        tick_step = max(1, total_heads // 16)
        ticks = list(range(0, total_heads, tick_step))

        def _add_boundaries(ax):
            if n_high > 0:
                ax.axhline(n_high - 0.5, color='green', linewidth=0.9, linestyle='--', alpha=0.6)
                ax.axvline(n_high - 0.5, color='green', linewidth=0.9, linestyle='--', alpha=0.6)
            if n_low > 0:
                ax.axhline(total_heads - n_low - 0.5, color='red', linewidth=0.9, linestyle='--', alpha=0.6)
                ax.axvline(total_heads - n_low - 0.5, color='red', linewidth=0.9, linestyle='--', alpha=0.6)
            ax.set_xticks(ticks); ax.set_xticklabels(ticks, fontsize=7)
            ax.set_yticks(ticks); ax.set_yticklabels(ticks, fontsize=7)
            ax.set_xlabel('Head rank  (0 = most important)', fontsize=8)
            ax.set_ylabel('Head rank  (0 = most important)', fontsize=8)

        abs_max = max(np.abs(sim_ranked).max(), 1e-6)
        im_s = ax_s.imshow(sim_ranked, aspect='auto', cmap='RdBu_r',
                           vmin=-abs_max, vmax=abs_max, interpolation='nearest')
        _add_boundaries(ax_s)
        ax_s.set_title('sim  (raw bilinear)\nred=cooperate · blue=compete', fontsize=8, fontweight='bold')
        fig2.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04).set_label('similarity score', fontsize=7)

        im_g = ax_g.imshow(G_ranked, aspect='auto', cmap='YlOrRd',
                           vmin=G_ranked.min(), vmax=G_ranked.max(), interpolation='nearest')
        _add_boundaries(ax_g)
        ax_g.set_title('G  (diffusion filter)\nhigh = heads blend during pooling', fontsize=8, fontweight='bold')
        fig2.colorbar(im_g, ax=ax_g, fraction=0.046, pad=0.04).set_label('blend weight', fontsize=7)

        interaction_path = os.path.join(
            viz_dir, f"sub{subject_id}_trial{trial_idx}_stage{s_idx}_interaction.png")
        fig2.savefig(interaction_path, dpi=130, bbox_inches='tight')
        plt.close(fig2)
        print(f"  [topomap] Stage {s_idx} interaction → {interaction_path}")


if __name__ == '__main__':
    import argparse
    import torch
    from IO.dataset import build_dataset_from_config

    parser = argparse.ArgumentParser(description='EEG Trial Topomap')
    parser.add_argument('--config',     default='config/analysis.json')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--subject',    type=int, default=None)
    parser.add_argument('--trial',      type=int, default=None)
    add_args(parser)
    args = parser.parse_args()

    cfg        = load_config(args.config)
    checkpoint = args.checkpoint or cfg.get('checkpoint', '')
    ds_name, subject = select_subject_dataset(cfg, args.subject)
    _ds_early  = next(
        (ds for ds, dc in cfg['dataset_params'].items() if 'trial_to_use' in dc),
        next(iter(cfg['dataset_params'])))
    trial_cfg  = args.trial if args.trial is not None else cfg['dataset_params'][_ds_early].get('trial_to_use')

    filtered = filter_config_to_subject(cfg, ds_name, subject)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl      = load_model(filtered, checkpoint, device)
    ds       = build_dataset_from_config(filtered, mode='tokenizer')
    t_idx, subject_id = pick_trial(ds, subject, trial_cfg)
    out      = resolve_output_dir(filtered, 'check')
    run(filtered, out, args, model=mdl, dataset=ds, trial_idx=t_idx, subject_id=subject_id)
