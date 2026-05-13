import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle
import random

from IO.dataset import build_dataset_from_config
from model.factory import build_model_from_config


def project_coords_2d(coords):
    """Azimuthal equidistant projection of 3D electrode coords to 2D."""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-8
    xn, yn, zn = x / r, y / r, z / r
    theta = np.arccos(np.clip(zn, -1, 1))
    phi = np.arctan2(xn, yn)
    rho = theta / (np.pi / 2)
    return np.stack([rho * np.sin(phi), rho * np.cos(phi)], axis=1)


def draw_topomap(ax, pos2d, values, n_grid=100, cmap='YlOrRd', vmin=None, vmax=None):
    """Interpolated topomap on a circular head outline. pos2d: [C,2], values: [C]."""
    px, py = pos2d[:, 0], pos2d[:, 1]
    vmin = vmin if vmin is not None else values.min()
    vmax = vmax if vmax is not None else values.max()

    xi = np.linspace(-1.1, 1.1, n_grid)
    yi = np.linspace(-1.1, 1.1, n_grid)
    Xi, Yi = np.meshgrid(xi, yi)

    try:
        triang = mtri.Triangulation(px, py)
        Zi = mtri.LinearTriInterpolator(triang, values)(Xi, Yi)
    except Exception:
        from scipy.interpolate import griddata
        Zi = griddata((px, py), values, (Xi, Yi), method='linear')

    Zi = np.array(Zi.data if hasattr(Zi, 'data') else Zi, dtype=float)
    Zi[np.sqrt(Xi**2 + Yi**2) > 1.0] = np.nan

    im = ax.pcolormesh(Xi, Yi, Zi, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='auto', rasterized=True)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color='k', linewidth=1.2))
    ax.plot([0, 0], [1.0, 1.15], 'k-', linewidth=1.2)
    ax.plot([-1.0, -1.1], [0.1, 0.0], 'k-', linewidth=1.0)
    ax.plot([1.0, 1.1], [0.1, 0.0], 'k-', linewidth=1.0)
    ax.scatter(px, py, s=6, c='k', zorder=5, alpha=0.5)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal'); ax.axis('off')
    return im


@torch.no_grad()
def extract_head_psd(model, x, coords, time_idx):
    """
    Returns per-stage per-head per-channel raw reconstructed PSD power.

    For each head h and channel c:
        PSD_h(c) = Σ_f  real_h(c,f)² + imag_h(c,f)²

    where real_h / imag_h is head h's unweighted additive contribution to the
    FFT output (v_q[h] decoded through M[h], no G blending applied).
    G is applied during training via column-sum scaling but is excluded here
    so topomaps show each head's intrinsic spatial pattern.

    Returns: list of [C, H] numpy arrays, one per decoder stage.
    """
    import math
    B, C, N, L = x.shape

    z = model.embed(x, coords=coords if model.use_spatial_embedding else None, time_idx=time_idx)
    needed = {h_cfg["stage_idx"] for h_cfg in model.decoder_heads_config}
    stage_latents = {}
    for i, block in enumerate(model.encoder.blocks):
        z = block(z)
        if i in needed:
            stage_latents[i] = z

    results = []
    for i, h_cfg in enumerate(model.decoder_heads_config):
        z_stage = stage_latents[h_cfg["stage_idx"]]
        B_s, C_s, N_s, D_s = z_stage.shape
        z_flat = z_stage.reshape(B_s * C_s, N_s, D_s)

        vq_head = model.vq_heads[i]
        decoder = model.decoders[i]

        # Train mode prevents EMA updates running a second time on the same data.
        was_vq = vq_head.training
        vq_head.train()
        v_q, _, _, _ = vq_head(z_flat)            # [B*C, N, H, r]
        vq_head.train(was_vq)

        H, r  = vq_head.num_heads, vq_head.r
        D_dim = vq_head.A.shape[0]

        M = torch.einsum('dhr,hdf->hrf',
                         vq_head.A.view(D_dim, H, r),
                         decoder.W)                             # [H, r, 2*F]

        # Raw per-head decoded spectrum, summed over patches: [B*C, H, 2*F]
        h_out = torch.einsum('bnhr,hrf->bhf', v_q, M) / math.sqrt(H)

        F_dim  = h_out.shape[-1] // 2
        h_real = h_out[..., :F_dim]
        h_imag = h_out[..., F_dim:]
        psd    = (h_real.pow(2) + h_imag.pow(2)).sum(dim=-1)   # [B*C, H]

        psd_ch_h = psd.reshape(B_s, C_s, H)[0].cpu().numpy()  # [C, H]
        results.append(psd_ch_h)

    return results


def save_topomap_grid(output_path, pos2d, psd_ch_h, display_order, importance,
                      subject_id, trial_idx, s_idx, f_min, f_max, cmap, title_tag):
    """Save a topomap grid figure for the given psd_ch_h [C, H] and display order."""
    n_show = len(display_order)
    n_cols = min(16, n_show)
    n_rows = int(np.ceil(n_show / n_cols))
    rank_norm = np.arange(n_show) / max(n_show - 1, 1)

    vmin = psd_ch_h[:, display_order].min()
    vmax = psd_ch_h[:, display_order].max()

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.8 * n_cols, 3.2 * n_rows),
                             squeeze=False)
    for pos, h_orig in enumerate(display_order):
        row, col = divmod(pos, n_cols)
        ax = axes[row][col]
        im = draw_topomap(ax, pos2d, psd_ch_h[:, h_orig],
                          cmap=cmap, vmin=vmin, vmax=vmax)
        title_color = plt.cm.RdYlGn(1.0 - rank_norm[pos])[:3]
        ax.set_title(f"#{pos + 1}  H{h_orig}\n{importance[h_orig]:.3f}",
                     fontsize=6, color=title_color, fontweight='bold')

    for pos in range(n_show, n_rows * n_cols):
        row, col = divmod(pos, n_cols)
        axes[row][col].axis('off')

    fig.subplots_adjust(right=0.88, hspace=0.4, wspace=0.05)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.012, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('PSD power', fontsize=9)
    fig.suptitle(
        f"Stage {s_idx}  ({f_min}–{f_max} Hz) — {n_show} heads sorted by reconstruction power  [{title_tag}]\n"
        f"Sub {subject_id}, Trial {trial_idx}  (left→right, top→bottom: most→least important)",
        fontsize=11, fontweight='bold'
    )
    fig.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='EEG Trial Topomap — sorted by per-head reconstruction power')
    parser.add_argument('--config',     type=str, default='config/config.json')
    parser.add_argument('--checkpoint', type=str, default='output/attnvq_v06/tokenizer/best_tokenizer.pth')
    parser.add_argument('--subject',    type=int, default=None)
    parser.add_argument('--trial',      type=int, default=None)
    parser.add_argument('--max_heads',  type=int, default=128,
                        help='Max heads to display in topomap grid (top-N by importance)')
    parser.add_argument('--n_std',      type=float, default=1.0,
                        help='Highlight heads beyond ±N std from mean importance')
    parser.add_argument('--cmap',       type=str, default='YlOrRd')
    args = parser.parse_args()

    # --- Output dir ---
    ckpt_parts = os.path.normpath(args.checkpoint).split(os.sep)
    if 'output' in ckpt_parts:
        model_name = ckpt_parts[ckpt_parts.index('output') + 1]
        output_dir = os.path.join('output', model_name, 'visualization', 'trial_topomap')
    else:
        output_dir = 'output/visualization/trial_topomap'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.config, 'r') as f:
        config = json.load(f)

    # --- Pick subject / dataset ---
    ds_names = list(config['dataset_params'].keys())
    chosen_ds = random.choice(ds_names) if args.subject is None else None

    for ds_name, ds_args in config['dataset_params'].items():
        with open(os.path.join(ds_args['dataset_path'], 'metadata.json'), 'r') as f:
            all_subs = list(json.load(f).get('data_structure', {}).keys())

        if args.subject is not None:
            use = [args.subject] if (str(args.subject) in all_subs or args.subject in all_subs) else []
        elif ds_name == chosen_ds:
            random_sub = random.choice(all_subs)
            use = [random_sub]
            args.subject = int(random_sub) if str(random_sub).isdigit() else random_sub
        else:
            use = []
        config['dataset_params'][ds_name]['subject_to_use'] = use

    dataset = build_dataset_from_config(config, mode='tokenizer')
    if len(dataset) == 0:
        print("Empty dataset."); return

    model = build_model_from_config(config, n_fft_trial=dataset.base_dataset.fft_params['n_fft']).to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if missing:
            print(f"  [checkpoint] {len(missing)} new keys not in checkpoint (initialised fresh): e.g. {missing[0]}")
        if unexpected:
            print(f"  [checkpoint] {len(unexpected)} keys in checkpoint not in model: {unexpected}")
    model.eval()

    # --- Pick trial ---
    sub_indices = (dataset.base_dataset.subject_data == args.subject).nonzero(as_tuple=True)[0] \
        if args.subject is not None else torch.arange(len(dataset))
    trial_idx = sub_indices[args.trial % len(sub_indices)].item() \
        if args.trial is not None else random.choice(sub_indices).item()
    subject_id = dataset.base_dataset.subject_data[trial_idx].item()

    x_patches, coords, time_indices, label, _ = dataset[trial_idx]
    x_in       = x_patches.unsqueeze(0).to(device)
    coords_in  = coords.unsqueeze(0).to(device)
    time_idx_in = time_indices.unsqueeze(0).to(device)

    ch_names = dataset.base_dataset.channel_names
    pos2d    = project_coords_2d(coords.numpy())
    print(f"Subject {subject_id}, Trial {trial_idx} | {len(ch_names)} channels")

    # --- Forward pass ---
    # all_head_importances: list of [B, H] — G diagonal / Laplacian survival score (for ordering)
    # stage_psd:            list of [C, H] — G-blended per-head PSD (used for both ordering and topomap values)
    with torch.no_grad():
        _, _, _, _, _, all_head_importances, all_sim, all_G = model(x_in, coords_in, time_idx_in)

    stage_psd = extract_head_psd(model, x_in, coords_in, time_idx_in)

    # --- One figure per decoder stage ---
    for s_idx, (psd_ch_h, head_imp, h_cfg) in enumerate(
            zip(stage_psd, all_head_importances, model.decoder_heads_config)):

        # psd_ch_h: [C, H] — G-weighted reconstructed PSD power per channel per head
        # importance: [H]  — mean G-weighted PSD power across channels (head ranking)
        # Note: Laplacian Survival Score from head_imp collapses to 1/H for large H
        # because softmax-based W is near-uniform, so we rank by reconstruction power instead.
        importance  = psd_ch_h.mean(axis=0)       # [H]
        f_min, f_max = h_cfg["freq_range"]
        total_heads  = importance.shape[0]
        n_show       = min(args.max_heads, total_heads)

        # Full ranking: index 0 = most important, index -1 = least important
        sorted_order = np.argsort(importance)[::-1]   # [H], descending
        display_order = sorted_order[:n_show]          # top-N by importance

        # --- Unweighted topomap grid ---
        topo_path = os.path.join(output_dir,
                                 f"sub{subject_id}_trial{trial_idx}_stage{s_idx}_topomap.png")
        save_topomap_grid(topo_path, pos2d, psd_ch_h, display_order, importance,
                          subject_id, trial_idx, s_idx, f_min, f_max, args.cmap,
                          title_tag='unweighted')
        print(f"  Stage {s_idx} topomap (unweighted) → {topo_path}")

        # --- Weighted topomap grid (G-blended: psd_weighted[c,h] = Σ_k G[h,k] * psd[c,k]) ---
        G_np_now = all_G[s_idx][0].cpu().numpy()          # [H, H]
        psd_weighted = psd_ch_h @ G_np_now.T              # [C, H]
        importance_w = psd_weighted.mean(axis=0)          # [H]
        sorted_order_w = np.argsort(importance_w)[::-1]
        display_order_w = sorted_order_w[:n_show]
        topo_w_path = os.path.join(output_dir,
                                   f"sub{subject_id}_trial{trial_idx}_stage{s_idx}_topomap_weighted.png")
        save_topomap_grid(topo_w_path, pos2d, psd_weighted, display_order_w, importance_w,
                          subject_id, trial_idx, s_idx, f_min, f_max, args.cmap,
                          title_tag='G-weighted')
        print(f"  Stage {s_idx} topomap (G-weighted)  → {topo_w_path}")

        # --- Combined: sim + G matrices + importance decay ---
        imp_ranked = importance[sorted_order]                         # [H] best→worst

        # Reorder both matrices by importance rank
        sim_np = all_sim[s_idx][0].cpu().numpy()                      # [H, H]
        idx    = np.ix_(sorted_order, sorted_order)
        sim_ranked = sim_np[idx]
        G_ranked   = G_np_now[idx]

        # Std-based region boundaries
        mean_imp   = imp_ranked.mean()
        std_imp    = imp_ranked.std()
        high_thresh = mean_imp + args.n_std * std_imp
        low_thresh  = mean_imp - args.n_std * std_imp
        n_high = int(np.sum(imp_ranked > high_thresh))   # prefix of sorted array
        n_low  = int(np.sum(imp_ranked < low_thresh))    # suffix of sorted array

        fig_w = max(18, total_heads * 0.18 + 4)
        fig2  = plt.figure(figsize=(fig_w, 11))
        gs    = fig2.add_gridspec(2, 2,
                                  height_ratios=[0.45, 1],
                                  width_ratios=[1, 1],
                                  hspace=0.08, wspace=0.30)
        ax_decay = fig2.add_subplot(gs[0, :])
        ax_sim   = fig2.add_subplot(gs[1, 0])
        ax_G     = fig2.add_subplot(gs[1, 1])

        # Importance decay line with std-based highlights
        ax_decay.plot(range(total_heads), imp_ranked, color='steelblue', linewidth=1.5, zorder=3)
        ax_decay.axhline(high_thresh, color='green', linewidth=0.9, linestyle='--', alpha=0.7)
        ax_decay.axhline(mean_imp,    color='gray',  linewidth=0.8, linestyle=':',  alpha=0.6)
        if low_thresh > 0:
            ax_decay.axhline(low_thresh, color='red', linewidth=0.9, linestyle='--', alpha=0.7)

        if n_high > 0:
            ax_decay.axvspan(-0.5, n_high - 0.5, alpha=0.10, color='green',
                             label=f'+{args.n_std:.1f}σ  ({n_high} heads)')
            ax_decay.scatter(range(n_high), imp_ranked[:n_high],
                             color='green', s=18, zorder=4)
        if n_low > 0:
            ax_decay.axvspan(total_heads - n_low - 0.5, total_heads - 0.5,
                             alpha=0.10, color='red',
                             label=f'−{args.n_std:.1f}σ  ({n_low} heads)')
            ax_decay.scatter(range(total_heads - n_low, total_heads),
                             imp_ranked[total_heads - n_low:], color='red', s=18, zorder=4)

        ax_decay.set_xlim(-0.5, total_heads - 0.5)
        ax_decay.set_ylim(0, imp_ranked.max() * 1.12)
        ax_decay.set_ylabel('Importance', fontsize=8)
        ax_decay.set_xticks([])
        ax_decay.legend(loc='upper right', fontsize=7)
        ax_decay.grid(alpha=0.3)
        ax_decay.set_title(
            f"Stage {s_idx}  ({f_min}–{f_max} Hz) — Head Interaction   "
            f"Sub {subject_id}, Trial {trial_idx}  "
            f"({total_heads} heads, left→right: most→least important, "
            f"μ={mean_imp:.3f}  σ={std_imp:.3f})",
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

        # Left: sim (raw bilinear, can be negative → fight/cooperate)
        abs_max = max(np.abs(sim_ranked).max(), 1e-6)
        im_sim = ax_sim.imshow(sim_ranked, aspect='auto', cmap='RdBu_r',
                               vmin=-abs_max, vmax=abs_max, interpolation='nearest')
        _add_boundaries(ax_sim)
        ax_sim.set_title('sim  (raw bilinear)\nred=cooperate · blue=compete', fontsize=8, fontweight='bold')
        cb_sim = fig2.colorbar(im_sim, ax=ax_sim, fraction=0.046, pad=0.04)
        cb_sim.set_label('similarity score', fontsize=7)

        # Right: G (diffusion filter, non-negative → actual pooling blend)
        im_G = ax_G.imshow(G_ranked, aspect='auto', cmap='YlOrRd',
                           vmin=G_ranked.min(), vmax=G_ranked.max(), interpolation='nearest')
        _add_boundaries(ax_G)
        ax_G.set_title('G  (diffusion filter)\nhigh = heads blend during pooling', fontsize=8, fontweight='bold')
        cb_G = fig2.colorbar(im_G, ax=ax_G, fraction=0.046, pad=0.04)
        cb_G.set_label('blend weight', fontsize=7)

        decay_path = os.path.join(output_dir,
                                  f"sub{subject_id}_trial{trial_idx}_stage{s_idx}_importance.png")
        fig2.savefig(decay_path, dpi=130, bbox_inches='tight')
        plt.close(fig2)
        print(f"  Stage {s_idx} interaction map → {decay_path}")

    print("Done.")


if __name__ == "__main__":
    main()
