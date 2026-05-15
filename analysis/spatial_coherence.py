"""
Spatial coherence analysis: per-head cosine similarity and token agreement
across channels for a single trial.
"""

import os
import csv

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


# ── Pure helpers ─────────────────────────────────────────────────────────────

def sort_channels_spatially(coords: np.ndarray) -> np.ndarray:
    """Sort channel indices by Y descending (front→back) then X ascending (left→right)."""
    return np.lexsort((coords[:, 0], -coords[:, 1]))


@torch.no_grad()
def get_spatial_latents(model, x: torch.Tensor, coords: torch.Tensor,
                        time_idx: torch.Tensor):
    """
    Extract quantized vectors and token indices for every stage.
    Returns (v_qs, indices) — each is a list of [C, N, H, r] tensors.
    """
    B, C, N, L = x.shape
    z = model.embed(x, coords=coords, time_idx=time_idx)
    needed = {h_cfg["stage_idx"] for h_cfg in model.decoder_heads_config}
    stage_latents = {}
    for i, block in enumerate(model.encoder.blocks):
        z = block(z)
        if i in needed:
            stage_latents[i] = z

    all_v_qs, all_indices = [], []
    for i, h_cfg in enumerate(model.decoder_heads_config):
        z_stage = stage_latents[h_cfg["stage_idx"]]
        B_s, C_s, N_s, _ = z_stage.shape
        z_flat  = z_stage.reshape(B_s * C_s, N_s, -1)
        vq_head = model.vq_heads[i]
        H_head  = vq_head.num_heads
        v_q, _, _, idx = vq_head(z_flat)
        v_q = v_q.reshape(B_s, C_s, N_s, H_head, -1)
        idx = idx.reshape(B_s, C_s, N_s, H_head, -1)
        all_v_qs.append(v_q[0])    # [C, N, H, r]
        all_indices.append(idx[0]) # [C, N, H, r]

    return all_v_qs, all_indices


def compute_all_similarity_matrices_batched(v_q: torch.Tensor) -> np.ndarray:
    """Batch cosine similarity. v_q: [C, N, H, r] → [H, N, C, C] numpy array."""
    C, N, H, r = v_q.shape
    v = v_q.permute(2, 1, 0, 3).reshape(H * N, C, r)
    v_norm = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.bmm(v_norm, v_norm.transpose(1, 2)).reshape(H, N, C, C).cpu().numpy()


def compute_all_agreement_matrices_batched(idx: torch.Tensor, num_discrete: int = 5) -> np.ndarray:
    """Batch token agreement. idx: [C, N, H, r] → [H, N, C, C] numpy array."""
    C, N, H, r = idx.shape
    powers = torch.tensor([num_discrete ** i for i in range(r)],
                          device=idx.device, dtype=idx.dtype)
    hashed = (idx * powers).sum(dim=-1).permute(2, 1, 0)  # [H, N, C]
    agr = (hashed.unsqueeze(-1) == hashed.unsqueeze(-2)).float()
    return agr.cpu().numpy()


# ── Module interface ──────────────────────────────────────────────────────────

def add_args(parser):
    pass  # no extra args


def run(config, output_dir, args, model=None, dataset=None, trial_idx=None, subject_id=None):
    viz_dir = os.path.join(output_dir, 'spatial_coherence')
    os.makedirs(viz_dir, exist_ok=True)

    device = next(model.parameters()).device

    x_patches, coords, time_indices, _, _ = dataset[trial_idx]
    x_in        = x_patches.unsqueeze(0).to(device)
    coords_in   = coords.unsqueeze(0).to(device)
    time_idx_in = time_indices.unsqueeze(0).to(device)

    spatial_idx     = sort_channels_spatially(coords.numpy())
    sorted_ch_names = [dataset.base_dataset.channel_names[i] for i in spatial_idx]

    v_qs, indices = get_spatial_latents(model, x_in, coords_in, time_idx_in)
    num_stages  = len(v_qs)
    num_patches = x_patches.shape[1]

    patch_rows, matrix_rows = [], []

    for s_idx in range(num_stages):
        v_q_s     = v_qs[s_idx]      # [C, N, H, r]
        idx_s     = indices[s_idx]    # [C, N, H, r]
        num_heads = v_q_s.shape[2]
        f_min, f_max = model.decoder_heads_config[s_idx]["freq_range"]

        # Per-head per-patch diversity → decide which patches are "active"
        patch_div  = v_q_s.std(dim=0).mean(dim=-1).T.cpu().numpy()      # [H, N]
        thresh     = np.percentile(patch_div, 30, axis=1, keepdims=True) # [H, 1]
        active     = patch_div >= thresh                                  # [H, N]
        active_idx = [np.where(active[h])[0] for h in range(num_heads)]

        v_q_sorted   = v_q_s[spatial_idx]
        idx_sorted   = idx_s[spatial_idx]
        num_discrete = model.vq_heads[s_idx].num_discrete

        sim_all = compute_all_similarity_matrices_batched(v_q_sorted)              # [H, N, C, C]
        agr_all = compute_all_agreement_matrices_batched(idx_sorted, num_discrete) # [H, N, C, C]

        num_cols = 3 + num_patches
        fig = plt.figure(figsize=(4 * num_cols, 4 * num_heads), constrained_layout=True)
        gs  = fig.add_gridspec(num_heads, num_cols,
                               width_ratios=[1.5, 1.5, 1.5] + [1.0] * num_patches)

        for h in range(num_heads):
            sim_h     = sim_all[h]                                    # [N, C, C]
            agr_h     = agr_all[h]
            act_h     = active_idx[h]
            sim_avg   = sim_h.mean(axis=0)
            sim_filt  = sim_h[act_h].mean(axis=0)
            agr_avg   = agr_h.mean(axis=0)

            mask_off  = ~np.eye(sim_avg.shape[0], dtype=bool)
            for p in range(num_patches):
                patch_rows.append({
                    'subject': subject_id, 'trial': trial_idx,
                    'stage': s_idx, 'freq_min': f_min, 'freq_max': f_max,
                    'head': h, 'patch': p,
                    'is_active':       bool(active[h, p]),
                    'patch_diversity': float(patch_div[h, p]),
                    'mean_sim':        float(sim_h[p][mask_off].mean()),
                    'std_sim':         float(sim_h[p][mask_off].std()),
                    'mean_agr':        float(agr_h[p][mask_off].mean()),
                })

            for ci, ch_i in enumerate(sorted_ch_names):
                for cj, ch_j in enumerate(sorted_ch_names):
                    matrix_rows.append({
                        'subject': subject_id, 'trial': trial_idx,
                        'stage': s_idx, 'freq_min': f_min, 'freq_max': f_max,
                        'head': h, 'ch_i': ch_i, 'ch_j': ch_j,
                        'sim_avg':          float(sim_avg[ci, cj]),
                        'sim_avg_filtered': float(sim_filt[ci, cj]),
                        'agr_avg':          float(agr_avg[ci, cj]),
                    })

            ax0 = fig.add_subplot(gs[h, 0])
            im  = ax0.imshow(sim_avg, cmap='viridis', vmin=0, vmax=1)
            ax0.set_title(f"H{h} Avg Sim (all)", fontsize=10, fontweight='bold')
            fig.colorbar(im, ax=ax0)
            if h == 0: ax0.set_ylabel("Sorted Channels")

            ax1  = fig.add_subplot(gs[h, 1])
            im_f = ax1.imshow(sim_filt, cmap='viridis', vmin=0, vmax=1)
            ax1.set_title(f"H{h} Avg Sim (active {len(act_h)}/{num_patches})",
                          fontsize=10, fontweight='bold')
            fig.colorbar(im_f, ax=ax1)

            ax2  = fig.add_subplot(gs[h, 2])
            im2  = ax2.imshow(agr_avg, cmap='magma', vmin=0, vmax=1)
            ax2.set_title(f"H{h} Token Agreement %", fontsize=10)
            fig.colorbar(im2, ax=ax2)

            for p in range(num_patches):
                ax_p = fig.add_subplot(gs[h, 3 + p])
                ax_p.imshow(sim_h[p], cmap='viridis', vmin=0, vmax=1)
                ax_p.set_title(f"P{p}", fontsize=7)
                ax_p.set_xticks([]); ax_p.set_yticks([])
                if not active[h, p]:
                    for spine in ax_p.spines.values():
                        spine.set_edgecolor('red'); spine.set_linewidth(2)

        fig.suptitle(
            f"Spatial Vector Similarity | Stage {s_idx} ({f_min}-{f_max}Hz) | "
            f"Sub {subject_id} Trial {trial_idx}", fontsize=16)
        fig.savefig(os.path.join(
            viz_dir, f"sub{subject_id}_trial{trial_idx}_stage{s_idx}_vector_sim.png"), dpi=120)
        plt.close(fig)
        print(f"  [spatial] Stage {s_idx} saved")

    prefix = f"sub{subject_id}_trial{trial_idx}"
    if patch_rows:
        with open(os.path.join(viz_dir, f"{prefix}_patch_stats.csv"), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=patch_rows[0].keys())
            w.writeheader(); w.writerows(patch_rows)
    if matrix_rows:
        with open(os.path.join(viz_dir, f"{prefix}_matrix_stats.csv"), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=matrix_rows[0].keys())
            w.writeheader(); w.writerows(matrix_rows)

    print(f"  [spatial] → {viz_dir}")


if __name__ == '__main__':
    import argparse
    import torch
    from IO.dataset import build_dataset_from_config

    parser = argparse.ArgumentParser(description='EEG Spatial Coherence Analysis')
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
