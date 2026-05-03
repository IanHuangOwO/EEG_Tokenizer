import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm

from IO.dataset import build_dataset_from_config
from model.factory import build_model_from_config

def sort_channels_spatially(coords):
    """
    Sorts channel indices based on 3D coordinates (Y - Front-to-Back, then X - Left-to-Right).
    coords: [C, 3]
    """
    # Y is typically front-back, X is left-right
    y = coords[:, 1]
    x = coords[:, 0]
    # Sort primarily by Y (descending - front to back), then X (ascending - left to right)
    indices = np.lexsort((x, -y))
    return indices

@torch.no_grad()
def get_spatial_latents(model, x, coords, time_idx):
    """
    Extracts quantized vectors (v_q) and token indices for all heads and stages.
    Returns: 
      v_qs: List of [C, N, H, r] tensors (one per stage)
      indices: List of [C, N, H] tensors (one per stage)
    """
    B, C, N, L = x.shape
    # 1. Encoder forward
    z = model.embed(x, coords=coords, time_idx=time_idx)
    
    needed_stages = {h_cfg["stage_idx"] for h_cfg in model.decoder_heads_config}
    stage_latents = {}
    for i, block in enumerate(model.encoder.blocks):
        z = block(z)
        if i in needed_stages:
            stage_latents[i] = z

    all_v_qs = []
    all_indices = []

    for i, h_cfg in enumerate(model.decoder_heads_config):
        z_stage = stage_latents[h_cfg["stage_idx"]] # [B, C, N, D]
        B_s, C_s, N_s, D_s = z_stage.shape
        z_flat = z_stage.reshape(B_s * C_s, N_s, -1)
        
        vq_head = model.vq_heads[i]
        H_head = vq_head.num_heads
        # v_q: [B*C, N, H, r], indices: [B*C, N, H]
        v_q, _, _, idx = vq_head(z_flat)
        
        # Reshape using known H_head
        v_q = v_q.reshape(B_s, C_s, N_s, H_head, -1)
        idx = idx.reshape(B_s, C_s, N_s, H_head, -1)

        all_v_qs.append(v_q[0]) # [C, N, H, r]
        all_indices.append(idx[0]) # [C, N, H, r]

    return all_v_qs, all_indices

def compute_all_similarity_matrices_batched(v_q):
    """
    Batch-computes cosine similarity across all heads and patches in one shot.
    v_q: [C, N, H, r] — returns [H, N, C, C] numpy array.
    """
    C, N, H, r = v_q.shape
    # [H, N, C, r]
    v = v_q.permute(2, 1, 0, 3).reshape(H * N, C, r)
    v_norm = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    sim = torch.bmm(v_norm, v_norm.transpose(1, 2))  # [H*N, C, C]
    return sim.reshape(H, N, C, C).cpu().numpy()

def compute_all_agreement_matrices_batched(idx, num_discrete=5):
    """
    Batch-computes token agreement across all heads and patches in one shot.
    idx: [C, N, H, r] — returns [H, N, C, C] numpy array.
    Hashes r sub-indices into a single scalar to avoid [N,C,C,r] intermediate.
    """
    C, N, H, r = idx.shape
    # Hash [C, N, H, r] -> [C, N, H] scalar via positional encoding
    powers = torch.tensor([num_discrete ** i for i in range(r)], device=idx.device, dtype=idx.dtype)
    hashed = (idx * powers).sum(dim=-1)  # [C, N, H]
    # [H, N, C] for broadcasting
    h = hashed.permute(2, 1, 0)  # [H, N, C]
    agr = (h.unsqueeze(-1) == h.unsqueeze(-2)).float()  # [H, N, C, C]
    return agr.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='EEG Spatial Coherence Analysis')
    parser.add_argument('--config', type=str, default='config/config.json')
    parser.add_argument('--checkpoint', type=str, default='output/attnvq_v02/tokenizer/best_tokenizer.pth')
    parser.add_argument('--subject', type=int, default=None)
    parser.add_argument('--trial', type=int, default=None)
    args = parser.parse_args()

    # --- Setup Output Dir ---
    ckpt_parts = os.path.normpath(args.checkpoint).split(os.sep)
    if 'output' in ckpt_parts:
        out_idx = ckpt_parts.index('output')
        model_name = ckpt_parts[out_idx + 1]
        output_dir = os.path.join('output', model_name, 'visualization', 'spatial_coherence')
    else:
        output_dir = 'output/visualization/spatial_coherence'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Fast loading for single subject
    for ds_name, ds_args in config['dataset_params'].items():
        meta_path = os.path.join(ds_args['dataset_path'], 'metadata.json')
        with open(meta_path, 'r') as f_meta:
            meta = json.load(f_meta)
        all_subs = list(meta.get('data_structure', {}).keys())
        if args.subject is not None:
            config['dataset_params'][ds_name]['subject_to_use'] = [args.subject]
        elif ds_args['subject_to_use'] in [["all"], "all"]:
            random_sub = random.choice(all_subs)
            config['dataset_params'][ds_name]['subject_to_use'] = [random_sub]
            args.subject = int(random_sub) if random_sub.isdigit() else random_sub

    dataset = build_dataset_from_config(config, mode='tokenizer')
    if len(dataset) == 0: return
    n_fft = dataset.base_dataset.fft_params['n_fft']
    model = build_model_from_config(config, n_fft_trial=n_fft).to(device)
    
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    sub_indices = (dataset.base_dataset.subject_data == args.subject).nonzero(as_tuple=True)[0] if args.subject is not None else torch.arange(len(dataset))
    trial_idx = sub_indices[args.trial % len(sub_indices)].item() if args.trial is not None else random.choice(sub_indices).item()
    subject_id = dataset.base_dataset.subject_data[trial_idx].item()

    # Get data
    x_patches, coords, time_indices, _, _ = dataset[trial_idx]
    x_in = x_patches.unsqueeze(0).to(device)
    coords_in = coords.unsqueeze(0).to(device)
    time_idx_in = time_indices.unsqueeze(0).to(device)

    # Sort channels for meaningful heatmaps
    spatial_idx = sort_channels_spatially(coords.numpy())
    sorted_ch_names = [dataset.base_dataset.channel_names[i] for i in spatial_idx]

    # Extract Latents
    v_qs, indices = get_spatial_latents(model, x_in, coords_in, time_idx_in)

    num_stages = len(v_qs)
    num_patches = x_patches.shape[1]

    # CSV accumulators
    patch_rows  = []  # one row per (stage, head, patch)
    matrix_rows = []  # one row per (stage, head, ch_i, ch_j)

    for s_idx in range(num_stages):
        v_q_s = v_qs[s_idx] # [C, N, H, r]
        idx_s = indices[s_idx] # [C, N, H, r]
        num_heads = v_q_s.shape[2]
        f_min, f_max = model.decoder_heads_config[s_idx]["freq_range"]

        # Per-head per-patch diversity: std of v_q across channels — [H, N]
        # Low std = all channels uniform = yellow heatmap → exclude these
        # v_q_s: [C, N, H, r] -> std over C -> [N, H, r] -> mean over r -> [N, H] -> T -> [H, N]
        patch_diversity = v_q_s.std(dim=0).mean(dim=-1).T.cpu().numpy()  # [H, N]
        thresh          = np.percentile(patch_diversity, 30, axis=1, keepdims=True)  # [H, 1]
        active_mask     = patch_diversity >= thresh   # [H, N] bool
        active_indices  = [np.where(active_mask[h])[0] for h in range(num_heads)]

        # Batch-compute all [H, N, C, C] matrices in one shot
        v_q_sorted   = v_q_s[spatial_idx]   # [C, N, H, r]
        idx_sorted   = idx_s[spatial_idx]    # [C, N, H, r]
        num_discrete = model.vq_heads[s_idx].num_discrete

        sim_all_heads = compute_all_similarity_matrices_batched(v_q_sorted)              # [H, N, C, C]
        agr_all_heads = compute_all_agreement_matrices_batched(idx_sorted, num_discrete) # [H, N, C, C]

        # --- Dashboard: Vector Similarity ---
        num_cols = 3 + num_patches  # avg | filtered avg | token agr | patch_0 ... patch_N
        fig_sim = plt.figure(figsize=(4 * num_cols, 4 * num_heads), constrained_layout=True)
        gs_sim = fig_sim.add_gridspec(num_heads, num_cols, width_ratios=[1.5, 1.5, 1.5] + [1.0] * num_patches)

        for h in range(num_heads):
            sim_all   = sim_all_heads[h]   # [N, C, C]
            agr_all   = agr_all_heads[h]   # [N, C, C]
            act_idx_h = active_indices[h]

            sim_avg          = sim_all.mean(axis=0)
            sim_avg_filtered = sim_all[act_idx_h].mean(axis=0)
            agr_avg          = agr_all.mean(axis=0)

            # --- Collect patch-level CSV rows ---
            for p_idx in range(num_patches):
                mat = sim_all[p_idx]
                n_ch = mat.shape[0]
                mask_off = ~np.eye(n_ch, dtype=bool)
                patch_rows.append({
                    'subject': subject_id, 'trial': trial_idx,
                    'stage': s_idx, 'freq_min': f_min, 'freq_max': f_max,
                    'head': h, 'patch': p_idx,
                    'is_active': bool(active_mask[h, p_idx]),
                    'patch_diversity': float(patch_diversity[h, p_idx]),
                    'mean_sim': float(mat[mask_off].mean()),
                    'std_sim':  float(mat[mask_off].std()),
                    'mean_agr': float(agr_all[p_idx][mask_off].mean()),
                })

            # --- Collect averaged-matrix CSV rows ---
            for ci, ch_i in enumerate(sorted_ch_names):
                for cj, ch_j in enumerate(sorted_ch_names):
                    matrix_rows.append({
                        'subject': subject_id, 'trial': trial_idx,
                        'stage': s_idx, 'freq_min': f_min, 'freq_max': f_max,
                        'head': h,
                        'ch_i': ch_i, 'ch_j': ch_j,
                        'sim_avg': float(sim_avg[ci, cj]),
                        'sim_avg_filtered': float(sim_avg_filtered[ci, cj]),
                        'agr_avg': float(agr_avg[ci, cj]),
                    })

            # Col 0: Avg Similarity (all patches)
            ax_avg = fig_sim.add_subplot(gs_sim[h, 0])
            im = ax_avg.imshow(sim_avg, cmap='viridis', vmin=0, vmax=1)
            ax_avg.set_title(f"Head {h} Avg Sim (all)", fontsize=10, fontweight='bold')
            fig_sim.colorbar(im, ax=ax_avg)
            if h == 0:
                ax_avg.set_ylabel("Sorted Channels")

            # Col 1: Filtered Avg Similarity (bottom 20% diversity excluded per head)
            ax_filt = fig_sim.add_subplot(gs_sim[h, 1])
            im_f = ax_filt.imshow(sim_avg_filtered, cmap='viridis', vmin=0, vmax=1)
            ax_filt.set_title(f"Head {h} Avg Sim (active {len(act_idx_h)}/{num_patches})", fontsize=10, fontweight='bold')
            fig_sim.colorbar(im_f, ax=ax_filt)

            # Col 2: Token Agreement % (all patches)
            ax_tag = fig_sim.add_subplot(gs_sim[h, 2])
            im2 = ax_tag.imshow(agr_avg, cmap='magma', vmin=0, vmax=1)
            ax_tag.set_title(f"Head {h} Token Agreement %", fontsize=10)
            fig_sim.colorbar(im2, ax=ax_tag)

            # Col 3+: All patch snapshots, red border = excluded for this head
            for p_idx in range(num_patches):
                ax_snap = fig_sim.add_subplot(gs_sim[h, 3 + p_idx])
                ax_snap.imshow(sim_all[p_idx], cmap='viridis', vmin=0, vmax=1)
                ax_snap.set_title(f"P{p_idx}", fontsize=7)
                ax_snap.set_xticks([]); ax_snap.set_yticks([])
                if not active_mask[h, p_idx]:
                    for spine in ax_snap.spines.values():
                        spine.set_edgecolor('red'); spine.set_linewidth(2)

            # --- Plot Token Agreement Dashboard (Detail) ---
            # Col 0: Avg Agreement
            # ax_aavg = fig_agr.add_subplot(gs_agr[h, 0])
            # im3 = ax_aavg.imshow(agr_avg, cmap='magma', vmin=0, vmax=1)
            # ax_aavg.set_title(f"Head {h} Avg Agreement", fontsize=10, fontweight='bold')
            # fig_agr.colorbar(im3, ax=ax_aavg)

            # Col 1-4: Snapshots of agreement
            # for i, p_idx in enumerate([0, 8, 16, 24]): # Fixed spacing for agreement
            #     if i >= 4: break
            #     ax_asnap = fig_agr.add_subplot(gs_agr[h, 1 + i])
            #     agr_p = compute_agreement_matrix(idx_s[spatial_idx, p_idx, h])
            #     ax_asnap.imshow(agr_p, cmap='magma', vmin=0, vmax=1)
            #     ax_asnap.set_title(f"Patch {p_idx} Match", fontsize=9)

        fig_sim.suptitle(f"Spatial Vector Similarity | Stage {s_idx} ({f_min}-{f_max}Hz) | Sub {subject_id} Trial {trial_idx}", fontsize=16)
        fig_sim.savefig(os.path.join(output_dir, f"sub{subject_id}_trial{trial_idx}_stage{s_idx}_vector_sim.png"), dpi=120)
        plt.close(fig_sim)

        # fig_agr.suptitle(f"Spatial Token Agreement | Stage {s_idx} ({f_min}-{f_max}Hz) | Sub {subject_id} Trial {trial_idx}", fontsize=16)
        # fig_agr.savefig(os.path.join(output_dir, f"sub{subject_id}_trial{trial_idx}_stage{s_idx}_token_agr.png"), dpi=120)
        # plt.close(fig_agr)

        print(f"Saved Spatial dashboards for Stage {s_idx}")

    # --- Save CSVs ---
    patch_csv  = os.path.join(output_dir, f"sub{subject_id}_trial{trial_idx}_patch_stats.csv")
    matrix_csv = os.path.join(output_dir, f"sub{subject_id}_trial{trial_idx}_matrix_stats.csv")

    with open(patch_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=patch_rows[0].keys())
        writer.writeheader(); writer.writerows(patch_rows)

    with open(matrix_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=matrix_rows[0].keys())
        writer.writeheader(); writer.writerows(matrix_rows)

    print(f"Saved CSVs to {output_dir}")

if __name__ == "__main__":
    main()
