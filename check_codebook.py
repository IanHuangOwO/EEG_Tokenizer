import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import csv
import argparse

from model.factory import build_model_from_config
from IO.dataset import build_dataset_from_config


def calc_effective_rank(matrix):
    if matrix.numel() == 0:
        return 0.0
    try:
        _, S, _ = torch.linalg.svd(matrix.float(), full_matrices=False)
    except Exception:
        return 0.0
    s_sum = S.sum()
    if s_sum == 0:
        return 0.0
    p = S / s_sum
    p = p[p > 0]
    return torch.exp(-torch.sum(p * torch.log(p))).item()


def compute_cosine_sim_matrix(tensor):
    normed = F.normalize(tensor.float(), p=2, dim=-1)
    return torch.matmul(normed, normed.t())


def collect_stage_data(vq_head, stage_idx, f_min, f_max, csv_dir, num_discrete):
    """Compute all metrics for one stage, save CSV, print summary, return data dict."""
    A = vq_head.A.detach().cpu()       # [D, H*r]
    D, Hr = A.shape
    H = vq_head.num_heads
    r = vq_head.r
    A_reshaped = A.view(D, H, r)       # [D, H, r]

    print(f"\n  Stage {stage_idx} ({f_min}-{f_max} Hz)  |  D={D}, H={H}, r={r}")

    # Inter-head similarity
    A_means = A_reshaped.mean(dim=2).t()   # [H, D]
    head_sim = compute_cosine_sim_matrix(A_means).numpy()

    # Singular values per head
    sv_all = []
    for h in range(H):
        _, S, _ = torch.linalg.svd(A_reshaped[:, h, :].float(), full_matrices=False)
        sv_all.append(S.numpy())
    sv_arr = np.stack(sv_all)              # [H, r]

    # Condition number & effective rank
    cond_numbers = sv_arr[:, 0] / (sv_arr[:, -1] + 1e-8)
    eff_ranks = [calc_effective_rank(A_reshaped[:, h, :]) for h in range(H)]

    # Perplexity & sharpness
    probs = vq_head.avg_probs.detach().cpu()        # [H, r, N_discrete]
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    perplexity = torch.exp(entropy).numpy()         # [H, r]
    max_probs = probs.max(dim=-1)[0].numpy()        # [H, r]

    # CSV
    tag = f"stage{stage_idx}_{f_min}-{f_max}Hz"
    rows = []
    for h in range(H):
        rows.append({
            'stage': stage_idx, 'freq_min': f_min, 'freq_max': f_max,
            'head': h,
            'condition_number': float(cond_numbers[h]),
            'effective_rank': float(eff_ranks[h]),
            'avg_singular_value': float(sv_arr[h].mean()),
            'perplexity_mean': float(perplexity[h].mean()),
            'perplexity_min': float(perplexity[h].min()),
            'sharpness_mean': float(max_probs[h].mean()),
            'sharpness_max': float(max_probs[h].max()),
        })
    csv_path = os.path.join(csv_dir, f"{tag}_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    print(f"    cond_number  : mean={cond_numbers.mean():.1f}  max={cond_numbers.max():.1f}")
    print(f"    eff_rank     : mean={np.mean(eff_ranks):.2f} / {r}")
    print(f"    perplexity   : mean={perplexity.mean():.3f}  min={perplexity.min():.3f}  (max={num_discrete})")
    print(f"    sharpness    : mean={max_probs.mean():.3f}  max={max_probs.max():.3f}")

    return {
        'stage_idx': stage_idx, 'f_min': f_min, 'f_max': f_max,
        'H': H, 'r': r, 'num_discrete': num_discrete,
        'head_sim': head_sim,
        'sv_arr': sv_arr,
        'cond_numbers': cond_numbers,
        'eff_ranks': eff_ranks,
        'perplexity': perplexity,
        'max_probs': max_probs,
    }


def plot_all_stages(all_data, viz_dir):
    N = len(all_data)
    stage_labels = [f"Stage {d['stage_idx']}\n({d['f_min']}-{d['f_max']} Hz)" for d in all_data]

    # ── 1. Inter-Head Similarity ──────────────────────────────────────────
    fig, axes = plt.subplots(1, N, figsize=(5 * N, 5))
    if N == 1: axes = [axes]
    for ax, d, label in zip(axes, all_data, stage_labels):
        im = ax.imshow(d['head_sim'], cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, label='Cosine Sim')
        ax.set_title(label)
        ax.set_xlabel("Head"); ax.set_ylabel("Head")
    fig.suptitle("Inter-Head Similarity", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(viz_dir, "1_inter_head_sim.png"), dpi=120)
    plt.close(fig)

    # ── 2. Singular Values ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, N, figsize=(5 * N, 8))
    if N == 1: axes = axes[:, None]
    for col, (d, label) in enumerate(zip(all_data, stage_labels)):
        sv = d['sv_arr']
        r = d['r']
        im = axes[0, col].imshow(sv, cmap='viridis', aspect='auto')
        fig.colorbar(im, ax=axes[0, col], label='SV')
        axes[0, col].set_title(label)
        axes[0, col].set_xlabel("SV Index"); axes[0, col].set_ylabel("Head")

        axes[1, col].plot(sv.mean(axis=0), color='steelblue', label='Mean')
        axes[1, col].fill_between(range(r),
                                  sv.mean(axis=0) - sv.std(axis=0),
                                  sv.mean(axis=0) + sv.std(axis=0),
                                  alpha=0.3, color='steelblue', label='±1σ')
        axes[1, col].set_yscale('log')
        axes[1, col].set_xlabel("SV Index"); axes[1, col].set_ylabel("Value (log)")
        axes[1, col].legend(fontsize=7); axes[1, col].grid(True, which='both', alpha=0.2)
    axes[0, 0].set_ylabel("Head\n(SV Heatmap)")
    axes[1, 0].set_ylabel("Value (log)\n(Mean Decay)")
    fig.suptitle("Singular Value Spectrum", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(viz_dir, "2_singular_values.png"), dpi=120)
    plt.close(fig)

    # ── 3. Geometry: Condition Number + Effective Rank ────────────────────
    fig, axes = plt.subplots(2, N, figsize=(5 * N, 6))
    if N == 1: axes = axes[:, None]
    for col, (d, label) in enumerate(zip(all_data, stage_labels)):
        H = d['H']; r = d['r']
        cond = d['cond_numbers']; eff = d['eff_ranks']
        axes[0, col].bar(range(H), cond, color='tomato', alpha=0.8)
        axes[0, col].axhline(np.median(cond), color='k', linestyle='--',
                             label=f'med={np.median(cond):.1f}')
        axes[0, col].set_title(label)
        axes[0, col].set_xlabel("Head"); axes[0, col].legend(fontsize=7)

        axes[1, col].bar(range(H), eff, color='steelblue', alpha=0.8)
        axes[1, col].axhline(r, color='k', linestyle='--', label=f'max={r}')
        axes[1, col].set_xlabel("Head"); axes[1, col].legend(fontsize=7)
    axes[0, 0].set_ylabel("κ(A_h)  (Cond. Number)")
    axes[1, 0].set_ylabel("Eff. Rank")
    fig.suptitle("Geometry Health", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(viz_dir, "3_geometry.png"), dpi=120)
    plt.close(fig)

    # ── 4. Perplexity ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, N, figsize=(5 * N, 8))
    if N == 1: axes = axes[:, None]
    for col, (d, label) in enumerate(zip(all_data, stage_labels)):
        perp = d['perplexity']; nd = d['num_discrete']
        im = axes[0, col].imshow(perp, cmap='magma', aspect='auto', vmin=1.0, vmax=float(nd))
        fig.colorbar(im, ax=axes[0, col], label='Perplexity')
        axes[0, col].set_title(label)
        axes[0, col].set_xlabel("Sub-dim (r)"); axes[0, col].set_ylabel("Head")

        axes[1, col].hist(perp.ravel(), bins=30, color='mediumpurple', edgecolor='white')
        axes[1, col].axvline(float(nd), color='red', linestyle='--', label=f'max ({nd})')
        axes[1, col].axvline(perp.mean(), color='k', linestyle='--', label=f'mean ({perp.mean():.2f})')
        axes[1, col].set_xlabel("Perplexity"); axes[1, col].legend(fontsize=7)
    axes[0, 0].set_ylabel("Head\n(Heatmap)")
    axes[1, 0].set_ylabel("Count\n(Distribution)")
    fig.suptitle("Code Usage (Perplexity)", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(viz_dir, "4_perplexity.png"), dpi=120)
    plt.close(fig)

    # ── 5. Sharpness ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, N, figsize=(5 * N, 4))
    if N == 1: axes = [axes]
    for ax, d, label in zip(axes, all_data, stage_labels):
        im = ax.imshow(d['max_probs'], cmap='hot', aspect='auto', vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label='Max Prob')
        ax.set_title(label)
        ax.set_xlabel("Sub-dim (r)"); ax.set_ylabel("Head")
    fig.suptitle("Code Sharpness (Max Prob per Expert)", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(viz_dir, "5_sharpness.png"), dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='AttnVQ Codebook Health Check')
    parser.add_argument('--config', type=str, default='config/config.json')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint. Auto-detected from config if omitted.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config, 'r') as f:
        config = json.load(f)

    model_name = config['training_params']['model_name']
    model_type = config['training_params']['model_type']

    if model_type != "AttnVQ":
        print(f"Error: script is for AttnVQ models, found: {model_type}")
        return

    # --- Determine n_fft: prefer checkpoint's freq_mask shape, fall back to config ---
    ckpt_path = args.checkpoint or f"output/{model_name}/tokenizer/best_tokenizer.pth"
    n_fft = None
    if os.path.exists(ckpt_path):
        _sd = torch.load(ckpt_path, map_location='cpu')
        _sd = _sd.get('model_state_dict', _sd)
        if 'freq_mask_0' in _sd:
            n_fft = (_sd['freq_mask_0'].shape[0] - 1) * 2
    if n_fft is None:
        patch_len = config['model_params']['AttnVQ']['preprocess']['patch_length']
        trial_len = config['model_params']['AttnVQ']['preprocess']['trial_length']
        n_fft = (trial_len // patch_len) * patch_len

    model = build_model_from_config(config, n_fft_trial=n_fft).to(device)

    # --- Load checkpoint ---
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt.get('model_state_dict', ckpt)
        msg = model.load_state_dict(sd, strict=False)
        print(f"  Missing: {len(msg.missing_keys)}  Unexpected: {len(msg.unexpected_keys)}")
    else:
        print(f"WARNING: checkpoint not found at {ckpt_path}, using random weights.")

    model.eval()

    viz_dir = f"output/{model_name}/visualization/codebook_health"
    csv_dir = os.path.join(viz_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    num_stages = len(model.vq_heads)
    print(f"\nAnalysing {num_stages} VQ stages...")

    all_data = []
    for i, (vq_head, h_cfg) in enumerate(zip(model.vq_heads, model.decoder_heads_config)):
        f_min, f_max = h_cfg["freq_range"]
        num_discrete = h_cfg.get("vq_num_discrete", vq_head.num_discrete)
        data = collect_stage_data(vq_head, i, f_min, f_max, csv_dir, num_discrete)
        all_data.append(data)

    print(f"\nSaving combined plots...")
    plot_all_stages(all_data, viz_dir)
    print(f"Done. 5 plots saved to: {viz_dir}")


if __name__ == "__main__":
    main()
