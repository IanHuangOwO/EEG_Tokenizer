"""
Codebook health analysis: geometry, perplexity, sharpness, inter-head similarity.
Does not require a specific trial — inspects model weights directly.
"""

import os
import csv

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from viz import load_config, load_model, resolve_output_dir


# ── Pure helpers ──────────────────────────────────────────────────────────────

def calc_effective_rank(matrix: torch.Tensor) -> float:
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


def compute_cosine_sim_matrix(tensor: torch.Tensor) -> torch.Tensor:
    normed = F.normalize(tensor.float(), p=2, dim=-1)
    return torch.matmul(normed, normed.t())


def collect_stage_data(vq_head, stage_idx: int, f_min: float, f_max: float,
                       csv_dir: str, num_discrete: int) -> dict:
    A = vq_head.A.detach().cpu()
    D, _ = A.shape
    H = vq_head.num_heads
    r = vq_head.r
    A3 = A.view(D, H, r)

    print(f"  Stage {stage_idx}  D={D} H={H} r={r}")

    head_sim = compute_cosine_sim_matrix(A3.mean(dim=2).t()).numpy()

    sv_arr = np.stack([
        torch.linalg.svd(A3[:, h, :].float(), full_matrices=False)[1].numpy()
        for h in range(H)
    ])

    cond_numbers = sv_arr[:, 0] / (sv_arr[:, -1] + 1e-8)
    eff_ranks    = [calc_effective_rank(A3[:, h, :]) for h in range(H)]

    probs      = vq_head.avg_probs.detach().cpu()
    perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10), dim=-1)).numpy()
    max_probs  = probs.max(dim=-1)[0].numpy()

    rows = [
        {
            'stage': stage_idx, 'freq_min': f_min, 'freq_max': f_max, 'head': h,
            'condition_number':   float(cond_numbers[h]),
            'effective_rank':     float(eff_ranks[h]),
            'avg_singular_value': float(sv_arr[h].mean()),
            'perplexity_mean':    float(perplexity[h].mean()),
            'perplexity_min':     float(perplexity[h].min()),
            'sharpness_mean':     float(max_probs[h].mean()),
            'sharpness_max':      float(max_probs[h].max()),
        }
        for h in range(H)
    ]
    csv_path = os.path.join(csv_dir, f"stage{stage_idx}_{f_min}-{f_max}Hz_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"    cond  mean={cond_numbers.mean():.1f}  max={cond_numbers.max():.1f}")
    print(f"    rank  mean={np.mean(eff_ranks):.2f} / {r}")
    print(f"    perp  mean={perplexity.mean():.3f}  min={perplexity.min():.3f}  (max={num_discrete})")
    print(f"    sharp mean={max_probs.mean():.3f}  max={max_probs.max():.3f}")

    return dict(
        stage_idx=stage_idx, f_min=f_min, f_max=f_max,
        H=H, r=r, num_discrete=num_discrete,
        head_sim=head_sim, sv_arr=sv_arr,
        cond_numbers=cond_numbers, eff_ranks=eff_ranks,
        perplexity=perplexity, max_probs=max_probs,
    )


def plot_all_stages(all_data: list, viz_dir: str):
    N = len(all_data)
    labels = [f"Stage {d['stage_idx']}\n({d['f_min']}-{d['f_max']} Hz)" for d in all_data]

    # 1. Inter-Head Similarity
    fig, axes = plt.subplots(1, N, figsize=(5 * N, 5))
    if N == 1: axes = [axes]
    for ax, d, lbl in zip(axes, all_data, labels):
        im = ax.imshow(d['head_sim'], cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, label='Cosine Sim')
        ax.set_title(lbl); ax.set_xlabel("Head"); ax.set_ylabel("Head")
    fig.suptitle("Inter-Head Similarity", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(viz_dir, "1_inter_head_sim.png"), dpi=120)
    plt.close(fig)

    # 2. Singular Values
    fig, axes = plt.subplots(2, N, figsize=(5 * N, 8))
    if N == 1: axes = axes[:, None]
    for col, (d, lbl) in enumerate(zip(all_data, labels)):
        sv = d['sv_arr']; r = d['r']
        im = axes[0, col].imshow(sv, cmap='viridis', aspect='auto')
        fig.colorbar(im, ax=axes[0, col], label='SV')
        axes[0, col].set_title(lbl)
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

    # 3. Geometry
    fig, axes = plt.subplots(2, N, figsize=(5 * N, 6))
    if N == 1: axes = axes[:, None]
    for col, (d, lbl) in enumerate(zip(all_data, labels)):
        H = d['H']; r = d['r']; cond = d['cond_numbers']; eff = d['eff_ranks']
        axes[0, col].bar(range(H), cond, color='tomato', alpha=0.8)
        axes[0, col].axhline(np.median(cond), color='k', linestyle='--',
                             label=f'med={np.median(cond):.1f}')
        axes[0, col].set_title(lbl); axes[0, col].set_xlabel("Head")
        axes[0, col].legend(fontsize=7)
        axes[1, col].bar(range(H), eff, color='steelblue', alpha=0.8)
        axes[1, col].axhline(r, color='k', linestyle='--', label=f'max={r}')
        axes[1, col].set_xlabel("Head"); axes[1, col].legend(fontsize=7)
    axes[0, 0].set_ylabel("κ(A_h)  (Cond. Number)")
    axes[1, 0].set_ylabel("Eff. Rank")
    fig.suptitle("Geometry Health", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(viz_dir, "3_geometry.png"), dpi=120)
    plt.close(fig)

    # 4. Perplexity
    fig, axes = plt.subplots(2, N, figsize=(5 * N, 8))
    if N == 1: axes = axes[:, None]
    for col, (d, lbl) in enumerate(zip(all_data, labels)):
        perp = d['perplexity']; nd = d['num_discrete']
        im = axes[0, col].imshow(perp, cmap='magma', aspect='auto', vmin=1.0, vmax=float(nd))
        fig.colorbar(im, ax=axes[0, col], label='Perplexity')
        axes[0, col].set_title(lbl)
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

    # 5. Sharpness
    fig, axes = plt.subplots(1, N, figsize=(5 * N, 4))
    if N == 1: axes = [axes]
    for ax, d, lbl in zip(axes, all_data, labels):
        im = ax.imshow(d['max_probs'], cmap='hot', aspect='auto', vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label='Max Prob')
        ax.set_title(lbl); ax.set_xlabel("Sub-dim (r)"); ax.set_ylabel("Head")
    fig.suptitle("Code Sharpness (Max Prob per Expert)", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(viz_dir, "5_sharpness.png"), dpi=120)
    plt.close(fig)


# ── Module interface ──────────────────────────────────────────────────────────

def add_args(parser):
    pass


def run(config, output_dir, args, model=None, dataset=None, trial_idx=None, subject_id=None):
    if config['training_params']['pretrain'].get('model_type') != 'MeFSQ':
        print("  [codebook] Skipping: not an MeFSQ model.")
        return

    viz_dir = os.path.join(output_dir, 'codebook')
    csv_dir = os.path.join(viz_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)

    print(f"  [codebook] Analysing MeFSQ VQ stage...")
    all_data = [collect_stage_data(model.mefsq_routed, 0, 0.0, 0.0, csv_dir, model.mefsq_routed.num_discrete)]

    plot_all_stages(all_data, viz_dir)
    print(f"  [codebook] -> {viz_dir}")


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MeFSQ Codebook Health Check')
    parser.add_argument('--config',     default='config/analysis.json')
    parser.add_argument('--checkpoint', default=None)
    add_args(parser)
    args = parser.parse_args()

    cfg        = load_config(args.config)
    checkpoint = args.checkpoint or cfg.get('checkpoint', '')
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl        = load_model(cfg, checkpoint, device)
    out        = resolve_output_dir(cfg, 'check')
    run(cfg, out, args, model=mdl)
