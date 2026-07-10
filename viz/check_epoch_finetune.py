"""
Finetune-stage epoch snapshot.
  recon/  power topomap (raw vs backbone-recon) + per-head activation topomap grid —
          identical in spirit to viz.check_epoch (pretrain), reusing the same backbone
          forward contract (recon, indices, v_q, gate_mask, lb_loss) via model.backbone.
  attn/   the finetune classification head's own attention (PerChannelHeadAttn) — each
          CHANNEL independently softmaxes over VQ heads (weights sum to 1 per channel, not
          jointly), computed after averaging over patches. Shown as a channel x head
          heatmap, and as one topomap per head.
"""
import os
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from viz.draw import project_coords_2d, draw_topomap, extract_head_psd


def _patchify(x, patch_len):
    """x: [C, T] -> x_patches [1, C, P, L], time_idx [1, P]."""
    C, T = x.shape
    P = T // patch_len
    x_patches = x[:, :P * patch_len].reshape(C, P, patch_len).unsqueeze(0)
    time_idx = torch.arange(P, dtype=torch.long).unsqueeze(0)
    return x_patches, time_idx


def _build_pad_mask(valid_channels, valid_length, P, patch_len):
    """Mirrors FinetuneCollate: a patch counts valid only if fully inside the real
    (non-padded) length. Returns [1, C, P] bool."""
    valid_length = valid_length.item() if torch.is_tensor(valid_length) else valid_length
    n_valid_patches = min(valid_length // patch_len, P)
    patch_valid = torch.arange(P) < n_valid_patches                    # [P]
    return (valid_channels.unsqueeze(-1) & patch_valid.unsqueeze(0)).unsqueeze(0)  # [1, C, P]


def _save_grid(values, sorted_ord, panel_scores, pos2d, cmap, suptitle, out_path, label_fmt):
    """values: [C, N]. One topomap per column of `values`, ordered by `sorted_ord`."""
    N = values.shape[1]
    n_tc = min(16, N)
    n_tr = math.ceil(N / n_tc)
    fig, axes = plt.subplots(n_tr, n_tc, figsize=(max(28.0, n_tc * 2.8), max(4.0, n_tr * 3.0)), squeeze=False)
    for pos, i in enumerate(sorted_ord):
        r, c = divmod(pos, n_tc)
        v = values[:, i]
        draw_topomap(axes[r][c], pos2d, v, cmap=cmap, vmin=v.min(), vmax=v.max())
        axes[r][c].set_title(label_fmt(i, panel_scores), fontsize=5.5)
    for pos in range(N, n_tr * n_tc):
        r, c = divmod(pos, n_tc)
        axes[r][c].axis('off')
    fig.suptitle(suptitle, fontsize=12, fontweight='bold')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def run(config, output_dir, model, dataset, trial_idx, subject_id=None, epoch=None,
        patch_len=None, cmap='YlOrRd'):
    """
    model:   the MeFSQFinetune instance — model.backbone reused for recon/, model(...) itself
             (the full finetune forward, PerChannelHeadAttn included) used for attn/.
    dataset: a FinetuneDataset (yields raw [C,T] trials, not pre-patched).
    """
    backbone = model.backbone
    device = next(model.parameters()).device
    pp = config.get('preprocess_params', {})
    patch_len = patch_len or pp.get('patch_length', 100)
    epoch_tag = f'_ep{epoch:04d}' if epoch is not None else ''

    x_raw, coords, label, valid_channels, valid_length = dataset[trial_idx]
    x_patches, time_idx = _patchify(x_raw, patch_len)
    x_in = x_patches.to(device)
    c_in = coords.unsqueeze(0).to(device)
    t_in = time_idx.to(device)
    pad_mask = _build_pad_mask(valid_channels, valid_length, x_patches.shape[2], patch_len).to(device)

    pos2d = project_coords_2d(coords.numpy())
    channel_names = dataset.base_dataset.channel_names

    was_training = model.training
    model.eval()

    # attn is computed once and used to rank heads in BOTH recon/ and attn/ grids: sum,
    # over channels, of the per-channel softmax weight each head received — a head that
    # many channels lean on heavily ranks first, regardless of which grid is being drawn.
    _, attn, _ = model(x_in, c_in, time_idx=t_in, pad_mask=pad_mask)
    attn_np = attn[0].detach().cpu().numpy()  # [C, H]
    head_score = attn_np.sum(axis=0)
    head_order = np.argsort(head_score)[::-1]

    # ── recon/ ──────────────────────────────────────────────────────────
    recon_dir = os.path.join(output_dir, 'recon')
    os.makedirs(recon_dir, exist_ok=True)

    recon, indices, v_q, gate_mask, _ = backbone(x_in, c_in, time_idx=t_in, bool_masked_pos=None)

    raw_np   = x_in[0].reshape(x_in.shape[1], -1).cpu().numpy()
    recon_np = recon[0].reshape(recon.shape[1], -1).detach().cpu().numpy()

    raw_power   = (raw_np   ** 2).mean(axis=-1)
    recon_power = (recon_np ** 2).mean(axis=-1)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f"Power Topo — Sub {subject_id}, Trial {trial_idx}{epoch_tag} [finetune]",
                 fontsize=10, fontweight='bold')
    for ax, power, title in zip(axes, [raw_power, recon_power], ['Raw', 'Recon']):
        im = draw_topomap(ax, pos2d, power, cmap=cmap)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02).set_label('Power', fontsize=6)
    fig.savefig(os.path.join(recon_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_power_topo.png"),
                dpi=120, bbox_inches='tight')
    plt.close(fig)

    stage_psd, head_norms, head_affinity, routing_score = extract_head_psd(backbone, x_in, c_in, t_in)
    psd_ch_h   = stage_psd[0]
    _save_grid(
        psd_ch_h, head_order, head_score, pos2d, cmap,
        f"Head Activation — Sub {subject_id}, Trial {trial_idx}{epoch_tag} [finetune]",
        os.path.join(recon_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_topo_heads.png"),
        label_fmt=lambda h, scores: f'H{h}\n{scores[h]:.3f}',
    )

    # ── attn/ ───────────────────────────────────────────────────────────
    attn_dir = os.path.join(output_dir, 'attn')
    os.makedirs(attn_dir, exist_ok=True)

    H = attn_np.shape[1]
    fig2, ax2 = plt.subplots(figsize=(max(6.0, H * 0.25), max(6.0, len(channel_names) * 0.18)))
    im2 = ax2.imshow(attn_np, aspect='auto', cmap='viridis')
    ax2.set_yticks(range(len(channel_names)))
    ax2.set_yticklabels(channel_names, fontsize=5)
    ax2.set_xlabel('VQ head')
    ax2.set_ylabel('Channel')
    ax2.set_title(f"Per-Channel Head Attention — Sub {subject_id}, Trial {trial_idx}{epoch_tag} [finetune]",
                  fontsize=10, fontweight='bold')
    fig2.colorbar(im2, ax=ax2, fraction=0.03, pad=0.02, label='Attention weight')
    fig2.savefig(os.path.join(attn_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_heatmap.png"),
                 dpi=120, bbox_inches='tight')
    plt.close(fig2)

    _save_grid(
        attn_np, head_order, head_score, pos2d, 'viridis',
        f"Per-Channel Head Attention (Topo) — Sub {subject_id}, Trial {trial_idx}{epoch_tag} [finetune]",
        os.path.join(attn_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_topo_heads_attn.png"),
        label_fmt=lambda h, scores: f'H{h}\n{scores[h]:.3f}',
    )

    model.train(was_training)
