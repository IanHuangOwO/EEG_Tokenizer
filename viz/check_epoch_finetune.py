"""
Finetune-stage epoch snapshot.
  recon/  power topomap (raw vs backbone-recon) + per-head activation topomap grid +
          per-head PSD grid — identical in spirit to viz.check_epoch (pretrain), reusing
          the same backbone forward contract (SimpleNamespace: recon, indices_routed/shared,
          v_q_routed/shared, gate_mask_routed, lb_loss) via model.backbone.
  attn/   the finetune classification head's own attention (PerChannelHeadAttn), three stages:
          attn_c [C] — softmax over channels (weights sum to 1 across the whole trial),
          shown as a barh next to the head heatmap, same channel order/axis. attn_h [C, H] —
          each CHANNEL independently softmaxes over VQ heads (weights sum to 1 per channel,
          not jointly), shown as a channel x head heatmap and one topomap per head. attn_n
          [C, H, N] — per channel & head, softmax over patches (temporal attention), shown
          as a channel x patch heatmap for the top-scoring head.
"""
import os
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from viz.draw import project_coords_2d, draw_topomap, extract_head_psd, extract_head_spectra


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


def _title_style(i, n_shared):
    """Highlight shared-expert heads (indices [0, n_shared)) in the panel title."""
    is_shared = n_shared is not None and i < n_shared
    return dict(color='crimson' if is_shared else 'black', fontweight='bold' if is_shared else 'normal')


def _save_grid(values, sorted_ord, panel_scores, pos2d, cmap, suptitle, out_path, label_fmt, n_shared=None):
    """values: [C, N]. One topomap per column of `values`, ordered by `sorted_ord`.
    n_shared: if given, heads [0, n_shared) are highlighted as shared-expert heads."""
    N = values.shape[1]
    n_tc = min(16, N)
    n_tr = math.ceil(N / n_tc)
    fig, axes = plt.subplots(n_tr, n_tc, figsize=(max(28.0, n_tc * 2.8), max(4.0, n_tr * 3.0)), squeeze=False)
    for pos, i in enumerate(sorted_ord):
        r, c = divmod(pos, n_tc)
        v = values[:, i]
        draw_topomap(axes[r][c], pos2d, v, cmap=cmap, vmin=v.min(), vmax=v.max())
        axes[r][c].set_title(label_fmt(i, panel_scores), fontsize=5.5, **_title_style(i, n_shared))
    for pos in range(N, n_tr * n_tc):
        r, c = divmod(pos, n_tc)
        axes[r][c].axis('off')
    fig.suptitle(suptitle, fontsize=12, fontweight='bold')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def _save_psd_grid(psd, freqs, sorted_ord, panel_scores, cmap, suptitle, out_path, label_fmt, freq_label='Hz', n_shared=None):
    """psd: [H, C, F]. One channel x frequency heatmap per head, ordered by `sorted_ord`.
    n_shared: if given, heads [0, n_shared) are highlighted as shared-expert heads."""
    H = psd.shape[0]
    n_tc = min(4, H)
    n_tr = math.ceil(H / n_tc)
    freq_ticks = np.linspace(freqs[0], freqs[-1], 5)
    fig, axes = plt.subplots(n_tr, n_tc, figsize=(max(28.0, n_tc * 7.0), max(4.0, n_tr * 3.4)), squeeze=False)
    for pos, h in enumerate(sorted_ord):
        r, c = divmod(pos, n_tc)
        p_h = psd[h]  # [C, F]
        axes[r][c].imshow(p_h, aspect='auto', cmap=cmap, origin='lower',
                           extent=[freqs[0], freqs[-1], 0, p_h.shape[0]])
        axes[r][c].set_title(label_fmt(h, panel_scores), fontsize=5.5, **_title_style(h, n_shared))
        axes[r][c].set_xticks(freq_ticks)
        axes[r][c].set_xticklabels([f'{f:.0f}' for f in freq_ticks], fontsize=5)
        axes[r][c].set_xlabel(freq_label, fontsize=6)
        axes[r][c].set_yticks([])
    for pos in range(H, n_tr * n_tc):
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
    # attn_h/attn_n and the recon/ PSD grids below (extract_head_psd/extract_head_spectra)
    # both cover ALL experts now (shared first, at indices [0, n_shared), then routed —
    # same order as encode_pre_vq), so one shared ranking works for both.
    _, attn_h, attn_n, attn_c = model(x_in, c_in, time_idx=t_in, pad_mask=pad_mask)
    attn_np = attn_h[0].detach().cpu().numpy()      # [C, H_total]
    attn_n_np = attn_n[0].detach().cpu().numpy()     # [C, H_total, N]
    attn_c_np = attn_c[0].detach().cpu().numpy()     # [C]
    head_score = attn_np.sum(axis=0)
    head_order = np.argsort(head_score)[::-1]
    n_shared = backbone.n_shared_experts  # heads [0, n_shared) are the shared-expert pool

    # ── recon/ ──────────────────────────────────────────────────────────
    recon_dir = os.path.join(output_dir, 'recon')
    os.makedirs(recon_dir, exist_ok=True)

    out = backbone(x_in, c_in, time_idx=t_in, bool_masked_pos=None)

    raw_np   = x_in[0].reshape(x_in.shape[1], -1).cpu().numpy()
    recon_np = out.recon[0].reshape(out.recon.shape[1], -1).detach().cpu().numpy()

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
        n_shared=n_shared,
    )

    # Per-head PSD grid — channel x frequency heatmap of each head's own decoded
    # reconstruction, cropped to the same bandpass range preprocessing already restricted
    # the signal to. Ranked by the same attn head_order as the other grids in this file.
    l_freq, h_freq = pp.get('l_freq'), pp.get('h_freq')
    fs = pp.get('target_freq')
    psd_hcf, freqs, _ = extract_head_spectra(backbone, x_in, c_in, t_in, fs=fs, freq_resolution=0.2)
    if l_freq is not None and h_freq is not None:
        band = (freqs >= l_freq) & (freqs <= h_freq)
        freqs = freqs[band]
        psd_hcf = psd_hcf[:, :, band]
    _save_psd_grid(
        psd_hcf, freqs, head_order, head_score, cmap,
        f"Head PSD — Sub {subject_id}, Trial {trial_idx}{epoch_tag} [finetune]",
        os.path.join(recon_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_psd_heads.png"),
        label_fmt=lambda h, scores: f'H{h}\n{scores[h]:.3f}',
        freq_label='Hz' if fs else 'cyc/patch',
        n_shared=n_shared,
    )

    # ── attn/ ───────────────────────────────────────────────────────────
    attn_dir = os.path.join(output_dir, 'attn')
    os.makedirs(attn_dir, exist_ok=True)

    H = attn_np.shape[1]
    n_tc = min(8, H)
    n_tr = math.ceil(H / n_tc)
    fig2 = plt.figure(figsize=(max(10.0, H * 0.25) + n_tc * 2.2 + 1.2, max(6.0, len(channel_names) * 0.18, n_tr * 2.4)))
    gs = fig2.add_gridspec(1, 3, width_ratios=[1.2, H * 0.25, n_tc * 2.2])

    # far left: stage-3 channel attention (attn_c) — one weight per channel, rows aligned
    # with the head heatmap next to it (same channel order, same y-axis direction)
    ax_c = fig2.add_subplot(gs[0, 0])
    ax_c.barh(range(len(channel_names)), attn_c_np, color='steelblue')
    ax_c.set_ylim(-0.5, len(channel_names) - 0.5)
    ax_c.invert_yaxis()  # channel 0 at top, matching imshow's default row order
    ax_c.set_yticks(range(len(channel_names)))
    ax_c.set_yticklabels(channel_names, fontsize=5)
    ax_c.set_xlabel('Attn wt', fontsize=7)
    ax_c.set_title('Channel Attn', fontsize=9, fontweight='bold')

    # middle: channel x head heatmap
    ax2 = fig2.add_subplot(gs[0, 1], sharey=ax_c)
    im2 = ax2.imshow(attn_np, aspect='auto', cmap='viridis')
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xticks(range(H))
    ax2.set_xticklabels([f'H{h}' for h in range(H)], fontsize=4, rotation=90)
    for h, tick in enumerate(ax2.get_xticklabels()):
        style = _title_style(h, n_shared)
        tick.set_color(style['color'])
        tick.set_fontweight(style['fontweight'])
    ax2.set_xlabel('VQ head (crimson = shared expert)')
    ax2.set_title('Per-Channel Head Attention', fontsize=9, fontweight='bold')
    fig2.colorbar(im2, ax=ax2, fraction=0.05, pad=0.02, label='Attn weight')

    # right: grid of channel x patch temporal-attention subplots, one per head, ordered by head_score
    gs_right = gs[0, 2].subgridspec(n_tr, n_tc, hspace=0.6, wspace=0.3)
    for pos, h in enumerate(head_order):
        r, c = divmod(pos, n_tc)
        axr = fig2.add_subplot(gs_right[r, c])
        axr.imshow(attn_n_np[:, h, :], aspect='auto', cmap='viridis')
        axr.set_yticks(range(len(channel_names)))
        axr.set_yticklabels(channel_names if c == 0 else [], fontsize=3)
        axr.set_xticks([])
        axr.set_title(f'H{h} ({head_score[h]:.3f})', fontsize=5.5, **_title_style(h, n_shared))
    for pos in range(H, n_tr * n_tc):
        r, c = divmod(pos, n_tc)
        fig2.add_subplot(gs_right[r, c]).axis('off')

    fig2.suptitle(f"Per-Channel Head + Temporal Attention — Sub {subject_id}, Trial {trial_idx}{epoch_tag} [finetune]",
                  fontsize=10, fontweight='bold')
    fig2.savefig(os.path.join(attn_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_heatmap.png"),
                 dpi=120, bbox_inches='tight')
    plt.close(fig2)

    _save_grid(
        attn_np, head_order, head_score, pos2d, 'viridis',
        f"Per-Channel Head Attention (Topo) — Sub {subject_id}, Trial {trial_idx}{epoch_tag} [finetune]",
        os.path.join(attn_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_topo_heads_attn.png"),
        label_fmt=lambda h, scores: f'H{h}\n{scores[h]:.3f}',
        n_shared=n_shared,
    )

    model.train(was_training)
