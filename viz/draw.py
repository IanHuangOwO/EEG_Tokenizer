"""
Drawing primitives: topomap, heatmap, temporal band-filter.
Model-agnostic except extract_head_psd which runs a partial forward pass.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle
import torch
import torch.nn.functional as F


# ── Topomap ───────────────────────────────────────────────────────────────────

def project_coords_2d(coords: np.ndarray) -> np.ndarray:
    """Azimuthal equidistant projection of 3-D electrode coords to 2-D. coords: [C, 3]"""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-8
    xn, yn, zn = x / r, y / r, z / r
    theta = np.arccos(np.clip(zn, -1, 1))
    phi = np.arctan2(xn, yn)
    rho = theta / (np.pi / 2)
    return np.stack([rho * np.sin(phi), rho * np.cos(phi)], axis=1)


def draw_topomap(ax, pos2d: np.ndarray, values: np.ndarray,
                 n_grid: int = 100, cmap: str = 'YlOrRd',
                 vmin=None, vmax=None):
    """
    Interpolated topomap on a circular head outline.
    pos2d: [C, 2], values: [C]
    Returns the pcolormesh artist (for colorbar attachment).
    """
    px, py = pos2d[:, 0], pos2d[:, 1]
    vmin = values.min() if vmin is None else vmin
    vmax = values.max() if vmax is None else vmax

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
    ax.plot([0, 0],        [1.0, 1.15],   'k-', linewidth=1.2)
    ax.plot([-1.0, -1.1],  [0.1, 0.0],    'k-', linewidth=1.0)
    ax.plot([1.0,  1.1],   [0.1, 0.0],    'k-', linewidth=1.0)
    ax.scatter(px, py, s=6, c='k', zorder=5, alpha=0.5)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal'); ax.axis('off')
    return im


def save_topomap_grid(output_path: str, pos2d: np.ndarray, psd_ch_h: np.ndarray,
                      display_order: np.ndarray, importance: np.ndarray,
                      subject_id, trial_idx: int, s_idx: int,
                      f_min: float, f_max: float, cmap: str, title_tag: str):
    """Save a sorted grid of per-head topomaps to disk."""
    n_show = len(display_order)
    n_cols = min(16, n_show)
    n_rows = int(np.ceil(n_show / n_cols))
    rank_norm = np.arange(n_show) / max(n_show - 1, 1)

    vmin = psd_ch_h[:, display_order].min()
    vmax = psd_ch_h[:, display_order].max()

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.8 * n_cols, 3.2 * n_rows),
                             squeeze=False)
    im = None
    for pos, h_orig in enumerate(display_order):
        row, col = divmod(pos, n_cols)
        im = draw_topomap(axes[row][col], pos2d, psd_ch_h[:, h_orig],
                          cmap=cmap, vmin=vmin, vmax=vmax)
        title_color = plt.cm.RdYlGn(1.0 - rank_norm[pos])[:3]
        tag = f'#{pos + 1}'
        axes[row][col].set_title(f"{tag} H{h_orig}\n{importance[h_orig]:.3f}",
                                 fontsize=6, color=title_color, fontweight='bold')

    for pos in range(n_show, n_rows * n_cols):
        row, col = divmod(pos, n_cols)
        axes[row][col].axis('off')

    if im is not None:
        fig.subplots_adjust(right=0.88, hspace=0.4, wspace=0.05)
        cb = fig.colorbar(im, cax=fig.add_axes([0.90, 0.15, 0.012, 0.7]))
        cb.set_label('PSD power', fontsize=9)

    fig.suptitle(
        f"Stage {s_idx}  ({f_min}–{f_max} Hz) — {n_show} heads sorted by reconstruction power  [{title_tag}]\n"
        f"Sub {subject_id}, Trial {trial_idx}  (left->right, top->bottom: most->least important)",
        fontsize=11, fontweight='bold',
    )
    fig.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def extract_head_psd(model, x: torch.Tensor, coords: torch.Tensor,
                     time_idx: torch.Tensor = None):
    """
    Per-head per-channel activation norm, combining BOTH MoE pools (shared experts first,
    at indices [0, n_shared_experts), then routed) — same order as encode_pre_vq, so this
    lines up with the finetune head's attn_h/attn_n head axis.

    Computed in up-projected embed_dim space (z_per_head = v_q @ vq_proj), not raw v_q —
    routed/shared pools can have different r (quantizer vocab width), so their raw v_q
    can't be concatenated/compared directly, but both always up-project to the same
    embed_dim, which is also the space the decoder actually reads.

    Returns ([psd_ch_h], head_norms, head_affinity, routing_score):
      psd_ch_h      — [C, H_total] numpy array, mean per-channel decoded activation norm per head
                       (fused per-patch VQ has no per-channel axis pre-decode, so this decodes
                       each pool through its own decoder to recover a per-channel magnitude,
                       rather than the pre-decode embedding norm used before the channel fusion)
      head_norms    — [H_total] numpy array
      head_affinity — [H_total, H_total] cosine similarity
      routing_score — [H_total] fraction of patches selecting each head (model.shared_weight for shared, always active but down-weighted)
    """
    B, C, N, L = x.shape
    out = model(x, coords=coords, time_idx=time_idx, bool_masked_pos=None)

    z_shared = torch.einsum('mhr,hdr->mhd', out.v_q_shared, model.vq_proj_shared)  # [M, Hs, D]
    z_routed = torch.einsum('mhr,hdr->mhd', out.v_q_routed, model.vq_proj_routed)  # [M, Hr, D]
    z_all    = torch.cat([z_shared, z_routed], dim=1)                              # [M, H_total, D]
    H = z_all.shape[1]

    head_norms    = z_all.norm(dim=-1).mean(dim=0).cpu().numpy()  # [H_total]
    v_mean_h      = z_all.mean(dim=0)
    v_mean_n      = F.normalize(v_mean_h, dim=-1)
    head_affinity = (v_mean_n @ v_mean_n.T).cpu().numpy()

    recon_shared = model.decoder_shared(z_shared)               # [M, Hs, C*patch_len]
    recon_routed = model.decoder_routed(z_routed)               # [M, Hr, C*patch_len]
    recon_all    = torch.cat([recon_shared, recon_routed], dim=1)  # [M, H_total, C*patch_len]
    patch_len    = recon_all.shape[-1] // C
    recon_all    = recon_all.reshape(B, N, H, C, patch_len)
    psd_ch_h     = recon_all.norm(dim=-1).mean(dim=1)[0].permute(1, 0).cpu().numpy()  # [C, H_total]

    # routing importance: shared experts are always active, scaled by their fixed contribution
    # weight to recon (model.shared_weight) rather than a flat 1.0 — otherwise they'd always
    # rank as "most important" even though their recon contribution is deliberately down-weighted;
    # routed experts by fraction of patches that selected them
    routing_shared = torch.full((model.n_shared_experts,), model.shared_weight, device=z_all.device)
    routing_routed = (out.gate_mask_routed.detach() > 0).float().mean(dim=0)
    routing_score  = torch.cat([routing_shared, routing_routed]).cpu().numpy()  # [H_total]

    return [psd_ch_h], head_norms, head_affinity, routing_score


@torch.no_grad()
def extract_head_spectra(model, x: torch.Tensor, coords: torch.Tensor,
                          time_idx: torch.Tensor = None, fs: float = None,
                          freq_resolution: float = None):
    """
    Per-head, per-channel power spectrum of that head's OWN decoded reconstruction
    (v_q -> vq_proj -> decoder, per head, un-gated so every routed head shows what it would
    reconstruct if selected — shared heads have no gating to begin with) — not the
    shared-embedding VQ activation norm extract_head_psd reports, an actual frequency-domain
    view of each head's specialization. Combines both MoE pools (shared experts first, at
    indices [0, n_shared_experts), then routed) — same order as encode_pre_vq.
    Returns (psd [H_total, C, F] numpy, freqs [F] numpy, routing_score [H_total] numpy).
    fs: sample rate in Hz for the freq axis; if None, freqs are cycles/patch (bin index).
    freq_resolution: Hz per bin via zero-padded FFT (n_fft = fs / freq_resolution) — the
    patch itself (L samples) is far shorter than what a fine resolution needs, so this is
    padding for display resolution, not real added information beyond the L-sample window.
    """
    B, C, N, L = x.shape
    out = model(x, coords=coords, time_idx=time_idx, bool_masked_pos=None)

    z_shared = torch.einsum('mhr,hdr->mhd', out.v_q_shared,     model.vq_proj_shared)  # [M, Hs, D]
    z_routed = torch.einsum('mhr,hdr->mhd', out.v_q_routed_raw, model.vq_proj_routed)  # [M, Hr, D], un-gated
    recon_shared = model.decoder_shared(z_shared)  # [M, Hs, C*L] — fused per-patch decode covers all channels jointly
    recon_routed = model.decoder_routed(z_routed)  # [M, Hr, C*L]
    recon_all = torch.cat([recon_shared, recon_routed], dim=1)  # [M, H_total, C*L]
    H = recon_all.shape[1]
    recon_all = recon_all.reshape(B, N, H, C, L)[0].permute(2, 0, 1, 3)   # [C, N, H, L] (single trial)

    n_fft = L
    if fs and freq_resolution:
        n_fft = max(L, int(round(fs / freq_resolution)))

    fft_c = torch.fft.rfft(recon_all.float(), n=n_fft, dim=-1)
    psd   = fft_c.real.pow(2) + fft_c.imag.pow(2)                      # [C, N, H, F]
    psd   = psd.mean(dim=1)                                            # [C, H, F] — average over patches
    psd   = psd.permute(1, 0, 2).cpu().numpy()                         # [H, C, F]

    freqs = np.fft.rfftfreq(n_fft, d=(1.0 / fs) if fs else 1.0)

    routing_shared = torch.full((model.n_shared_experts,), model.shared_weight, device=recon_routed.device)
    routing_routed = (out.gate_mask_routed.detach() > 0).float().mean(dim=0)
    routing_score  = torch.cat([routing_shared, routing_routed]).cpu().numpy()  # [H_total]

    return psd, freqs, routing_score


@torch.no_grad()
def extract_filter_psd(model, x: torch.Tensor, coords: torch.Tensor,
                       time_idx: torch.Tensor = None, valid_channels: torch.Tensor = None):
    """
    MeSAEPretrain analog of extract_head_psd: per-filter per-channel decoded activation
    norm. No Router/MoE here, so there's no gate score to rank by — filter_importance is
    the mean magnitude of each filter's own decoded contribution to the reconstruction
    instead (how much of the final signal that filter is actually responsible for).

    Returns ([psd_ch_q], filter_norms, filter_affinity, filter_importance):
      psd_ch_q         — [C, Q] numpy array, mean per-channel decoded activation norm per filter
      filter_norms      — [Q] numpy array, norm of the SAE-decoded vector the decoder reads
      filter_affinity   — [Q, Q] cosine similarity between filters' mean SAE-decoded vectors
      filter_importance — [Q] numpy array, mean decoded-contribution magnitude per filter
    """
    B, C, N, L = x.shape
    z = model.stage_features(x, coords, time_idx=time_idx)
    z_bnc, valid_mask = model._pool_channels(z, valid_channels)
    pooled, _ = model.filter_pool(z_bnc, valid_mask)
    sae_out, _, _ = model.sae(pooled)  # [M, Q, D] — the space the decoder actually reads
    Q = sae_out.shape[1]

    filter_norms = sae_out.norm(dim=-1).mean(dim=0).cpu().numpy()  # [Q]
    v_mean_n     = F.normalize(sae_out.mean(dim=0), dim=-1)
    filter_affinity = (v_mean_n @ v_mean_n.T).cpu().numpy()

    recon_per_filter = model.decoder(sae_out)  # [M, Q, C*patch_len]
    patch_len = recon_per_filter.shape[-1] // C
    recon_per_filter = recon_per_filter.reshape(B, N, Q, C, patch_len)
    psd_ch_q = recon_per_filter.norm(dim=-1).mean(dim=1)[0].permute(1, 0).cpu().numpy()  # [C, Q]

    filter_importance = recon_per_filter.norm(dim=-1).mean(dim=(0, 1, 3)).cpu().numpy()  # [Q]

    return [psd_ch_q], filter_norms, filter_affinity, filter_importance


@torch.no_grad()
def extract_filter_spectra(model, x: torch.Tensor, coords: torch.Tensor,
                           time_idx: torch.Tensor = None, valid_channels: torch.Tensor = None,
                           fs: float = None, freq_resolution: float = None):
    """
    MeSAEPretrain analog of extract_head_spectra: per-filter, per-channel power spectrum
    of that filter's own decoded contribution. No gating to worry about — every filter is
    always active — so this is just each filter's own contribution, un-gated by construction.
    Returns (psd [Q, C, F] numpy, freqs [F] numpy, filter_importance [Q] numpy).
    """
    B, C, N, L = x.shape
    z = model.stage_features(x, coords, time_idx=time_idx)
    z_bnc, valid_mask = model._pool_channels(z, valid_channels)
    pooled, _ = model.filter_pool(z_bnc, valid_mask)
    sae_out, _, _ = model.sae(pooled)  # [M, Q, D]
    Q = sae_out.shape[1]

    recon_per_filter = model.decoder(sae_out)  # [M, Q, C*L]
    recon_per_filter = recon_per_filter.reshape(B, N, Q, C, L)[0].permute(2, 0, 1, 3)  # [C, N, Q, L]

    n_fft = L
    if fs and freq_resolution:
        n_fft = max(L, int(round(fs / freq_resolution)))

    fft_c = torch.fft.rfft(recon_per_filter.float(), n=n_fft, dim=-1)
    psd   = fft_c.real.pow(2) + fft_c.imag.pow(2)          # [C, N, Q, F]
    psd   = psd.mean(dim=1)                                 # [C, Q, F] — average over patches
    psd   = psd.permute(1, 0, 2).cpu().numpy()              # [Q, C, F]

    freqs = np.fft.rfftfreq(n_fft, d=(1.0 / fs) if fs else 1.0)

    filter_importance = recon_per_filter.norm(dim=-1).mean(dim=(0, 1)).cpu().numpy()  # [Q]

    return psd, freqs, filter_importance


# ── Shared epoch-snapshot panels (topo_psd_filter, attn_topo) ──────────────────
#
# One canonical pair of figures reused by check_epoch_pretrain.py (MeFSQ Experts or
# MeSAE Filters — extract_head_psd/spectra and extract_filter_psd/spectra already return
# matching shapes: ([psd_ch_x], norms, affinity, importance) and (psd, freqs, importance))
# and check_epoch_finetune.py (same backbone, plus the finetune head's own attn_h). Keeps
# the two training-phase scripts producing the exact same panel format instead of drifting.

def plot_topo_psd_filter(out_path, pos2d, raw_power, recon_power, psd_raw, psd_recon,
                          psd_ch_x, psd_x, freqs, importance, cmap='YlOrRd',
                          subject_id=None, trial_idx=None, epoch_tag='', unit_label='Filter',
                          l_freq=None, h_freq=None):
    """
    Raw / Full-recon / per-unit (Expert or Filter) topo + PSD, side by side, one row each,
    sorted by `importance` below the first two fixed rows (Raw, Full Recon). Raw/recon PSD
    must already be computed with the same n_fft/freq axis as psd_x so every row is directly
    comparable, not just visually similar (see check_epoch_pretrain.py for the FFT call).

    psd_ch_x: [C, Q] per-unit per-channel decoded activation. psd_x: [Q, C, F] per-unit PSD.
    freqs/psd_x/psd_raw/psd_recon are expected already band-cropped by the caller if desired
    (l_freq/h_freq here are only used for the x-axis label/ticks, not re-cropping).
    """
    Q = psd_ch_x.shape[1]
    sorted_ord = np.argsort(importance)[::-1]
    freq_label = 'Hz' if freqs is not None and len(freqs) else 'cyc/patch'
    freq_ticks = np.linspace(freqs[0], freqs[-1], 5)

    entries = [('Raw', raw_power, psd_raw), ('Full Recon', recon_power, psd_recon)]
    entries += [(f'{unit_label[0]}{q} ({importance[q]:.3f})', psd_ch_x[:, q], psd_x[q]) for q in sorted_ord]
    n_items = len(entries)
    n_rows = math.ceil(n_items / 2)

    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 3.0 * n_rows), squeeze=False, constrained_layout=True)
    fig.suptitle(f"Raw / Recon / {unit_label} Topo + PSD ({unit_label}s sorted by contribution) — "
                 f"Sub {subject_id}, Trial {trial_idx}{epoch_tag}", fontsize=13, fontweight='bold')

    def _psd_row(ax_topo, ax_psd, power, psd_cf, label):
        im_t = draw_topomap(ax_topo, pos2d, power, cmap=cmap, vmin=power.min(), vmax=power.max())
        ax_topo.set_title(f'{label} Topo (power)', fontsize=9, fontweight='bold')
        fig.colorbar(im_t, ax=ax_topo, fraction=0.05, pad=0.02)

        ax_psd.imshow(psd_cf, aspect='auto', cmap=cmap, origin='lower',
                      extent=[freqs[0], freqs[-1], 0, psd_cf.shape[0]])
        ax_psd.set_title(f'{label} PSD (channel x freq)', fontsize=9, fontweight='bold')
        ax_psd.set_xticks(freq_ticks)
        ax_psd.set_xticklabels([f'{f:.0f}' for f in freq_ticks], fontsize=6)
        ax_psd.set_xlabel(freq_label, fontsize=7)
        ax_psd.set_yticks([])

    for i, (label, power, psd_cf) in enumerate(entries):
        row, col_pair = divmod(i, 2)
        _psd_row(axes[row, col_pair * 2], axes[row, col_pair * 2 + 1], power, psd_cf, label)
    for i in range(n_items, n_rows * 2):
        row, col_pair = divmod(i, 2)
        axes[row, col_pair * 2].axis('off')
        axes[row, col_pair * 2 + 1].axis('off')

    fig.text(0.5, 0.005, 'PSD y-axis: Channel (Fp1 -> Iz)', ha='center', fontsize=8)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_attn_topo(out_path, pos2d, attn, importance, channel_names, valid_channels=None,
                    subject_id=None, trial_idx=None, epoch_tag='', unit_label='Filter'):
    """
    Per-unit (Expert/Filter/Head) channel-attention topography, one topomap per unit
    (sorted by `importance`), plus one large Channel x Unit heatmap spanning all rows.
    attn: [Q, C] — each unit's own attention weight over channels (rows sum to 1).
    valid_channels: [C] bool or None — padded/invalid channels hidden from the topomaps only.
    """
    Q, C = attn.shape
    sorted_ord = np.argsort(importance)[::-1]
    attn_masked = attn if valid_channels is None else np.where(valid_channels[None, :], attn, np.nan)

    n_topo_rows = math.ceil(Q / 2)
    fig = plt.figure(figsize=(40.0, 3.0 * n_topo_rows), constrained_layout=True)
    gs = fig.add_gridspec(n_topo_rows, 3, width_ratios=[1.0, 1.0, 3.5], wspace=0.3)

    for i, q_orig in enumerate(sorted_ord):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        valid = ~np.isnan(attn_masked[q_orig])
        im = draw_topomap(ax, pos2d[valid], attn_masked[q_orig][valid], cmap='viridis')
        ax.set_title(f'{unit_label} {q_orig}  ({importance[q_orig]:.3f})', fontsize=9, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    if Q % 2:
        fig.add_subplot(gs[n_topo_rows - 1, 1]).axis('off')

    ax_big = fig.add_subplot(gs[:, 2])
    attn_ordered = attn[sorted_ord].T  # [C, Q], columns ordered to match the topomap panels
    im_big = ax_big.imshow(attn_ordered, aspect='auto', cmap='viridis')
    ax_big.set_yticks(range(C))
    ax_big.set_yticklabels(channel_names, fontsize=5)
    ax_big.set_xticks(range(Q))
    ax_big.set_xticklabels([f'{unit_label[0]}{q}\n{importance[q]:.3f}' for q in sorted_ord], fontsize=6, rotation=90)
    ax_big.set_xlabel(f'{unit_label} (sorted by contribution)', fontsize=9)
    ax_big.set_title(f'Channel x {unit_label} Attention', fontsize=10, fontweight='bold')
    fig.colorbar(im_big, ax=ax_big, fraction=0.03, pad=0.02).set_label('Attention weight', fontsize=8)

    fig.suptitle(f"Per-{unit_label} Channel Attention ({unit_label}s sorted by contribution) — "
                 f"Sub {subject_id}, Trial {trial_idx}{epoch_tag}", fontsize=12, fontweight='bold')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── Heatmap ───────────────────────────────────────────────────────────────────

def _imshow_row(ax, data2d, cmap, vmin, vmax, ylabel, ch_labels=None, show_x=False, freqs=None):
    """imshow a [C, F] array as a heatmap row."""
    im = ax.imshow(data2d, aspect='auto', origin='lower', cmap=cmap,
                   vmin=vmin, vmax=vmax,
                   extent=[freqs[0], freqs[-1], -0.5, data2d.shape[0] - 0.5]
                   if freqs is not None else None)
    ax.set_ylabel(ylabel, fontsize=8)
    if ch_labels is not None:
        ax.set_yticks(range(len(ch_labels)))
        ax.set_yticklabels(ch_labels, fontsize=6)
    else:
        ax.set_yticks([])
    if show_x and freqs is not None:
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
    else:
        ax.set_xticks([])
    return im


def _plot_6row_heatmap(rows, titles, cmaps, vmins, vmaxs, freqs_act, ch_labels,
                       suptitle, out_path):
    """Generic N-row heatmap figure with per-row colorbars."""
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.4 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = [axes]
    for i, (ax, data, title, cmap, vmin, vmax) in enumerate(
            zip(axes, rows, titles, cmaps, vmins, vmaxs)):
        im = _imshow_row(ax, data, cmap, vmin, vmax, title,
                         ch_labels=(ch_labels if i == 0 else None),
                         show_x=(i == n_rows - 1), freqs=freqs_act)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── Band filter ───────────────────────────────────────────────────────────────

def _band_filter(orig, recon, freqs, fs=200.0):
    """Band-filter orig and recon using MNE IIR filter. Returns (orig_filtered, recon_filtered)."""
    if freqs is None:
        return orig, recon
    try:
        import mne
        l_f, h_f = freqs
        yo = mne.filter.filter_data(orig.reshape(1, -1).astype(np.float64),
                                    fs, l_f, h_f, method='iir', verbose=False)[0]
        yr = mne.filter.filter_data(recon.reshape(1, -1).astype(np.float64),
                                    fs, l_f, h_f, method='iir', verbose=False)[0]
        return yo, yr
    except Exception:
        return np.zeros_like(orig), np.zeros_like(recon)
