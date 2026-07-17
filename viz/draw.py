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
    Per-head per-channel VQ activation norm.
    Returns ([psd_ch_h], head_norms, head_affinity, routing_score):
      psd_ch_h      — [C, H] numpy array, mean ||v_q|| per channel per head
      head_norms    — [H] numpy array
      head_affinity — [H, H] cosine similarity
      routing_score — [H] fraction of patches that selected each head (1.0 when no routing)
    """
    B, C, N, L = x.shape
    _, _, v_q, gate_mask, _ = model(x, coords=coords, time_idx=time_idx, bool_masked_pos=None)
    # v_q: [B*C*N, H, r]
    H = v_q.shape[1]

    head_norms    = v_q.norm(dim=-1).mean(dim=0).cpu().numpy()  # [H]
    v_mean_h      = v_q.mean(dim=0)
    v_mean_n      = F.normalize(v_mean_h, dim=-1)
    head_affinity = (v_mean_n @ v_mean_n.T).cpu().numpy()
    psd_ch_h      = v_q.norm(dim=-1).reshape(B, C, N, H).mean(dim=2)[0].cpu().numpy()  # [C, H]

    # router importance: mean raw logit per head — more spread than sigmoid, no lb_loss flattening
    gate_logits = getattr(model, '_last_gate_logits', None)
    if gate_logits is not None:
        routing_score = gate_logits.mean(dim=0).cpu().numpy()              # [H] mean logit
    else:
        routing_score = (gate_mask.detach() > 0.5).float().mean(dim=0).cpu().numpy()  # fallback

    return [psd_ch_h], head_norms, head_affinity, routing_score


@torch.no_grad()
def extract_head_spectra(model, x: torch.Tensor, coords: torch.Tensor,
                          time_idx: torch.Tensor = None, fs: float = None,
                          freq_resolution: float = None):
    """
    Per-head, per-channel power spectrum of that head's OWN decoded reconstruction
    (v_q -> vq_proj -> decoder, per head, un-gated so every head shows what it would
    reconstruct if selected) — not the shared-embedding VQ activation norm extract_head_psd
    reports, an actual frequency-domain view of each head's specialization.
    Returns (psd [H, C, F] numpy, freqs [F] numpy, routing_score [H] numpy).
    fs: sample rate in Hz for the freq axis; if None, freqs are cycles/patch (bin index).
    freq_resolution: Hz per bin via zero-padded FFT (n_fft = fs / freq_resolution) — the
    patch itself (L samples) is far shorter than what a fine resolution needs, so this is
    padding for display resolution, not real added information beyond the L-sample window.
    """
    B, C, N, L = x.shape
    _, _, v_q, gate_mask, _ = model(x, coords=coords, time_idx=time_idx, bool_masked_pos=None)
    # v_q: [B*C*N, H, r], un-gated (every head's own code, not zeroed by routing)
    H = v_q.shape[1]

    z_all     = torch.einsum('mhr,hdr->mhd', v_q, model.vq_proj)       # [M, H, D]
    recon_all = model.decoder(z_all)                                   # [M, H, L]
    recon_all = recon_all.reshape(B, C, N, H, L)[0]                    # [C, N, H, L] (single trial)

    n_fft = L
    if fs and freq_resolution:
        n_fft = max(L, int(round(fs / freq_resolution)))

    fft_c = torch.fft.rfft(recon_all.float(), n=n_fft, dim=-1)
    psd   = fft_c.real.pow(2) + fft_c.imag.pow(2)                      # [C, N, H, F]
    psd   = psd.mean(dim=1)                                            # [C, H, F] — average over patches
    psd   = psd.permute(1, 0, 2).cpu().numpy()                         # [H, C, F]

    freqs = np.fft.rfftfreq(n_fft, d=(1.0 / fs) if fs else 1.0)

    gate_logits = getattr(model, '_last_gate_logits', None)
    if gate_logits is not None:
        routing_score = gate_logits.mean(dim=0).cpu().numpy()          # [H] mean logit
    else:
        routing_score = (gate_mask.detach() > 0.5).float().mean(dim=0).cpu().numpy()  # fallback

    return psd, freqs, routing_score


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
