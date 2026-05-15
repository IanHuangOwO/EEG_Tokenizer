"""
Reusable topomap drawing and PSD extraction helpers.
Model-agnostic functions stay here; analysis workflow lives in analysis/topomap.py.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle
import torch


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
        axes[row][col].set_title(f"#{pos + 1}  H{h_orig}\n{importance[h_orig]:.3f}",
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
        f"Sub {subject_id}, Trial {trial_idx}  (left→right, top→bottom: most→least important)",
        fontsize=11, fontweight='bold',
    )
    fig.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def extract_head_psd(model, x: torch.Tensor, coords: torch.Tensor,
                     time_idx: torch.Tensor) -> list:
    """
    Per-stage per-head per-channel reconstructed PSD power.
    Uses the model's actual forward pass (correct G filter + bias).
    Returns a list of [C, H] numpy arrays, one per decoder stage.
    """
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
        B_s, C_s, N_s, _ = z_stage.shape
        z_flat = z_stage.reshape(B_s * C_s, N_s, -1)

        vq_head  = model.vq_heads[i]
        decoder  = model.decoders[i]
        pooler   = model.poolers[i]

        v_q, _, _, _ = vq_head(z_flat)
        _, _, _, G   = pooler(v_q, B_s, C_s, decoder_W=decoder.W)   # G: [B, H, H]

        # Apply G filter (same as FastAdditiveDecoder.forward)
        g_w   = G.sum(dim=1)                                              # [B, H]
        g_w   = g_w.unsqueeze(1).expand(B_s, C_s, -1).reshape(B_s * C_s, -1)  # [B*C, H]
        v_q_g = v_q * g_w.unsqueeze(1).unsqueeze(-1)                     # [B*C, N, H, r]

        H, r  = vq_head.num_heads, vq_head.r
        D_dim = vq_head.A.shape[0]
        M     = torch.einsum('dhr,hdf->hrf', vq_head.A.view(D_dim, H, r), decoder.W)

        # Per-head contribution: sum over N patches, keep H separate → [B*C, H, 2F]
        h_out = torch.einsum('bnhr,hrf->bhf', v_q_g, M) / math.sqrt(H)

        F_dim = h_out.shape[-1] // 2
        psd   = (h_out[..., :F_dim].pow(2) + h_out[..., F_dim:].pow(2)).sum(dim=-1)  # [B*C, H]
        results.append(psd.reshape(B_s, C_s, H)[0].cpu().numpy())                     # [C, H]

    return results
