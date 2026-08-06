"""Topomap drawing primitive: 3-D electrode coords -> 2-D projection -> interpolated head plot."""

import numpy as np
import matplotlib.tri as mtri
from matplotlib.patches import Circle


def project_coords_2d(coords: np.ndarray) -> np.ndarray:
    """Azimuthal equidistant projection of 3-D electrode coords to 2-D. coords: [C, 3]"""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-8
    xn, yn, zn = x / r, y / r, z / r
    theta = np.arccos(np.clip(zn, -1, 1))
    phi = np.arctan2(xn, yn)
    rho = theta / (np.pi / 2)
    return np.stack([rho * np.sin(phi), rho * np.cos(phi)], axis=1)


def build_triangulation(pos2d: np.ndarray):
    """Delaunay triangulation of electrode positions, reusable across every unit's
    draw_topomap call within one panel — pos2d is the same for all of them, so building
    this once instead of per-call avoids redoing an O(C log C) triangulation ~Q times
    (Q = number of Experts/Filters, e.g. 64) for a single figure. Returns None if
    triangulation fails, signaling draw_topomap to fall back to scipy griddata."""
    try:
        return mtri.Triangulation(pos2d[:, 0], pos2d[:, 1])
    except Exception:
        return None


def draw_topomap(ax, pos2d: np.ndarray, values: np.ndarray,
                 n_grid: int = 100, cmap: str = 'YlOrRd',
                 vmin=None, vmax=None, triang=None):
    """
    Interpolated topomap on a circular head outline.
    pos2d: [C, 2], values: [C]
    triang: optional pre-built build_triangulation(pos2d) result, reused across calls that
    share the same pos2d — pass None to build it fresh (also the fallback path when a
    passed-in triang was None because triangulation failed for this pos2d).
    Returns the pcolormesh artist (for colorbar attachment).
    """
    px, py = pos2d[:, 0], pos2d[:, 1]
    vmin = values.min() if vmin is None else vmin
    vmax = values.max() if vmax is None else vmax

    xi = np.linspace(-1.1, 1.1, n_grid)
    yi = np.linspace(-1.1, 1.1, n_grid)
    Xi, Yi = np.meshgrid(xi, yi)

    triang = triang if triang is not None else build_triangulation(pos2d)
    if triang is not None:
        Zi = mtri.LinearTriInterpolator(triang, values)(Xi, Yi)
    else:
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
