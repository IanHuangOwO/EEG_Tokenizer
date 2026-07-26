"""
Head grouping analysis: spatio-spectral similarity between MeFSQ heads and ICA components.

Fingerprint: frequency-band spatial map [C, n_bands]
  For each band, compute per-channel power of the component's signal.
  Two components are similar iff they activate the SAME channels at the SAME frequencies.

Outputs (output/<model_name>/visualization/head_analysis/):
  1_head_sim.png       H×H cosine sim on [C*n_bands] fingerprint + dendrogram
  2_ica_sim.png        K×K cosine sim on [C*n_bands] fingerprint + dendrogram
  3_head_topos.png     Per-head broadband power topomap grid
  4_ica_topos.png      Per-ICA broadband power topomap grid
  5_head_band_topos/   One file per band: all heads' spatial power map
  6_ica_band_topos/    One file per band: all ICA components' spatial power map
"""

import os
import sys
import copy
import json
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.signal import butter, sosfilt
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz import load_config, load_model, resolve_output_dir
from viz.draw import project_coords_2d, draw_topomap
from IO.dataset import build_dataset_from_config


BANDS = [
    ('delta', 1.0,  4.0),
    ('theta', 4.0,  8.0),
    ('alpha', 8.0,  13.0),
    ('beta',  13.0, 30.0),
    ('gamma', 30.0, None),   # None -> fs/2 - 1
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _band_power(sig2d, fs, l_freq, h_freq):
    """
    sig2d: [C, T]  (or any 2D where last axis is time)
    Returns per-channel band power [C].
    """
    nyq   = fs / 2.0
    h     = min(h_freq, nyq - 1.0) if h_freq is not None else nyq - 1.0
    sos   = butter(4, [l_freq / nyq, h / nyq], btype='bandpass', output='sos')
    filt  = sosfilt(sos, sig2d.astype(np.float64), axis=-1)
    return filt.var(axis=-1).astype(np.float32)                    # [C]


def band_spatial_fp(sig_ct, fs):
    """
    sig_ct: [C, T]
    Returns [C, n_bands] band-power fingerprint.
    """
    n_bands = len(BANDS)
    C       = sig_ct.shape[0]
    fp      = np.zeros((C, n_bands), dtype=np.float32)
    for b, (_, l, h) in enumerate(BANDS):
        fp[:, b] = _band_power(sig_ct, fs, l, h)
    return fp


def cosine_sim(X):
    """X: [N, D] -> [N, N] cosine similarity"""
    n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return n @ n.T


def cluster_order(sim):
    dist = np.clip(1.0 - sim, 0, 2)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    Z    = linkage(dist[np.triu_indices(len(dist), k=1)], method='average')
    return leaves_list(Z), Z


# ── Data ──────────────────────────────────────────────────────────────────────

def _unpack(batch, device):
    x_patches, coords, mask, time_indices, _, _ = batch
    x        = x_patches.to(device)
    coords   = coords.to(device)
    time_idx = time_indices.to(device)
    B, C, N, L = x.shape
    return x, coords, time_idx, mask.view(B, C, N).to(device)


def build_val_loader(config, batch_size):
    random.seed(42)
    val_cfg = copy.deepcopy(config)
    for ds_name, ds_args in config['dataset_params']['pretrain'].items():
        meta_path = os.path.join(ds_args['dataset_path'], 'metadata.json')
        with open(meta_path) as f:
            meta = json.load(f)
        all_subs = list(meta.get('data_structure', {}).keys())
        try:
            all_subs = sorted([int(s) for s in all_subs])
        except Exception:
            all_subs = sorted(all_subs)
        random.shuffle(all_subs)
        split = max(1, int(len(all_subs) * config['training_params']['pretrain'].get('train_val_split', 0.9)))
        val_cfg['dataset_params']['pretrain'][ds_name]['subject_to_use'] = all_subs[split:]
    dataset = build_dataset_from_config(val_cfg, transform=None, mode='pretrain')
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)


@torch.no_grad()
def accumulate(model, loader, device, n_batches):
    """Returns vq_all [B,N,H,r] (one shared code per patch, all channels), x_all [B,C,T],
    coords [C,3], head_sel_rate [H]"""
    model.eval()
    vq_list, x_list, gate_list, coords_out = [], [], [], None
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        x, coords, time_idx, _ = _unpack(batch, device)
        B, C, N, L = x.shape
        out = model(x, coords, time_idx, bool_masked_pos=None)
        v_q, gate_mask = out.v_q_routed, out.gate_mask_routed  # routed pool only — shared pool doesn't route
        vq_list.append(v_q.float().reshape(B, N, v_q.shape[1], v_q.shape[2]).cpu())
        x_list.append(x.float().reshape(B, C, N * L).cpu())
        # gate_mask [B*N, H] — softmax weight (nonzero) for selected experts
        gate_list.append((gate_mask.detach() > 0).float().mean(dim=0).cpu())  # [H]
        if coords_out is None:
            coords_out = coords[0].cpu().numpy()
    head_sel_rate = torch.stack(gate_list, 0).mean(0).numpy()  # [H] mean selection rate
    return torch.cat(vq_list, 0), torch.cat(x_list, 0), coords_out, head_sel_rate


# ── Fingerprints ──────────────────────────────────────────────────────────────

def head_cluster_assignments(model, vq_all, n_clusters=8):
    """
    Compute rfft mean+var fingerprints per head, run sklearn KMeans for post-hoc grouping.
    Each head's code decodes jointly to all C channels (fused per-patch VQ) — fingerprints
    are computed per channel from that joint decode, same as before the fusion change.
    Returns (assignments [H], fp [H, 2*C*F]).
    """
    B, N, H, r = vq_all.shape
    C = model.num_channels
    vq_proj   = model.vq_proj_routed.detach().cpu().float()
    decoder   = copy.deepcopy(model.decoder_routed).cpu().eval()   # per-head decoder, moved off-GPU for this analysis

    with torch.no_grad():
        z_all     = torch.einsum('bnhr,hdr->bnhd', vq_all, vq_proj)      # [B, N, H, embed_dim]
        recon_all = decoder(z_all)                                       # [B, N, H, C*patch_len] — decoder already per-head & vectorized
    patch_len = recon_all.shape[-1] // C
    recon_all = recon_all.reshape(B, N, H, C, patch_len)
    fft_all   = torch.fft.rfft(recon_all.float(), dim=-1).abs()          # [B, N, H, C, F]

    fp_mean = fft_all.mean(dim=[0, 1]).reshape(H, -1)                    # [H, C*F]
    fp_var  = fft_all.var(dim=[0, 1]).reshape(H, -1)                     # [H, C*F]
    fp = torch.cat([F.normalize(fp_mean, dim=-1),
                    F.normalize(fp_var,  dim=-1)], dim=-1).numpy()         # [H, 2*C*F]

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    assignments = km.fit_predict(fp)                                        # [H]
    return assignments, fp


def head_fingerprints(model, vq_all, fs):
    """
    For each head h:
      1. Reconstruct head h's joint all-channel contribution: recon_h [B, C, T]
      2. Reshape to [C, B*T] and compute band_spatial_fp -> [C, n_bands]
    Returns fps [H, C, n_bands]
    """
    B, N, H, r = vq_all.shape
    C = model.num_channels
    vq_proj = model.vq_proj_routed.detach().cpu().float()          # [H, embed_dim, r]
    decoder = copy.deepcopy(model.decoder_routed).cpu().eval()     # per-head decoder, moved off-GPU for this analysis

    with torch.no_grad():
        z_all     = torch.einsum('bnhr,hdr->bnhd', vq_all, vq_proj)   # [B, N, H, embed_dim]
        recon_all = decoder(z_all)                                    # [B, N, H, C*patch_len] — per-head, vectorized
    patch_len = recon_all.shape[-1] // C
    recon_all = recon_all.reshape(B, N, H, C, patch_len)

    fps = np.zeros((H, C, len(BANDS)), dtype=np.float32)
    for h in range(H):
        sig    = recon_all[:, :, h, :, :].permute(0, 2, 1, 3).reshape(B, C, N * patch_len).numpy()  # [B, C, T]
        sig_ct = sig.transpose(1, 0, 2).reshape(C, B * N * patch_len)         # [C, B*T]
        fps[h] = band_spatial_fp(sig_ct, fs)
    return fps                                                                     # [H, C, n_bands]


def ica_fingerprints(x_all, n_components, fs):
    """
    ICA on accumulated EEG, then per-component band_spatial_fp from channel-projected sources.
    Returns fps [K, C, n_bands], mixing [C, K], sources [K, T]
    """
    B, C, T = x_all.shape
    X = x_all.numpy().transpose(1, 0, 2).reshape(C, B * T).T    # [B*T, C]

    print(f"  FastICA [{B*T}, {C}] -> {n_components} components...")
    ica = FastICA(n_components=n_components, random_state=42, max_iter=500, tol=0.01)
    S   = ica.fit_transform(X).T                                  # [K, B*T]
    A   = ica.mixing_                                             # [C, K]

    fps = np.zeros((n_components, C, len(BANDS)), dtype=np.float32)
    for k in range(n_components):
        # Project source k back to channel space: [C, B*T]
        proj_k = A[:, k:k+1] * S[k:k+1, :]
        fps[k] = band_spatial_fp(proj_k, fs)

    return fps, A, S                                              # [K, C, n_bands]


# ── Plots ─────────────────────────────────────────────────────────────────────

def _sim_heatmap(sim, order, Z, title, out_path, labels):
    fig = plt.figure(figsize=(12, 10))
    gs  = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[1, 4],
                           hspace=0.02, wspace=0.02)
    ax_dt = fig.add_subplot(gs[0, 1])
    ax_dl = fig.add_subplot(gs[1, 0])
    ax_h  = fig.add_subplot(gs[1, 1])

    with plt.rc_context({'lines.linewidth': 0.5}):
        dendrogram(Z, ax=ax_dt, orientation='top',  no_labels=True, color_threshold=0)
        dendrogram(Z, ax=ax_dl, orientation='left', no_labels=True, color_threshold=0)
    ax_dt.axis('off'); ax_dl.axis('off')

    im = ax_h.imshow(sim[np.ix_(order, order)], cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ticks = np.arange(len(order))
    ax_h.set_xticks(ticks)
    ax_h.set_xticklabels([labels[i] for i in order], fontsize=4, rotation=90)
    ax_h.set_yticks(ticks)
    ax_h.set_yticklabels([labels[i] for i in order], fontsize=4)
    fig.colorbar(im, ax=ax_h, fraction=0.03, pad=0.02, label='Cosine similarity')
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {os.path.basename(out_path)}")


def _topo_grid(fps_cb, pos2d, titles, suptitle, out_path, sel_rate=None):
    """
    fps_cb: [N, C, n_bands] -> broadband (sum over bands) topo per component
    sel_rate: optional [N] float array, per-head routing selection frequency (0-1).
              Shown as subtitle; dead heads (sel_rate < 0.05) get grey background.
    """
    fps = fps_cb.sum(axis=-1)                                      # [N, C]
    N     = len(fps)
    ncols = min(10, N)
    nrows = int(np.ceil(N / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 3 * nrows), squeeze=False)
    for i in range(N):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        rate = sel_rate[i] if sel_rate is not None else 1.0
        dead = sel_rate is not None and rate < 0.05
        if dead:
            ax.set_facecolor('#e0e0e0')
        draw_topomap(ax, pos2d, fps[i], cmap='YlOrRd',
                     vmin=fps[i].min(), vmax=fps[i].max())
        label = titles[i]
        if sel_rate is not None:
            label += f'\n{rate:.0%}'
        color = '#888888' if dead else 'black'
        ax.set_title(label, fontsize=6, color=color)
    for i in range(N, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].axis('off')
    fig.suptitle(suptitle, fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {os.path.basename(out_path)}")


def _band_topo_dir(fps_cb, pos2d, titles, out_dir, prefix):
    """
    fps_cb: [N, C, n_bands]
    Saves one file per band: <out_dir>/<prefix>_<band>.png with all N topomaps.
    """
    os.makedirs(out_dir, exist_ok=True)
    N = len(fps_cb)
    for b, (band_name, l, h) in enumerate(BANDS):
        band_fps = fps_cb[:, :, b]                                 # [N, C]
        ncols = min(10, N)
        nrows = int(np.ceil(N / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 3 * nrows), squeeze=False)
        vmin, vmax = band_fps.min(), band_fps.max()
        for i in range(N):
            r, c = divmod(i, ncols)
            draw_topomap(axes[r][c], pos2d, band_fps[i], cmap='YlOrRd',
                         vmin=band_fps[i].min(), vmax=band_fps[i].max())
            axes[r][c].set_title(titles[i], fontsize=6)
        for i in range(N, nrows * ncols):
            r, c = divmod(i, ncols)
            axes[r][c].axis('off')
        h_label = f'{h:.0f}' if h else 'nyq'
        fig.suptitle(f'{band_name.capitalize()} ({l:.0f}–{h_label} Hz)', fontsize=11, fontweight='bold')
        fig.tight_layout()
        fname = os.path.join(out_dir, f'{prefix}_{band_name}.png')
        fig.savefig(fname, dpi=130, bbox_inches='tight')
        plt.close(fig)
    print(f"  saved {prefix} band topos -> {out_dir}")


def _tsne_plot(fp, assignments, K, out_path, perplexity=10):
    """
    fp: [H, D] rfft fingerprint numpy array
    assignments: [H] int cluster ids
    """
    H = len(fp)
    perplexity = min(perplexity, H - 1)
    tsne  = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    proj  = tsne.fit_transform(fp)                                        # [H, 2]

    cmap  = matplotlib.colormaps.get_cmap('tab20').resampled(K)
    colors = [cmap(int(a)) for a in assignments]

    fig, ax = plt.subplots(figsize=(9, 7))
    for h in range(H):
        ax.scatter(proj[h, 0], proj[h, 1], color=colors[h], s=80, zorder=3)
        ax.annotate(f'H{h}', (proj[h, 0], proj[h, 1]),
                    fontsize=5, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points')

    # Legend: one entry per cluster
    for k in range(K):
        ax.scatter([], [], color=cmap(k), label=f'G{k}', s=60)
    ax.legend(title='Cluster', fontsize='x-small', title_fontsize='x-small',
              loc='best', ncol=max(1, K // 8))

    ax.set_title(f't-SNE of Head Fingerprints  (H={H}, K={K})', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {os.path.basename(out_path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='config/config.json')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--n_batches',  type=int, default=20)
    parser.add_argument('--n_ica',      type=int, default=20)
    parser.add_argument('--n_clusters', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    fs     = float(config['preprocess_params'].get('target_freq', 200.0))

    checkpoint = args.checkpoint or os.path.join(
        'output', config['training_params']['pretrain']['model_name'], 'pretrain', 'best_pretrain.pth')
    print(f"Checkpoint: {checkpoint}")

    model   = load_model(config, checkpoint, device)
    out_dir = resolve_output_dir(config, 'head_analysis')

    # ── Accumulate ──────────────────────────────────────────────────────
    print(f"Accumulating {args.n_batches} val batches...")
    loader = build_val_loader(config, config['training_params']['pretrain']['batch_size'])
    vq_all, x_all, coords, head_sel_rate = accumulate(model, loader, device, args.n_batches)
    B, N, H, r = vq_all.shape
    print(f"  vq_all {list(vq_all.shape)}   x_all {list(x_all.shape)}")
    print(f"  head sel_rate min={head_sel_rate.min():.2f} max={head_sel_rate.max():.2f} mean={head_sel_rate.mean():.2f}")

    pos2d = project_coords_2d(coords)                             # [C, 2]

    # ── Fingerprints ─────────────────────────────────────────────────────
    print("Computing head band-spatial fingerprints...")
    h_fps = head_fingerprints(model, vq_all, fs)                  # [H, C, n_bands]

    print("Running ICA + band-spatial fingerprints...")
    i_fps, A_ica, S_ica = ica_fingerprints(x_all, args.n_ica, fs)  # [K, C, n_bands]

    # Flatten to [N, C*n_bands] for cosine sim
    h_flat = h_fps.reshape(H, -1)
    i_flat = i_fps.reshape(args.n_ica, -1)

    # ── Similarity + clustering ───────────────────────────────────────────
    sim_h = cosine_sim(h_flat); h_ord, Z_h = cluster_order(sim_h)
    sim_i = cosine_sim(i_flat); i_ord, Z_i = cluster_order(sim_i)

    print(f"Computing cluster assignments (K={args.n_clusters})...")
    cluster_assign, head_fp = head_cluster_assignments(model, vq_all, n_clusters=args.n_clusters)
    h_labels = [f'H{h}\nG{cluster_assign[h]}' for h in range(H)]
    print(f"  assignments: { {h: int(cluster_assign[h]) for h in range(H)} }")
    i_labels = [f'ICA{k}' for k in range(args.n_ica)]

    # ── Save figures ─────────────────────────────────────────────────────
    print("Saving figures...")

    _sim_heatmap(sim_h, h_ord, Z_h,
                 f'Head Spatio-Spectral Similarity — band-power spatial maps  (H={H})',
                 os.path.join(out_dir, '1_head_sim.png'), h_labels)

    _sim_heatmap(sim_i, i_ord, Z_i,
                 f'ICA Spatio-Spectral Similarity — band-power spatial maps  (K={args.n_ica})',
                 os.path.join(out_dir, '2_ica_sim.png'), i_labels)

    _topo_grid(h_fps, pos2d, h_labels,
               f'Head Broadband Spatial Power  (H={H})',
               os.path.join(out_dir, '3_head_topos.png'),
               sel_rate=head_sel_rate)

    _topo_grid(i_fps, pos2d, i_labels,
               f'ICA Broadband Spatial Power  (K={args.n_ica})',
               os.path.join(out_dir, '4_ica_topos.png'))

    _band_topo_dir(h_fps, pos2d, h_labels,
                   os.path.join(out_dir, '5_head_band_topos'), 'head')

    _band_topo_dir(i_fps, pos2d, i_labels,
                   os.path.join(out_dir, '6_ica_band_topos'), 'ica')

    _tsne_plot(head_fp, cluster_assign, args.n_clusters,
               os.path.join(out_dir, '7_head_tsne.png'))

    print(f"\nDone -> {out_dir}")


if __name__ == '__main__':
    main()
