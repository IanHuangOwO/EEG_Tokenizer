"""
Raw + reconstructed data visualization: band temporal, amp/phase heatmaps,
real/imag heatmaps, reconstructed PSD, reconstructed topo, masking.
Requires model for reconstruction; masking uses the pretrain dataset.
"""

import os
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import welch

from IO.dataset import (
    build_dataset_from_config,
    MaskedPretrainDataset,
    BlockMaskingStrategy,
    RandomMaskingStrategy,
)
from analysis import (
    load_config,
    load_model,
    resolve_output_dir,
    select_subject_dataset,
    filter_config_to_subject,
    pick_trial,
)
def spectrum_to_time(real: torch.Tensor, imag: torch.Tensor,
                     mask: torch.Tensor, n_fft: int, n_samples: int) -> torch.Tensor:
    full_fft = torch.zeros((real.shape[0], n_fft // 2 + 1),
                           dtype=torch.complex64, device=real.device)
    full_fft[:, mask] = torch.complex(real.float(), imag.float())
    return torch.fft.irfft(full_fft, n=n_fft, dim=-1, norm='ortho')[:, :n_samples]


@torch.no_grad()
def get_detailed_outputs(model, x: torch.Tensor, coords: torch.Tensor,
                         time_idx: torch.Tensor):
    """
    Stage-wise reconstructions using the model's actual forward pass.
    Returns (stage_recons, [], stage_names).
    """
    B, C, N, L = x.shape
    T_total = N * L

    all_pred_real, all_pred_imag, *_ = model(x, coords=coords, time_idx=time_idx)

    stage_recons, stage_names = [], []
    for i, h_cfg in enumerate(model.decoder_heads_config):
        mask  = getattr(model, f'freq_mask_{i}')
        p_real = all_pred_real[i].reshape(B * C, -1)  # [B*C, F]
        p_imag = all_pred_imag[i].reshape(B * C, -1)
        recon  = spectrum_to_time(p_real, p_imag, mask, model.n_fft, T_total)
        stage_recons.append(recon.reshape(B, C, T_total))
        f_min, f_max = h_cfg["freq_range"]
        stage_names.append(f"Stage {i} ({f_min}-{f_max}Hz)")

    return stage_recons, [], stage_names
from utils.topomap_helpers import project_coords_2d, draw_topomap


# ── Pure helpers ──────────────────────────────────────────────────────────────

def _select_channels(total_ch: int, n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n >= total_ch:
        return np.arange(total_ch)
    return np.sort(rng.choice(total_ch, n, replace=False))


@torch.no_grad()
def _run_reconstruction(model, dataset, trial_idx: int, device):
    """
    Run model forward on one trial.
    Returns a dict with raw signal, reconstructed signal, per-band signals,
    and spectral components — all as numpy arrays.
    """
    x_patches, coords, time_indices, _, _ = dataset[trial_idx]
    C, N, L = x_patches.shape
    T_total  = N * L
    fs = dataset.base_dataset.config['model_params']['AttnVQ']['preprocess']['target_freq']

    x_in        = x_patches.unsqueeze(0).to(device)
    coords_in   = coords.unsqueeze(0).to(device)
    time_idx_in = time_indices.unsqueeze(0).to(device)

    stage_recons, _, stage_names = get_detailed_outputs(model, x_in, coords_in, time_idx_in)
    num_stages = len(stage_recons)

    raw    = x_patches.reshape(C, T_total).cpu().numpy()       # [C, T]
    recon  = torch.stack(stage_recons).sum(dim=0)[0].cpu().numpy()  # [C, T]

    raw_t   = torch.from_numpy(raw).to(device)
    recon_t = torch.from_numpy(recon).to(device)

    # Active frequency mask (union of all stage masks)
    full_mask = torch.zeros(model.n_fft // 2 + 1, dtype=torch.bool, device=device)
    for i in range(num_stages):
        full_mask |= getattr(model, f'freq_mask_{i}')
    full_mask_np = full_mask.cpu().numpy()
    freqs        = torch.fft.rfftfreq(model.n_fft, d=1.0 / fs).cpu().numpy()
    freqs_act    = freqs[full_mask_np]

    raw_fft_full   = torch.fft.rfft(raw_t,   n=model.n_fft, dim=-1, norm='ortho')  # [C, F]
    recon_fft_full = torch.fft.rfft(recon_t, n=model.n_fft, dim=-1, norm='ortho')  # [C, F]

    raw_fft_act   = raw_fft_full[:,   full_mask].cpu().numpy()  # [C, F_act] complex
    recon_fft_act = recon_fft_full[:, full_mask].cpu().numpy()  # [C, F_act] complex

    # Band-filtered raw + per-stage reconstructed
    bands_raw, bands_recon = [], []
    for i in range(num_stages):
        m = getattr(model, f'freq_mask_{i}')
        fft_i = torch.zeros_like(raw_fft_full)
        fft_i[:, m] = raw_fft_full[:, m]
        b_raw = torch.fft.irfft(fft_i, n=model.n_fft, dim=-1, norm='ortho')[:, :T_total]
        bands_raw.append(b_raw.cpu().numpy())                       # [C, T]
        bands_recon.append(stage_recons[i][0].cpu().numpy())        # [C, T]

    return {
        'raw':           raw,
        'recon':         recon,
        'raw_fft_act':   raw_fft_act,
        'recon_fft_act': recon_fft_act,
        'freqs_act':     freqs_act,
        'bands_raw':     bands_raw,
        'bands_recon':   bands_recon,
        'coords':        coords.numpy(),
        'stage_names':   stage_names,
        'T': T_total, 'N': N, 'L': L, 'fs': fs,
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _imshow_row(ax, data2d, cmap, vmin, vmax, ylabel, ch_labels=None, show_x=False, freqs=None):
    """Shared helper: imshow a [C, F] array as a heatmap row."""
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
    """Generic 6-row heatmap figure with shared colorbars per pair."""
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


def _plot_ampphase_comparison(data, sel_ch, ch_names, subject_id, trial_idx, viz_dir):
    raw_fft   = data['raw_fft_act'][sel_ch]    # [n, F]
    recon_fft = data['recon_fft_act'][sel_ch]
    freqs     = data['freqs_act']
    labels    = [ch_names[i] for i in sel_ch]

    raw_amp   = np.abs(raw_fft);   recon_amp   = np.abs(recon_fft)
    raw_db    = 20 * np.log10(raw_amp   + 1e-8)
    recon_db  = 20 * np.log10(recon_amp + 1e-8)
    amp_err   = (raw_amp - recon_amp) ** 2
    raw_ph    = np.angle(raw_fft);  recon_ph    = np.angle(recon_fft)
    ph_err    = np.abs(np.angle(np.exp(1j * (recon_ph - raw_ph))))

    shared_db = [min(raw_db.min(), recon_db.min()), max(raw_db.max(), recon_db.max())]
    rows   = [raw_db, recon_db, amp_err, raw_ph, recon_ph, ph_err]
    titles = ["Amp Raw (dB)", "Amp Recon (dB)", "Amp MSE",
              "Phase Raw", "Phase Recon", "Phase Error"]
    cmaps  = ['jet', 'jet', 'magma', 'twilight', 'twilight', 'magma']
    vmins  = [shared_db[0], shared_db[0], 0, -np.pi, -np.pi, 0]
    vmaxs  = [shared_db[1], shared_db[1], amp_err.max(), np.pi, np.pi, np.pi]

    _plot_6row_heatmap(rows, titles, cmaps, vmins, vmaxs, freqs, labels,
                       f"Amplitude & Phase — Sub {subject_id} Trial {trial_idx}",
                       os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}_ampphase.png"))
    print(f"  [data] AmpPhase saved")


def _plot_realimag_comparison(data, sel_ch, ch_names, subject_id, trial_idx, viz_dir):
    raw_fft   = data['raw_fft_act'][sel_ch]
    recon_fft = data['recon_fft_act'][sel_ch]
    freqs     = data['freqs_act']
    labels    = [ch_names[i] for i in sel_ch]

    raw_re    = raw_fft.real;   recon_re   = recon_fft.real
    raw_im    = raw_fft.imag;   recon_im   = recon_fft.imag
    re_err    = (raw_re - recon_re) ** 2
    im_err    = (raw_im - recon_im) ** 2

    shared_re = [min(raw_re.min(), recon_re.min()), max(raw_re.max(), recon_re.max())]
    shared_im = [min(raw_im.min(), recon_im.min()), max(raw_im.max(), recon_im.max())]
    rows   = [raw_re, recon_re, re_err, raw_im, recon_im, im_err]
    titles = ["Real Raw", "Real Recon", "Real MSE", "Imag Raw", "Imag Recon", "Imag MSE"]
    cmaps  = ['RdBu_r', 'RdBu_r', 'magma', 'RdBu_r', 'RdBu_r', 'magma']
    vmins  = [shared_re[0], shared_re[0], 0, shared_im[0], shared_im[0], 0]
    vmaxs  = [shared_re[1], shared_re[1], re_err.max(), shared_im[1], shared_im[1], im_err.max()]

    _plot_6row_heatmap(rows, titles, cmaps, vmins, vmaxs, freqs, labels,
                       f"Real & Imaginary — Sub {subject_id} Trial {trial_idx}",
                       os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}_realimag.png"))
    print(f"  [data] RealImag saved")


def _plot_band_temporal(data, sel_ch, ch_names, subject_id, trial_idx, viz_dir):
    """All stages in one figure: columns = stages, rows = channels."""
    stage_names = data['stage_names']
    num_stages  = len(stage_names)
    fs          = data['fs']
    T           = data['T']
    t_vec       = np.arange(T) / fs
    n_ch        = len(sel_ch)

    fig, axes = plt.subplots(n_ch, num_stages,
                             figsize=(10 * num_stages, 2.0 * n_ch),
                             constrained_layout=True)
    if n_ch == 1:
        axes = axes[np.newaxis, :]
    if num_stages == 1:
        axes = axes[:, np.newaxis]

    for s_idx, name in enumerate(stage_names):
        for row, ci in enumerate(sel_ch):
            ax = axes[row, s_idx]
            ax.plot(t_vec, data['bands_raw'][s_idx][ci],
                    color='#888888', alpha=0.6, linewidth=0.8, label='Raw')
            ax.plot(t_vec, data['bands_recon'][s_idx][ci],
                    color=f'C{s_idx}', alpha=0.9, linewidth=1.0, linestyle='--', label='Recon')
            ax.set_yticks([])
            ax.grid(True, alpha=0.08)
            if row == 0:
                ax.set_title(name, fontsize=9, fontweight='bold')
                ax.legend(fontsize=6, loc='upper right', ncol=2)
            if s_idx == 0:
                ax.set_ylabel(ch_names[ci], fontsize=7, rotation=0, labelpad=30, va='center')
            if row < n_ch - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Time (s)", fontsize=7)

    fig.suptitle(f"Band Temporal — Sub {subject_id} Trial {trial_idx}",
                 fontsize=12, fontweight='bold')
    out = os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}_band_temporal.png")
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [data] Band temporal saved")


def _plot_psd_reconstructed(data, sel_ch, ch_names, subject_id, trial_idx, viz_dir):
    """
    Welch PSD heatmap [n_ch × freq] for raw and reconstructed signals (dB/Hz),
    clipped to the model's max preprocessed frequency.
    """
    raw   = data['raw']    # [C, T]
    recon = data['recon']  # [C, T]
    fs    = data['fs']
    f_max = float(data['freqs_act'].max())
    nperseg = min(256, data['T'])
    labels  = [ch_names[i] for i in sel_ch]

    f_ref, _ = welch(raw[sel_ch[0]], fs=fs, nperseg=nperseg)
    mask_f   = f_ref <= f_max
    f_ref    = f_ref[mask_f]

    raw_psd   = np.stack([welch(raw[ci],   fs=fs, nperseg=nperseg)[1][mask_f] for ci in sel_ch])
    recon_psd = np.stack([welch(recon[ci], fs=fs, nperseg=nperseg)[1][mask_f] for ci in sel_ch])

    raw_db   = 10 * np.log10(raw_psd   + 1e-12)
    recon_db = 10 * np.log10(recon_psd + 1e-12)
    vmin = min(raw_db.min(), recon_db.min())
    vmax = max(raw_db.max(), recon_db.max())

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), constrained_layout=True)
    for ax, mat, title in zip(axes,
                               [raw_db, recon_db],
                               ["Raw PSD (dB/Hz)", "Reconstructed PSD (dB/Hz)"]):
        im = ax.imshow(mat, aspect='auto', origin='lower', cmap='RdBu_r',
                       vmin=vmin, vmax=vmax,
                       extent=[f_ref[0], f_ref[-1], -0.5, len(sel_ch) - 0.5])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_ylabel(title, fontsize=9)
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    fig.suptitle(f"PSD Heatmap — Sub {subject_id} Trial {trial_idx}", fontsize=12, fontweight='bold')
    fig.savefig(os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}_psd_recon.png"),
                dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [data] PSD (recon) saved")


def _plot_topo_reconstructed(data, config, subject_id, trial_idx, viz_dir):
    """Per-band + total power topomap: raw (top row) vs reconstructed (bottom row)."""
    coords      = data['coords']      # [C, 3]
    bands_raw   = data['bands_raw']   # list of [C, T]
    bands_recon = data['bands_recon'] # list of [C, T]
    stage_names = data['stage_names']
    num_stages  = len(bands_recon)
    num_cols    = num_stages + 1      # +1 for Total

    pos2d = project_coords_2d(coords)
    cmap  = config.get('check', {}).get('topomap', {}).get('cmap', 'YlOrRd')

    fig, axes = plt.subplots(2, num_cols,
                             figsize=(4 * num_cols, 8), constrained_layout=True)

    total_raw   = (data['raw']   ** 2).mean(axis=-1)
    total_recon = (data['recon'] ** 2).mean(axis=-1)

    # Per-column (per-band) power pairs for shared vmin/vmax (log10 scale)
    col_pairs = [
        ([bands_raw[s], bands_recon[s]], stage_names[s], False)
        for s in range(num_stages)
    ] + [([data['raw'], data['recon']], 'Total', True)]

    row_labels = ['Raw', 'Recon']
    for col, (signals, name, is_total) in enumerate(col_pairs):
        log_powers = [np.log10((s ** 2).mean(axis=-1) + 1e-12) for s in signals]
        vmin = min(p.min() for p in log_powers)
        vmax = max(p.max() for p in log_powers)
        for row, lp in enumerate(log_powers):
            im = draw_topomap(axes[row, col], pos2d, lp, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, col].set_title(
                f"{row_labels[row]}\n{name}", fontsize=9,
                fontweight='bold' if is_total else 'normal')
        cb = fig.colorbar(im, ax=axes[:, col].tolist(), fraction=0.046, pad=0.04)
        cb.set_label('log₁₀ Power', fontsize=7)

    fig.suptitle(f"Power Topomap — Sub {subject_id} Trial {trial_idx}",
                 fontsize=12, fontweight='bold')
    fig.savefig(os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}_topo_recon.png"),
                dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [data] Topo (recon) saved")


# ── Module interface ──────────────────────────────────────────────────────────

def add_args(parser):
    parser.add_argument('--num_channels', type=int, default=None,
                        help='Number of channels to randomly select for plots (default: 8)')


def run(config, output_dir, args, model=None, dataset=None, trial_idx=None, subject_id=None):
    check_cfg     = config.get('check', {}).get('data', {})
    num_channels  = (getattr(args, 'num_channels', None)
                     or check_cfg.get('num_channels', 8))
    plot_band     = check_cfg.get('plot_band_temporal', True)
    plot_ampphase = check_cfg.get('plot_ampphase',      True)
    plot_realimag = check_cfg.get('plot_realimag',      True)
    plot_psd      = check_cfg.get('plot_psd',           True)
    plot_topo     = check_cfg.get('plot_topo',          True)
    plot_masking  = check_cfg.get('plot_masking',       True)

    viz_dir = os.path.join(output_dir, 'data')
    os.makedirs(viz_dir, exist_ok=True)

    device   = next(model.parameters()).device
    ch_names = dataset.base_dataset.channel_names

    data    = _run_reconstruction(model, dataset, trial_idx, device)
    C       = data['raw'].shape[0]
    sel_ch  = _select_channels(C, num_channels)

    if plot_band:
        _plot_band_temporal(data, sel_ch, ch_names, subject_id, trial_idx, viz_dir)

    if plot_ampphase:
        _plot_ampphase_comparison(data, sel_ch, ch_names, subject_id, trial_idx, viz_dir)

    if plot_realimag:
        _plot_realimag_comparison(data, sel_ch, ch_names, subject_id, trial_idx, viz_dir)

    if plot_psd:
        _plot_psd_reconstructed(data, sel_ch, ch_names, subject_id, trial_idx, viz_dir)

    if plot_topo:
        _plot_topo_reconstructed(data, config, subject_id, trial_idx, viz_dir)

    if plot_masking:
        base_ds       = dataset.base_dataset
        pp            = config['model_params']['AttnVQ']['preprocess']
        patch_len     = pp.get('patch_length',    200)
        mask_ratio    = pp.get('mask_ratio',      0.5)
        strategy_name = pp.get('masking_strategy', 'random')
        strategy = (BlockMaskingStrategy(
                        row_prob=pp.get('mask_row_prob', 0.5),
                        col_prob=pp.get('mask_col_prob', 0.5))
                    if strategy_name == 'block'
                    else RandomMaskingStrategy())
        from utils.visualization import visualize_masking
        pretrain_ds = MaskedPretrainDataset(base_ds, patch_len=patch_len,
                                            mask_ratio=mask_ratio,
                                            masking_strategy=strategy)
        visualize_masking(pretrain_ds, subject_id, output_dir=viz_dir, trial_idx=trial_idx)

    print(f"  [data] → {viz_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='EEG Data Check with Reconstruction')
    parser.add_argument('--config',     default='config/analysis.json')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--subject',    type=int, default=None)
    parser.add_argument('--trial',      type=int, default=None)
    parser.add_argument('--dataset',    type=str, default=None)
    add_args(parser)
    args = parser.parse_args()

    cfg        = load_config(args.config)
    checkpoint = args.checkpoint or cfg.get('checkpoint', '')
    ds_name, subject = select_subject_dataset(cfg, args.subject, args.dataset)
    _ds_early  = args.dataset or next(
        (ds for ds, dc in cfg['dataset_params'].items() if 'trial_to_use' in dc),
        next(iter(cfg['dataset_params'])))
    trial_cfg  = args.trial if args.trial is not None else cfg['dataset_params'][_ds_early].get('trial_to_use')

    filtered = filter_config_to_subject(cfg, ds_name, subject)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl      = load_model(filtered, checkpoint, device)
    from IO.dataset import build_dataset_from_config
    ds       = build_dataset_from_config(filtered, mode='tokenizer')
    t_idx, subject_id = pick_trial(ds, subject, trial_cfg)
    out      = resolve_output_dir(filtered, 'check')
    run(filtered, out, args, model=mdl, dataset=ds, trial_idx=t_idx, subject_id=subject_id)
