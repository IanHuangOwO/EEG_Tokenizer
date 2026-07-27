"""
Pretrain-stage epoch snapshot — covers both MeFSQ and MeSAE (dispatched on model_type,
inferred from the model instance) with one shared panel format:
  recon_signal     raw vs recon time series, band-filtered, masked patches shaded
                    (viz/train.py:visualize_reconstruction)
  topo_psd_filter  raw / full-recon / per-unit topo + PSD grid, sorted by contribution
  attn_topo        per-unit channel-attention topography + Channel x Unit heatmap

"Unit" = MeFSQ's Expert (routed+shared pools, combined) or MeSAE's Filter. Both models
expose the same shapes for this (out.attn; extract_head_psd/spectra vs
extract_filter_psd/spectra return matching ([psd_ch], ..., importance) / (psd, freqs,
importance) tuples), so one script covers both instead of drifting into two formats —
see viz/draw.py:plot_topo_psd_filter / plot_attn_topo for the shared plotting code.
"""

import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch

from viz import (
    load_config,
    load_model,
    resolve_output_dir,
    select_subject_dataset,
    filter_config_to_subject,
    pick_trial,
)
from viz.draw import (
    project_coords_2d,
    extract_head_psd, extract_head_spectra,
    extract_filter_psd, extract_filter_spectra,
    plot_topo_psd_filter, plot_attn_topo,
)
from viz.compute import _run_reconstruction, _run_reconstruction_sae
from viz.train import visualize_reconstruction


# ── Module interface ──────────────────────────────────────────────────────────

def add_args(parser):
    parser.add_argument('--recon_cmap', type=str, default='YlOrRd')


@torch.no_grad()
def run(config, output_dir, args, model=None, dataset=None,
        trial_idx=None, subject_id=None, epoch=None):

    model_type = 'MeFSQ' if hasattr(model, 'n_routed_experts') else 'MeSAE'
    unit_label = 'Expert' if model_type == 'MeFSQ' else 'Filter'

    viz_dir = os.path.join(output_dir, 'recon')
    os.makedirs(viz_dir, exist_ok=True)
    epoch_tag = f'_ep{epoch:04d}' if epoch is not None else ''
    cmap = getattr(args, 'recon_cmap', None) or config.get('check', {}).get('reconstruction', {}).get('cmap', 'YlOrRd')

    device = next(model.parameters()).device
    model.eval()

    x_patches, coords, mask, time_indices, _, _, valid_channels = dataset[trial_idx]
    x_in  = x_patches.unsqueeze(0).to(device)
    c_in  = coords.unsqueeze(0).to(device)
    t_in  = time_indices.unsqueeze(0).to(device)
    vc_in = valid_channels.unsqueeze(0).to(device)

    data  = _run_reconstruction(model, dataset, trial_idx, device) if model_type == 'MeFSQ' \
        else _run_reconstruction_sae(model, dataset, trial_idx, device)
    pos2d = project_coords_2d(coords.numpy())

    # ── recon_signal ─────────────────────────────────────────────────────────
    raw_t   = torch.from_numpy(data['raw']).unsqueeze(0)
    recon_t = torch.from_numpy(data['recon']).unsqueeze(0)
    C, N, _ = x_patches.shape
    mask_np   = mask.numpy().reshape(C, N)  # [C*N] -> [C, N] — dataset's generated mask,
    # shown for reference regardless of training stage; the forward calls below still run
    # unmasked (bool_masked_pos not passed) for a clean reconstruction snapshot.
    patch_len = x_patches.shape[-1]
    visualize_reconstruction(
        None, (raw_t, recon_t), epoch,
        output_dir=viz_dir,
        channel_names=dataset.base_dataset.channel_names,
        subject_id=subject_id, trial_idx=trial_idx,
        mask=mask_np, patch_len=patch_len,
    )

    # ── topo_psd_filter ──────────────────────────────────────────────────────
    try:
        if model_type == 'MeFSQ':
            psd_list, _, _, importance = extract_head_psd(model, x_in, c_in, t_in)
        else:
            psd_list, _, _, importance = extract_filter_psd(model, x_in, c_in, t_in, vc_in)
        psd_ch_x = psd_list[0]  # [C, Q]

        pp = config.get('preprocess_params', {})
        fs = pp.get('target_freq')
        l_freq, h_freq = pp.get('l_freq'), pp.get('h_freq')
        if model_type == 'MeFSQ':
            psd_x, freqs, _ = extract_head_spectra(model, x_in, c_in, t_in, fs=fs, freq_resolution=0.2)
        else:
            psd_x, freqs, _ = extract_filter_spectra(model, x_in, c_in, t_in, vc_in, fs=fs, freq_resolution=0.2)

        n_fft = max(patch_len, int(round(fs / 0.2))) if fs else patch_len
        x_cnl     = x_patches.numpy()                       # [C, N, L]
        recon_cnl = data['recon'].reshape(C, N, patch_len)   # [C, N, L]
        fft_raw   = np.fft.rfft(x_cnl,     n=n_fft, axis=-1)
        fft_recon = np.fft.rfft(recon_cnl, n=n_fft, axis=-1)
        psd_raw   = (fft_raw.real**2   + fft_raw.imag**2  ).mean(axis=1)  # [C, F]
        psd_recon = (fft_recon.real**2 + fft_recon.imag**2).mean(axis=1)  # [C, F]

        if l_freq is not None and h_freq is not None:
            band = (freqs >= l_freq) & (freqs <= h_freq)
            freqs     = freqs[band]
            psd_x     = psd_x[:, :, band]
            psd_raw   = psd_raw[:, band]
            psd_recon = psd_recon[:, band]

        raw_power   = (data['raw']   ** 2).mean(axis=-1)  # [C]
        recon_power = (data['recon'] ** 2).mean(axis=-1)  # [C]

        out_path = os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_topo_psd_filter.png")
        plot_topo_psd_filter(
            out_path, pos2d, raw_power, recon_power, psd_raw, psd_recon,
            psd_ch_x, psd_x, freqs, importance, cmap=cmap,
            subject_id=subject_id, trial_idx=trial_idx, epoch_tag=epoch_tag,
            unit_label=unit_label, l_freq=l_freq, h_freq=h_freq,
        )
        print(f"  [epoch] -> {out_path}")
    except Exception as e:
        print(f"  [epoch] topo_psd_filter failed: {e}")

    # ── attn_topo ────────────────────────────────────────────────────────────
    try:
        if model_type == 'MeFSQ':
            _, _, _, importance = extract_head_psd(model, x_in, c_in, t_in)
            out = model(x_in, c_in, time_idx=t_in, bool_masked_pos=None, valid_channels=vc_in)
        else:
            _, _, _, importance = extract_filter_psd(model, x_in, c_in, t_in, vc_in)
            out = model(x_in, c_in, time_idx=t_in, valid_channels=vc_in)

        # attn: [B, N, Q, C] -> average over patches for a stable "typical topography per
        # unit" summary (a training-time health check, not a per-patch event locator).
        attn = out.attn[0].mean(dim=0).cpu().numpy()  # [Q, C]
        out_path = os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_attn_topo.png")
        plot_attn_topo(
            out_path, pos2d, attn, importance, dataset.base_dataset.channel_names,
            valid_channels=valid_channels.numpy(),
            subject_id=subject_id, trial_idx=trial_idx, epoch_tag=epoch_tag, unit_label=unit_label,
        )
        print(f"  [epoch] -> {out_path}")
    except Exception as e:
        print(f"  [epoch] attn_topo failed: {e}")


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    from IO.dataset import build_dataset_from_config

    parser = argparse.ArgumentParser(description='EEG Pretrain Epoch Snapshot (MeFSQ or MeSAE)')
    parser.add_argument('--config',     default='config/analysis.json')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--subject',    type=int, default=None)
    parser.add_argument('--trial',      type=int, default=None)
    parser.add_argument('--dataset',    type=str, default=None)
    add_args(parser)
    args = parser.parse_args()

    cfg        = load_config(args.config)
    checkpoint = args.checkpoint or cfg.get('checkpoint', '')
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl        = load_model(cfg, checkpoint, device)

    # --subject overrides to a single ad-hoc target; otherwise run every target configured
    # in training_params.visualize.targets (same list train_pretrain.py's periodic snapshot
    # loop reads — see docs on pick_trial's dataset_name disambiguation).
    if args.subject is not None:
        targets = [{'dataset': args.dataset, 'subject': args.subject, 'trial': args.trial}]
    else:
        targets = cfg.get('training_params', {}).get('visualize', {}).get('targets') or [{}]

    for t in targets:
        ds_name, subject = select_subject_dataset(cfg, t.get('subject'), dataset_name=t.get('dataset'))
        filtered = filter_config_to_subject(cfg, ds_name, subject)
        ds       = build_dataset_from_config(filtered, mode='pretrain')
        trial_cfg = t.get('trial') if t.get('trial') is not None else cfg['dataset_params']['pretrain'][ds_name].get('trial_to_use')
        t_idx, subject_id = pick_trial(ds, subject, trial_cfg, dataset_name=ds_name)
        out = resolve_output_dir(filtered, 'check')
        run(filtered, out, args, model=mdl, dataset=ds, trial_idx=t_idx, subject_id=subject_id)
        print(f"[check] done: dataset={ds_name} subject={subject_id} trial_idx={t_idx}")
