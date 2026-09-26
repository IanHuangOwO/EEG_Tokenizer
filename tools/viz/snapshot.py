"""Per-trial snapshot figures from a SnapshotBundle (tools/analysis/snapshot.py): the band-filtered
original-vs-reconstruction grid, the stamp gallery, and the per-patch stamp grid. Called by
train_pretrain.py's periodic visualisation and by the snapshot / class_snapshots panels."""
import os

import numpy as np

from tools.viz.extract import extract_flat_stamp_gallery, extract_flat_stamp_psd_by_patch
from tools.viz.stamp_plots import plot_stamp_by_patch, plot_stamp_gallery
from tools.viz.timeseries import visualize_reconstruction
from tools.viz.topomap import project_coords_2d


def render_recon(bundle, config, out_dir):
    """Band-filtered orig-vs-recon grid for one trial."""
    viz_dir = out_dir
    os.makedirs(viz_dir, exist_ok=True)

    pp = config.get('preprocess_params', {})
    fs = pp.get('sample_freq')
    bandpass = pp.get('bandpass_filter', {})
    l_freq, h_freq = bandpass.get('l_freq'), bandpass.get('h_freq')
    viz_cfg = config.get('training_params', {}).get('visualize_params', {})
    band_edges = ({name: tuple(edges) for name, edges in viz_cfg['bands'].items()}
                  if viz_cfg.get('bands') else None)

    visualize_reconstruction(
        None, (bundle.raw_t, bundle.recon_t), bundle.epoch,
        output_dir=viz_dir,
        channel_names=bundle.channel_names,
        subject_id=bundle.subject_id, trial_idx=bundle.trial_idx,
        mask=bundle.mask_np, patch_len=bundle.patch_len,
        patch_stride=pp.get('patch_stride', bundle.patch_len),
        tag=bundle.filename_tag.lstrip('_') + ('_' if bundle.filename_tag else ''),
        fs=fs or 200.0, l_freq=l_freq, h_freq=h_freq, band_edges=band_edges,
        event_onset_sec=bundle.event_onset_sec,
        valid_start=bundle.valid_start, valid_end=bundle.valid_end,
    )


def render_stamp_gallery(bundle, config, out_dir, cmap='YlOrRd'):
    """Stamp gallery for one trial: whole-trial raw/recon PSD, then per stamp its topography, PSD,
    phase and waveform."""
    model = bundle.psd_model
    viz_dir = out_dir
    os.makedirs(viz_dir, exist_ok=True)

    pos2d = project_coords_2d(bundle.coords)
    epoch_tag = (f'_ep{bundle.epoch:04d}' if bundle.epoch is not None else '') + bundle.filename_tag
    tagged_epoch_tag = f'{epoch_tag}{bundle.title_suffix}'

    pp = config.get('preprocess_params', {})
    fs = pp.get('sample_freq')
    bandpass = pp.get('bandpass_filter', {})
    l_freq, h_freq = bandpass.get('l_freq'), bandpass.get('h_freq')
    viz_cfg = config.get('training_params', {}).get('visualize_params', {})
    fft_resolution = viz_cfg.get('fft_resolution', 0.2)
    psd_range = viz_cfg.get('psd_freq_range')
    psd_l_freq, psd_h_freq = tuple(psd_range) if psd_range else (l_freq, h_freq)

    # Whole-trial raw/recon PSD -- same FFT settings as the gallery's own per-stamp PSD
    # (freq_resolution=fft_resolution drives both), so the header row is directly
    # comparable to the stamp rows below it. n_fft must match extract_flat_stamp_
    # gallery's own n_fft (round(fs/fft_resolution)); rfft's n= transparently
    # zero-pads a short trial or truncates a long one to match.
    raw_t   = bundle.raw_t[0].numpy()
    recon_t = bundle.recon_t[0].numpy()
    T = raw_t.shape[-1]
    n_fft = int(round(fs / fft_resolution)) if fs else T

    def _demean_hann_rfft_np(x):
        x = x - x.mean(axis=-1, keepdims=True)
        win = np.hanning(x.shape[-1])
        return np.fft.rfft(x * win, n=n_fft, axis=-1)

    fft_raw   = _demean_hann_rfft_np(raw_t)
    fft_recon = _demean_hann_rfft_np(recon_t)
    psd_raw   = fft_raw.real**2   + fft_raw.imag**2
    psd_recon = fft_recon.real**2 + fft_recon.imag**2

    raw_power   = (bundle.raw_cnl   ** 2).mean(axis=(1, 2))
    recon_power = (bundle.recon_cnl ** 2).mean(axis=(1, 2))

    (used_ids, gal_importance, psd_ch_x_g, psd_x_g, gal_freqs, phase_ch_x_g,
     waveforms_g, iclabel_probs) = extract_flat_stamp_gallery(
        model, bundle.x_in, bundle.c_in, time_idx=bundle.t_in, valid_channels=bundle.vc_in,
        fs=fs, freq_resolution=fft_resolution)

    if psd_l_freq is not None and psd_h_freq is not None:
        band = (gal_freqs >= psd_l_freq) & (gal_freqs <= psd_h_freq)
        gal_freqs = gal_freqs[band]
        psd_x_g = psd_x_g[:, :, band]
        psd_raw, psd_recon = psd_raw[:, band], psd_recon[:, band]

    out_path = os.path.join(
        viz_dir, f"sub{bundle.subject_id}_trial{bundle.trial_idx}{epoch_tag}_stamp_gallery.png")
    plot_stamp_gallery(
        out_path, pos2d, raw_power, recon_power, psd_raw, psd_recon,
        psd_ch_x_g, psd_x_g, gal_freqs, gal_importance, cmap=cmap,
        phase_ch_x=phase_ch_x_g, waveforms=waveforms_g,
        subject_id=bundle.subject_id, trial_idx=bundle.trial_idx, epoch_tag=tagged_epoch_tag,
        unit_label='Stamp', unit_ids=used_ids, n_routed=model.n_routed_stamps,
        iclabel_probs=iclabel_probs,
    )
    print(f"  [snapshot] -> {out_path}")


def render_stamp_by_patch(bundle, config, out_dir, cmap='YlOrRd'):
    """Per-patch stamp selection grid for one trial (not in any preset; kept for occasional use)."""
    model = bundle.psd_model
    viz_dir = out_dir
    os.makedirs(viz_dir, exist_ok=True)

    pos2d = project_coords_2d(bundle.coords)
    epoch_tag = (f'_ep{bundle.epoch:04d}' if bundle.epoch is not None else '') + bundle.filename_tag
    tagged_epoch_tag = f'{epoch_tag}{bundle.title_suffix}'

    pp = config.get('preprocess_params', {})
    fs = pp.get('sample_freq')
    bandpass = pp.get('bandpass_filter', {})
    l_freq, h_freq = bandpass.get('l_freq'), bandpass.get('h_freq')
    viz_cfg = config.get('training_params', {}).get('visualize_params', {})
    fft_resolution = viz_cfg.get('fft_resolution', 0.2)
    psd_range = viz_cfg.get('psd_freq_range')
    psd_l_freq, psd_h_freq = tuple(psd_range) if psd_range else (l_freq, h_freq)

    grid = extract_flat_stamp_psd_by_patch(
        model, bundle.x_in, bundle.c_in, time_idx=bundle.t_in, valid_channels=bundle.vc_in,
        fs=fs, freq_resolution=fft_resolution, patch_stride=5)

    if psd_l_freq is not None and psd_h_freq is not None:
        band = (grid.freqs >= psd_l_freq) & (grid.freqs <= psd_h_freq)
        grid.freqs = grid.freqs[band]
        grid.psd = grid.psd[:, :, :, band]
        grid.recon_psd = grid.recon_psd[:, :, band]
        grid.raw_psd = grid.raw_psd[:, :, band]

    # Real-trial event marker: convert the onset from seconds to a displayed-COLUMN
    # index. grid.patch_ids holds the raw patch-n each displayed column represents;
    # patch n's own start time is n * model.patch_stride / fs -- searchsorted finds the
    # first displayed column at or after the onset. None (an assembled continuous
    # window has no single event) draws nothing, see plot_stamp_by_patch's onset_col doc.
    onset_col = None
    if bundle.event_onset_sec is not None and fs:
        onset_patch_n = bundle.event_onset_sec * fs / model.patch_stride
        onset_col = int(np.searchsorted(grid.patch_ids, onset_patch_n))

    out_path = os.path.join(
        viz_dir, f"sub{bundle.subject_id}_trial{bundle.trial_idx}{epoch_tag}_stamp_by_patch.png")
    plot_stamp_by_patch(
        out_path, pos2d, grid, cmap=cmap,
        subject_id=bundle.subject_id, trial_idx=bundle.trial_idx, epoch_tag=tagged_epoch_tag,
        unit_label='Stamp', n_routed=model.n_routed_stamps,
        signed_stamps=True,  # grid.topo is signed amp (mixing columns)
        onset_col=onset_col,
    )
    print(f"  [snapshot] -> {out_path}")
