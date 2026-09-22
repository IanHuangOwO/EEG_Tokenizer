"""stamp_gallery panel: whole-trial Raw/Full-Recon view plus every stamp used somewhere
in the trial (tools/viz/panels.py's plot_stamp_gallery, driven by
tools/viz/extract.py's extract_flat_stamp_gallery). Reads ctx.bundle and ctx.cmap -- the
caller builds the bundle via build_pretrain_bundle/build_finetune_bundle before selecting
this panel."""
import os

import numpy as np

from tools.viz.extract import extract_flat_stamp_gallery
from tools.viz.panels import plot_stamp_gallery
from tools.viz.topomap import project_coords_2d

STAGES = frozenset({'pretrain', 'finetune'})
NEEDS_CHECKPOINT = True
NEEDS_DATASET = True


def run(ctx):
    bundle = ctx.bundle
    config = ctx.config
    model = bundle.psd_model
    viz_dir = os.path.join(ctx.output_dir, 'recon')
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
        psd_ch_x_g, psd_x_g, gal_freqs, gal_importance, cmap=ctx.cmap,
        phase_ch_x=phase_ch_x_g, waveforms=waveforms_g,
        subject_id=bundle.subject_id, trial_idx=bundle.trial_idx, epoch_tag=tagged_epoch_tag,
        unit_label='Stamp', unit_ids=used_ids, n_routed=model.n_routed_stamps,
        iclabel_probs=iclabel_probs,
    )
    print(f"  [panel] -> {out_path}")
