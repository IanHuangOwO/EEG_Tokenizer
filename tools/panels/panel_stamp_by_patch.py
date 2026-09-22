"""stamp_by_patch panel: the real per-patch stamp selection grid for one trial
(tools/viz/stamp_plots.py's plot_stamp_by_patch, driven by tools/viz/extract.py's
extract_flat_stamp_psd_by_patch). Reads ctx.bundle and ctx.cmap -- the caller builds the
bundle via build_pretrain_bundle/build_finetune_bundle before selecting this panel."""
import os

import numpy as np

from tools.viz.extract import extract_flat_stamp_psd_by_patch
from tools.viz.stamp_plots import plot_stamp_by_patch
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
        out_path, pos2d, grid, cmap=ctx.cmap,
        subject_id=bundle.subject_id, trial_idx=bundle.trial_idx, epoch_tag=tagged_epoch_tag,
        unit_label='Stamp', n_routed=model.n_routed_stamps,
        signed_stamps=True,  # grid.topo is signed amp (mixing columns)
        onset_col=onset_col,
    )
    print(f"  [panel] -> {out_path}")
