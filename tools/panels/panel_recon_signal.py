"""recon_signal panel: band-filtered orig-vs-recon grid for one trial
(tools/viz/timeseries.py's visualize_reconstruction). Reads ctx.bundle (a
tools.analysis.snapshot.SnapshotBundle) -- the caller builds it via
build_pretrain_bundle/build_finetune_bundle before selecting this panel."""
import os

from tools.viz.timeseries import visualize_reconstruction

STAGES = frozenset({'pretrain', 'finetune'})
NEEDS_CHECKPOINT = True
NEEDS_DATASET = True


def run(ctx):
    bundle = ctx.bundle
    config = ctx.config
    viz_dir = os.path.join(ctx.output_dir, 'recon')
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
        tag=bundle.filename_tag.lstrip('_') + ('_' if bundle.filename_tag else ''),
        fs=fs or 200.0, l_freq=l_freq, h_freq=h_freq, band_edges=band_edges,
        event_onset_sec=bundle.event_onset_sec,
        valid_start=bundle.valid_start, valid_end=bundle.valid_end,
    )
