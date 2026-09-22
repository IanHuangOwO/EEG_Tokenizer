"""SnapshotBundle + bundle-builders for the recon_signal/stamp_by_patch/stamp_gallery
panels (tools/panels/panel_recon_signal.py, panel_stamp_by_patch.py,
panel_stamp_gallery.py). Moved and consolidated from the retired
model/base_checker.py's BaseEpochChecker + model/MeSAE/plugin.py's MeSAEChecker -- MeSAE
is the only registered model (MeFSQ removed, docs/adr/0013), so the override machinery
those classes existed for collapses into these two concrete functions.

build_pretrain_bundle/build_finetune_bundle each do the stage-specific part (how to
patchify/forward the model for one trial); the panels that read the resulting bundle own
the stage-agnostic rendering part. This mirrors the split model/base_checker.py's own
docstring described (check_pretrain/check_finetune build a bundle, _render_snapshot
renders it) -- only the renderer is now three separate panel files instead of one method."""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from model.MeSAE.MeSAE_modules import overlap_add_patches
from model.MeSAE.plugin import MeSAETrainer


@dataclass
class SnapshotBundle:
    """Stage-normalised input to the recon_signal/stamp_by_patch/stamp_gallery panels.
    Built by build_pretrain_bundle/build_finetune_bundle, each of which knows how to
    patchify/mask/forward its own stage; the panels know nothing about pretrain vs.
    finetune."""
    x_in: torch.Tensor           # [1, C, N, L] patches
    c_in: torch.Tensor           # [1, C, 3] coords
    t_in: torch.Tensor           # [1, N] time indices
    vc_in: torch.Tensor          # [1, C] valid-channel mask
    psd_model: nn.Module         # the model stamp_by_patch/stamp_gallery actually run on
                                  # (the full model for pretrain, model.backbone for finetune)
    raw_t: torch.Tensor          # [1, C, T] full-resolution raw signal
    recon_t: torch.Tensor        # [1, C, T] full-resolution reconstruction
    raw_cnl: np.ndarray          # [C, N, L] raw patches
    recon_cnl: np.ndarray        # [C, N, L] reconstruction patches
    coords: np.ndarray           # [C, 3]
    channel_names: List[str]
    valid_channels: np.ndarray   # [C] bool
    patch_len: int
    mask_np: Optional[np.ndarray] = None   # [C, N] masked-patch overlay, or None (finetune)
    title_suffix: str = ''                 # e.g. ' [finetune]'
    event_onset_sec: Optional[float] = None  # real-trial event onset (s into raw_t/recon_t)
    valid_start: Optional[int] = None  # [sample idx into raw_t/recon_t's T axis]
    valid_end: Optional[int] = None    # real content lies in [valid_start, valid_end)
    subject_id: Optional[int] = None
    trial_idx: Optional[int] = None
    epoch: Optional[int] = None
    filename_tag: str = ''


def _lookup_event_onset(config, dataset, trial_idx):
    """config['check']['event_onset_sample'] ({dataset_name: samples} dict, or a scalar
    for all) -> seconds, or None. Only meaningful for a genuine single real trial: an
    assembled continuous window (assemble_trials=True) mixes multiple real trials
    together with no one event to mark, so this deliberately returns None whenever
    base_dataset.assemble_trials is True rather than draw a misleading line. `is not
    None` (not truthiness) throughout -- an onset of literal 0 is a real, legitimate
    value, not "not configured"."""
    base_dataset = dataset.base_dataset
    if getattr(base_dataset, 'assemble_trials', True):
        return None
    eo = config.get('check', {}).get('event_onset_sample')
    if eo is None:
        return None
    fs = config.get('preprocess_params', {}).get('sample_freq')
    if not fs:
        return None
    if isinstance(eo, dict):
        base_idx = trial_idx % len(base_dataset)
        ds_name = base_dataset.dataset_names[base_idx]
        onset = eo.get(ds_name)
    else:
        onset = eo
    return (onset / fs) if onset is not None else None


def _lookup_valid_range(dataset, trial_idx):
    """(valid_start, valid_end) sample indices into raw_t/recon_t's T axis -- real content
    lies in [valid_start, valid_end), everything outside is compile-time zero-pad."""
    base_dataset = dataset.base_dataset
    base_idx = trial_idx % len(base_dataset)
    return int(base_dataset.row_valid_start[base_idx]), int(base_dataset.row_valid_end[base_idx])


def _patchify(x, patch_len):
    C, T = x.shape
    P = T // patch_len
    x_patches = x[:, :P * patch_len].reshape(C, P, patch_len).unsqueeze(0)
    time_idx = torch.arange(P, dtype=torch.long).unsqueeze(0)
    return x_patches, time_idx


@torch.no_grad()
def _run_reconstruction(model, dataset, trial_idx, device):
    """Always runs unmasked (bool_masked_pos not passed) for a clean reconstruction
    snapshot, regardless of whether the model is currently in the Masked training stage."""
    x_patches, coords, _, time_indices, _, _, valid_channels = dataset[trial_idx]
    C, N, L = x_patches.shape
    pp = dataset.base_dataset.config['preprocess_params']
    fs = pp['sample_freq']
    stride = pp.get('patch_stride', L)

    x_in      = x_patches.unsqueeze(0).to(device)
    coords_in = coords.unsqueeze(0).to(device)
    t_in      = time_indices.unsqueeze(0).to(device)
    vc_in     = valid_channels.unsqueeze(0).to(device)

    out = model(x_in, coords=coords_in, time_idx=t_in, valid_channels=vc_in)
    raw_stitched   = overlap_add_patches(x_patches.to(device), stride)
    recon_stitched = overlap_add_patches(out.recon[0], stride)
    T_total = raw_stitched.shape[-1]

    return {
        'raw':    raw_stitched.cpu().numpy(),
        'recon':  recon_stitched.cpu().numpy(),
        'coords': coords.numpy(),
        'T': T_total, 'N': N, 'L': L, 'fs': fs,
    }


@torch.no_grad()
def build_pretrain_bundle(model, dataset, trial_idx, config, device,
                           subject_id=None, epoch=None):
    """Pretrain-stage bundle: full (masked-phase-restored) model forward, unmasked
    reconstruction, per-trial stamp usage for title colors. Returns (bundle, metrics)
    where metrics = {'recon_mse': ..., **MeSAETrainer().epoch_metrics(model, out)},
    matching today's BaseEpochChecker.check_pretrain's returned metrics dict exactly."""
    was_training = model.training
    model.eval()
    try:
        x_patches, coords, mask, time_indices, _, _, valid_channels = dataset[trial_idx]
        x_in  = x_patches.unsqueeze(0).to(device)
        c_in  = coords.unsqueeze(0).to(device)
        t_in  = time_indices.unsqueeze(0).to(device)
        vc_in = valid_channels.unsqueeze(0).to(device)

        data = _run_reconstruction(model, dataset, trial_idx, device)

        C, N, patch_len = x_patches.shape
        mask_np = mask.numpy().reshape(C, N)

        out = model(x_in, c_in, time_idx=t_in, bool_masked_pos=None, valid_channels=vc_in)
        recon_cnl = out.recon[0].detach().cpu().numpy()

        metrics = {'recon_mse': float(np.mean((data['raw'] - data['recon']) ** 2))}
        metrics.update(MeSAETrainer().epoch_metrics(model, out))

        event_onset_sec = _lookup_event_onset(config, dataset, trial_idx)
        valid_start, valid_end = _lookup_valid_range(dataset, trial_idx)

        bundle = SnapshotBundle(
            x_in=x_in, c_in=c_in, t_in=t_in, vc_in=vc_in, psd_model=model,
            raw_t=torch.from_numpy(data['raw']).unsqueeze(0),
            recon_t=torch.from_numpy(data['recon']).unsqueeze(0),
            raw_cnl=x_patches.numpy(), recon_cnl=recon_cnl,
            coords=coords.numpy(), channel_names=dataset.base_dataset.channel_names,
            valid_channels=valid_channels.numpy(), patch_len=patch_len, mask_np=mask_np,
            event_onset_sec=event_onset_sec, valid_start=valid_start, valid_end=valid_end,
            subject_id=subject_id, trial_idx=trial_idx, epoch=epoch,
        )
        return bundle, metrics
    finally:
        model.train(was_training)


@torch.no_grad()
def build_finetune_bundle(model, dataset, trial_idx, config, device,
                           subject_id=None, tag=''):
    """Finetune-stage bundle: model.backbone forward on one patchified trial.
    tag: extra filename/title suffix (e.g. '_target2_Feet_correct') -- folded into the
    bundle's filename_tag, leading underscore, filename-safe, caller's responsibility."""
    backbone = model.backbone
    pp = config.get('preprocess_params', {})
    patch_len = pp.get('patch_length', 100)

    x_raw, coords, label, valid_channels, valid_length = dataset[trial_idx]
    x_patches, time_idx = _patchify(x_raw, patch_len)
    x_in = x_patches.to(device)
    c_in = coords.unsqueeze(0).to(device)
    t_in = time_idx.to(device)
    vc_in = valid_channels.unsqueeze(0).to(device)

    channel_names = dataset.base_dataset.channel_names
    C, N, L = x_patches.shape[1], x_patches.shape[2], patch_len

    was_training = model.training
    model.eval()
    try:
        out = backbone(x_in, c_in, time_idx=t_in, bool_masked_pos=None, valid_channels=vc_in)
        raw_cnl   = x_patches[0].numpy()
        recon_cnl = out.recon[0].reshape(C, N, L).detach().cpu().numpy()

        metrics = {'recon_mse': float(np.mean((raw_cnl - recon_cnl) ** 2))}
        metrics.update(MeSAETrainer().epoch_metrics(backbone, out))

        bundle = SnapshotBundle(
            x_in=x_in, c_in=c_in, t_in=t_in, vc_in=vc_in, psd_model=backbone,
            raw_t=torch.from_numpy(raw_cnl.reshape(1, C, N * L)),
            recon_t=torch.from_numpy(recon_cnl.reshape(1, C, N * L)),
            raw_cnl=raw_cnl, recon_cnl=recon_cnl,
            coords=coords.numpy(), channel_names=channel_names,
            valid_channels=valid_channels.numpy(), patch_len=patch_len, mask_np=None,
            title_suffix=' [finetune]',
            subject_id=subject_id, trial_idx=trial_idx, filename_tag=tag,
        )
        return bundle, metrics
    finally:
        model.train(was_training)
