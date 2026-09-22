# Snapshot panels migration (Sub-project B) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `BaseEpochChecker`/`MeSAEChecker` with three `tools/panels/panel_*.py`
files (`recon_signal`, `stamp_by_patch`, `stamp_gallery`) backed by two bundle-builders in
`tools/analysis/snapshot.py`. Migrate all three real callers
(`analysis_pretrain.py`, `analysis_finetune.py`, `train_pretrain.py`) to the new panels,
then delete the retired class hierarchy and its confirmed-dead code.

**Architecture:** Build the new pieces first (bundle-builders, then panels), verified in
isolation against the still-live old code. Migrate each caller one task at a time,
verified against a captured pre-migration baseline. Delete the old code only in the last
task, once nothing calls it anymore.

**Tech Stack:** Python, PyTorch, `eeg_fm` conda env
(`/home/mamechin/anaconda3/envs/eeg_fm/bin/python`).

**Spec:** `docs/superpowers/specs/2026-09-22-snapshot-panels-migration-design.md`

## Global Constraints

- No test suite exists (CLAUDE.md) — validation is smoke runs against real
  checkpoints/training, never pytest files.
- `probes/` stays untouched — standing ruling for this whole effort.
- `model/base_codebook_checker.py`, `MeSAECodebookChecker`,
  `analysis_pretrain.py`'s `--analysis codebook` path are out of scope — sub-project C.
- Every commit ends with:
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18`

**Interface refinements beyond the spec (decided here, binding for every task):**
`SnapshotBundle` gains four fields the spec's text didn't enumerate but the panels need:
`subject_id: Optional[int] = None`, `trial_idx: Optional[int] = None`,
`epoch: Optional[int] = None`, `filename_tag: str = ''`. `PanelContext` gains `cmap: str =
'YlOrRd'` in addition to the spec's `bundle` field — panels need a render color map, which
is a per-CLI-invocation setting, not per-trial, so it belongs on `PanelContext`, not the
bundle. `run_panels` itself is NOT changed (no new try/except) — each panel's `run(ctx)`
propagates exceptions naturally, same as `recon_signal`'s rendering does today (it was
never wrapped in try/except); only `train_pretrain.py` keeps resilience, via its own
existing caller-side `try/except` around the whole bundle-build-and-render call, unchanged
in shape from today. This is a narrow, disclosed behavior change: today, a
`stamp_by_patch`/`stamp_gallery` extraction failure in the CLI scripts is caught and
logged (`_render_snapshot`'s `try/except` around `extract_psd`); after this migration, the
same failure raises normally in the CLI scripts (train_pretrain.py is unaffected — its own
try/except still catches it). Task 6's testing step calls this out explicitly.

---

## Task 1: `tools/analysis/snapshot.py` — `SnapshotBundle` + bundle-builders

**Files:**
- Create: `tools/analysis/snapshot.py`

**Interfaces:**
- Produces: `SnapshotBundle` (dataclass — all fields from today's
  `model/base_checker.py:27-61` PLUS the four new fields listed under Global Constraints
  above), `build_pretrain_bundle(model, dataset, trial_idx, config, device,
  subject_id=None, epoch=None) -> (SnapshotBundle, dict)`, `build_finetune_bundle(model,
  dataset, trial_idx, config, device, subject_id=None, tag='') -> (SnapshotBundle, dict)`.
  Task 2's panels read `SnapshotBundle`'s fields. Tasks 3-5 call the two builder functions.

- [ ] **Step 1: Write `tools/analysis/snapshot.py`**

```python
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
    unit_colors: Optional[List[str]] = None  # [Q] per-stamp title/label color override
    unit_ids: Optional[np.ndarray] = None    # [Q] real global stamp ids, or None
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


def _compute_unit_colors(model, out):
    """red = shared stamp (always-on, structural). black = routed stamp. Restricted to
    stamps actually used somewhere in this trial, capped at 100 (see
    MeSAEPretrain.used_stamp_ids) -- with n_stamps=800 and hard top-k selection, showing
    every stamp regardless of whether this trial ever touched it is mostly noise."""
    used_ids = model.used_stamp_ids(out, max_stamps=100)
    colors = ['red' if i >= model.n_routed_stamps else 'black' for i in used_ids.tolist()]
    return colors, used_ids


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

        unit_colors, used_ids = _compute_unit_colors(model, out)

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
            unit_colors=unit_colors, unit_ids=used_ids.cpu().numpy(),
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

        unit_colors, _used_ids = _compute_unit_colors(backbone, out)

        bundle = SnapshotBundle(
            x_in=x_in, c_in=c_in, t_in=t_in, vc_in=vc_in, psd_model=backbone,
            raw_t=torch.from_numpy(raw_cnl.reshape(1, C, N * L)),
            recon_t=torch.from_numpy(recon_cnl.reshape(1, C, N * L)),
            raw_cnl=raw_cnl, recon_cnl=recon_cnl,
            coords=coords.numpy(), channel_names=channel_names,
            valid_channels=valid_channels.numpy(), patch_len=patch_len, mask_np=None,
            title_suffix=' [finetune]', unit_colors=unit_colors,
            subject_id=subject_id, trial_idx=trial_idx, filename_tag=tag,
        )
        return bundle, metrics
    finally:
        model.train(was_training)
```

- [ ] **Step 2: Smoke — import clean, and a real bundle build against a live checkpoint**

```bash
CUDA_VISIBLE_DEVICES='' /home/mamechin/anaconda3/envs/eeg_fm/bin/python -c "
import json, torch
from tools.analysis import load_model
from tools.analysis.snapshot import build_pretrain_bundle
cfg = json.load(open('output/pretrain/mesae_v10_all_share/artifacts/config.json'))
device = torch.device('cpu')
mdl = load_model(cfg, 'output/pretrain/mesae_v10_all_share/checkpoint/last.pth', device, mode='pretrain')
from tools.analysis import filter_config_to_subject, select_subject_dataset
from IO.dataset import build_dataset_from_config
ds_name, subject = select_subject_dataset(cfg, None, dataset_name='BNCI2014001', mode='pretrain')
filtered = filter_config_to_subject(cfg, ds_name, subject, mode='pretrain')
ds = build_dataset_from_config(filtered, mode='pretrain', assemble_trials=False)
bundle, metrics = build_pretrain_bundle(mdl, ds, 0, filtered, device, subject_id=subject)
print('bundle ok, raw_t shape', bundle.raw_t.shape, 'metrics', metrics)
"
```
Expected: `bundle ok, raw_t shape torch.Size([1, ...])`, a `metrics` dict containing at
least `recon_mse`. This proves the bundle-builder works standalone, before any panel or
caller depends on it.

- [ ] **Step 3: `git status` sanity check**

```bash
git status --short tools/analysis/snapshot.py
```
Expected: exactly one new untracked file.

- [ ] **Step 4: Commit**

```bash
git add tools/analysis/snapshot.py
git commit -m "feat: add tools/analysis/snapshot.py -- SnapshotBundle + pretrain/finetune bundle-builders

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18"
```

---

## Task 2: Three snapshot panels + `PanelContext` extension

**Files:**
- Modify: `tools/panels/__init__.py`
- Create: `tools/panels/panel_recon_signal.py`
- Create: `tools/panels/panel_stamp_by_patch.py`
- Create: `tools/panels/panel_stamp_gallery.py`

**Interfaces:**
- Consumes: `tools.analysis.snapshot.SnapshotBundle` (Task 1) — read via `ctx.bundle`.
- Produces: `PanelContext.bundle`/`PanelContext.cmap` (new fields), three panels
  (`recon_signal`, `stamp_by_patch`, `stamp_gallery`) discoverable via
  `tools.panels.discover_panel_names()`. Tasks 3-5 select these panel names and pass a
  `PanelContext` with `bundle`/`cmap` set.

- [ ] **Step 1: Add `bundle` and `cmap` to `PanelContext`**

Read `tools/panels/__init__.py` first. Replace:
```python
@dataclass
class PanelContext:
    config: dict
    output_dir: str
    device: object            # torch.device -- kept untyped here so this module doesn't
                               # need to import torch just to define the dataclass
    args: object               # argparse.Namespace, for panel-specific CLI flags
    model: Optional[object] = None
    dataset: Optional[object] = None
    checkpoint: Optional[str] = None
```
with:
```python
@dataclass
class PanelContext:
    config: dict
    output_dir: str
    device: object            # torch.device -- kept untyped here so this module doesn't
                               # need to import torch just to define the dataclass
    args: object               # argparse.Namespace, for panel-specific CLI flags
    model: Optional[object] = None
    dataset: Optional[object] = None
    checkpoint: Optional[str] = None
    bundle: Optional[object] = None   # a tools.analysis.snapshot.SnapshotBundle, for
                                       # panels that render one already-prepared trial
                                       # (kept untyped for the same reason as model/dataset
                                       # above -- this package stays import-light)
    cmap: str = 'YlOrRd'       # matplotlib colormap, shared by every panel this ctx runs
```

- [ ] **Step 2: Write `tools/panels/panel_recon_signal.py`**

```python
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
```

- [ ] **Step 3: Write `tools/panels/panel_stamp_by_patch.py`**

```python
"""stamp_by_patch panel: the real per-patch stamp selection grid for one trial
(tools/viz/panels.py's plot_stamp_by_patch, driven by tools/viz/extract.py's
extract_flat_stamp_psd_by_patch). Reads ctx.bundle and ctx.cmap -- the caller builds the
bundle via build_pretrain_bundle/build_finetune_bundle before selecting this panel."""
import os

import numpy as np

from tools.viz.extract import extract_flat_stamp_psd_by_patch
from tools.viz.panels import plot_stamp_by_patch
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
```

- [ ] **Step 4: Write `tools/panels/panel_stamp_gallery.py`**

```python
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
```

- [ ] **Step 5: Smoke — imports clean, panels discoverable**

```bash
CUDA_VISIBLE_DEVICES='' /home/mamechin/anaconda3/envs/eeg_fm/bin/python -c "
import tools.panels, tools.panels.panel_recon_signal, tools.panels.panel_stamp_by_patch, tools.panels.panel_stamp_gallery
print(sorted(tools.panels.discover_panel_names()))
"
```
Expected: `['profile', 'recon_signal', 'stamp_by_patch', 'stamp_gallery']`.

- [ ] **Step 6: Smoke — run all three against a real checkpoint, same trial as Task 1's bundle test**

```bash
CUDA_VISIBLE_DEVICES='' /home/mamechin/anaconda3/envs/eeg_fm/bin/python -c "
import json, torch
from tools.analysis import load_model, filter_config_to_subject, select_subject_dataset, resolve_output_dir
from tools.analysis.snapshot import build_pretrain_bundle
from tools.panels import PanelContext, run_panels
from IO.dataset import build_dataset_from_config

cfg = json.load(open('output/pretrain/mesae_v10_all_share/artifacts/config.json'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mdl = load_model(cfg, 'output/pretrain/mesae_v10_all_share/checkpoint/last.pth', device, mode='pretrain')
ds_name, subject = select_subject_dataset(cfg, None, dataset_name='BNCI2014001', mode='pretrain')
filtered = filter_config_to_subject(cfg, ds_name, subject, mode='pretrain')
ds = build_dataset_from_config(filtered, mode='pretrain', assemble_trials=False)
bundle, metrics = build_pretrain_bundle(mdl, ds, 0, filtered, device, subject_id=subject)
out = resolve_output_dir(filtered, 'analysis', ds_name, mode='pretrain')
ctx = PanelContext(config=filtered, output_dir=out, device=device, args=None, model=mdl,
                    dataset=ds, bundle=bundle)
run_panels(['recon_signal', 'stamp_by_patch', 'stamp_gallery'], 'pretrain', ctx)
print('metrics:', metrics)
"
```
Expected: three `[panel] -> ...` lines naming the three PNG files, no traceback, then a
printed `metrics` dict. Confirm the three files actually exist on disk afterward
(`ls output/mesae_v10_all_share/analysis/BNCI2014001/recon/ | tail -3` or equivalent path
under `out`).

- [ ] **Step 7: `git diff --stat` sanity check**

```bash
git status --short tools/panels
```
Expected: one modified file (`tools/panels/__init__.py`) and three new untracked files.

- [ ] **Step 8: Commit**

```bash
git add tools/panels
git commit -m "feat: add recon_signal/stamp_by_patch/stamp_gallery panels

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18"
```

---

## Task 3: Migrate `analysis_pretrain.py`

**Files:**
- Modify: `analysis_pretrain.py`

**Interfaces:**
- Consumes: `tools.analysis.snapshot.build_pretrain_bundle` (Task 1),
  `tools.panels.PanelContext`/`run_panels` (Task 2, already imported for the `--panel`
  branch — this task adds a second import for the same names used by the legacy snapshot
  path).

- [ ] **Step 1: Capture a baseline before touching anything**

```bash
/home/mamechin/anaconda3/envs/eeg_fm/bin/python analysis_pretrain.py --config config/analysis.json \
  --checkpoint output/pretrain/mesae_v10_all_share/checkpoint/last.pth \
  --analysis snapshot --dataset BNCI2014001 --subject 1 --trial 0
find output/mesae_v10_all_share/analysis -type f | sort > /tmp/pretrain_snapshot_baseline.txt
cat /tmp/pretrain_snapshot_baseline.txt
```
Note the printed `recon_mse=...` value too — you'll compare both after migrating.

- [ ] **Step 2: Delete `run()` and the now-unused top-level `MODEL_REGISTRY` import**

Read the file first. Replace:
```python
import os
import json

from model.factory import MODEL_REGISTRY


def _cap_subjects_by_trial_budget(cfg, ds_args, max_trials, rng, min_subjects=20):
```
with:
```python
import os
import json


def _cap_subjects_by_trial_budget(cfg, ds_args, max_trials, rng, min_subjects=20):
```

Then delete the `run(...)` function entirely:
```python
def run(config, output_dir, model, dataset, trial_idx, subject_id=None,
        epoch=None, cmap='YlOrRd', plot_recon=True, plot_topo_psd=True, plot_attn_topo=True):
    model_type = 'MeSAE'  # only registered model (MeFSQ removed, docs/adr/0013)
    plugin  = MODEL_REGISTRY[model_type]
    checker = plugin.checker_cls()
    trainer = plugin.trainer_cls()
    return checker.check_pretrain(
        config, output_dir, model, dataset, trial_idx,
        subject_id=subject_id, epoch=epoch, cmap=cmap,
        plot_recon=plot_recon, plot_topo_psd=plot_topo_psd, plot_attn_topo=plot_attn_topo,
        trainer=trainer,
    )


```
(the two blank lines before `if __name__ == '__main__':` collapse to the one that was
already between `run(...)` and it).

- [ ] **Step 3: Import the new pieces in `__main__`**

Replace:
```python
    from tools.analysis import (
        _deep_merge, load_model,
        select_subject_dataset, filter_config_to_subject, pick_trial, resolve_output_dir,
    )
```
with:
```python
    from tools.analysis import (
        _deep_merge, load_model,
        select_subject_dataset, filter_config_to_subject, pick_trial, resolve_output_dir,
    )
    from tools.analysis.snapshot import build_pretrain_bundle
    from tools.panels import PanelContext, run_panels
```

- [ ] **Step 4: Migrate the `targets` branch**

Replace:
```python
            targets = cfg['training_params']['visualize_params'][data_mode]['targets']
            for t in targets:
                t_dataset = t.get('dataset')
                subj = t.get('subject') if t.get('subject') is not None else _first_subject(t_dataset)
                t_idx, subject_id = pick_trial(ds, subj, trial=t.get('trial'), dataset_name=t_dataset)
                out = resolve_output_dir(cfg, 'analysis', t_dataset or 'multi', mode=mode)
                metrics = run(
                    cfg, out, mdl, ds, t_idx, subject_id=subject_id, cmap=cmap,
                    plot_recon=check_cfg.get('plot_recon', True),
                    plot_topo_psd=check_cfg.get('plot_topo_psd', True),
                    plot_attn_topo=check_cfg.get('plot_attn_topo', True),
                )
                metrics_str = '  '.join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[check] done: dataset={t_dataset} subject={subject_id} trial_idx={t_idx}  |  {metrics_str}")
```
with:
```python
            targets = cfg['training_params']['visualize_params'][data_mode]['targets']
            for t in targets:
                t_dataset = t.get('dataset')
                subj = t.get('subject') if t.get('subject') is not None else _first_subject(t_dataset)
                t_idx, subject_id = pick_trial(ds, subj, trial=t.get('trial'), dataset_name=t_dataset)
                out = resolve_output_dir(cfg, 'analysis', t_dataset or 'multi', mode=mode)
                bundle, metrics = build_pretrain_bundle(mdl, ds, t_idx, cfg, device, subject_id=subject_id)
                panels = []
                if check_cfg.get('plot_recon', True):
                    panels.append('recon_signal')
                if check_cfg.get('plot_topo_psd', True):
                    panels += ['stamp_by_patch', 'stamp_gallery']
                panel_ctx = PanelContext(config=cfg, output_dir=out, device=device, args=args,
                                          model=mdl, dataset=ds, cmap=cmap, bundle=bundle)
                run_panels(panels, 'pretrain', panel_ctx)
                metrics_str = '  '.join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[check] done: dataset={t_dataset} subject={subject_id} trial_idx={t_idx}  |  {metrics_str}")
```

- [ ] **Step 5: Migrate the default (else) branch**

Replace:
```python
            for subject in subjects:
                ds_name, subject = select_subject_dataset(cfg, subject, dataset_name=dataset_name, mode=data_mode)
                filtered  = filter_config_to_subject(cfg, ds_name, subject, mode=data_mode)
                # assemble_trials=False -> one snapshot = one real trial, patch count matches
                # the trial's own length (see the targets branch above for the full rationale).
                ds        = build_dataset_from_config(filtered, mode=data_mode, assemble_trials=False)
                trial_cfg = args.trial if args.trial is not None else ds_cfg.get('trial_to_use')
                t_idx, subject_id = pick_trial(ds, subject, trial_cfg, dataset_name=ds_name)
                out = resolve_output_dir(filtered, 'analysis', ds_name, mode=mode)
                metrics = run(
                    filtered, out, mdl, ds, t_idx, subject_id=subject_id, cmap=cmap,
                    plot_recon=check_cfg.get('plot_recon', True),
                    plot_topo_psd=check_cfg.get('plot_topo_psd', True),
                    plot_attn_topo=check_cfg.get('plot_attn_topo', True),
                )
                metrics_str = '  '.join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[check] done: dataset={ds_name} subject={subject_id} trial_idx={t_idx}  |  {metrics_str}")
```
with:
```python
            for subject in subjects:
                ds_name, subject = select_subject_dataset(cfg, subject, dataset_name=dataset_name, mode=data_mode)
                filtered  = filter_config_to_subject(cfg, ds_name, subject, mode=data_mode)
                # assemble_trials=False -> one snapshot = one real trial, patch count matches
                # the trial's own length (see the targets branch above for the full rationale).
                ds        = build_dataset_from_config(filtered, mode=data_mode, assemble_trials=False)
                trial_cfg = args.trial if args.trial is not None else ds_cfg.get('trial_to_use')
                t_idx, subject_id = pick_trial(ds, subject, trial_cfg, dataset_name=ds_name)
                out = resolve_output_dir(filtered, 'analysis', ds_name, mode=mode)
                bundle, metrics = build_pretrain_bundle(mdl, ds, t_idx, filtered, device, subject_id=subject_id)
                panels = []
                if check_cfg.get('plot_recon', True):
                    panels.append('recon_signal')
                if check_cfg.get('plot_topo_psd', True):
                    panels += ['stamp_by_patch', 'stamp_gallery']
                panel_ctx = PanelContext(config=filtered, output_dir=out, device=device, args=args,
                                          model=mdl, dataset=ds, cmap=cmap, bundle=bundle)
                run_panels(panels, 'pretrain', panel_ctx)
                metrics_str = '  '.join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[check] done: dataset={ds_name} subject={subject_id} trial_idx={t_idx}  |  {metrics_str}")
```

- [ ] **Step 6: Smoke — imports clean**

```bash
CUDA_VISIBLE_DEVICES='' /home/mamechin/anaconda3/envs/eeg_fm/bin/python -c "import analysis_pretrain; print('ok')"
```

- [ ] **Step 7: Smoke — rerun the exact baseline command, compare**

```bash
/home/mamechin/anaconda3/envs/eeg_fm/bin/python analysis_pretrain.py --config config/analysis.json \
  --checkpoint output/pretrain/mesae_v10_all_share/checkpoint/last.pth \
  --analysis snapshot --dataset BNCI2014001 --subject 1 --trial 0
find output/mesae_v10_all_share/analysis -type f | sort > /tmp/pretrain_snapshot_after.txt
diff /tmp/pretrain_snapshot_baseline.txt /tmp/pretrain_snapshot_after.txt
```
Expected: `diff` prints nothing (same three filenames — the render calls moved verbatim,
filenames didn't change) — and the printed `recon_mse=...` value matches the baseline
(deterministic given `model.eval()`, no dropout).

- [ ] **Step 8: `git diff --stat` sanity check**

```bash
git diff --stat analysis_pretrain.py
```
Expected: net negative or roughly neutral (a 12-line `run()` function and its import
removed, replaced by a few lines per loop) — not a rewrite of surrounding code.

- [ ] **Step 9: Commit**

```bash
git add analysis_pretrain.py
git commit -m "refactor: migrate analysis_pretrain.py's snapshot loops to the new panels

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18"
```

---

## Task 4: Migrate `analysis_finetune.py`

**Files:**
- Modify: `analysis_finetune.py`

**Interfaces:**
- Consumes: `tools.analysis.snapshot.build_finetune_bundle` (Task 1),
  `tools.panels.PanelContext`/`run_panels` (Task 2).

- [ ] **Step 1: Capture a baseline before touching anything**

```bash
cat > /tmp/finetune_snapshot_overlay.json <<'EOF'
{
  "checkpoint": "output/baseline/BNCI2014001_intra_raw_band/finetune/run_1_fold0/head.pth",
  "mode": "finetune",
  "dataset_params": {
    "finetune": {
      "BNCI2014001": {"dataset_path": "datas/finetune/BNCI2014001", "subject_to_use": ["1"], "channels_to_use": ["all"]}
    }
  },
  "check": {"plot_recon": true, "plot_topo_psd": true, "cmap": "YlOrRd"}
}
EOF
/home/mamechin/anaconda3/envs/eeg_fm/bin/python analysis_finetune.py \
  --config /tmp/finetune_snapshot_overlay.json \
  --base-config output/baseline/BNCI2014001_intra_raw_band/artifacts/config.json \
  --dataset BNCI2014001 > /tmp/finetune_snapshot_baseline.log 2>&1
cat /tmp/finetune_snapshot_baseline.log
find output/baseline/BNCI2014001_intra_raw_band/analysis -type f | sort > /tmp/finetune_snapshot_baseline_files.txt
```
Confirm exit 0 and a handful of `[check] done: ...` lines (one per correct/wrong example
found, up to `num_classes * 2`).

- [ ] **Step 2: Delete `run()` and the now-unused top-level `MODEL_REGISTRY` import**

Read the file first. Replace:
```python
import os
import json

from model.factory import MODEL_REGISTRY


def _load_target_names(dataset_path, num_classes):
```
with:
```python
import os
import json


def _load_target_names(dataset_path, num_classes):
```

Then delete the `run(...)` function entirely:
```python
def run(config, output_dir, model, dataset, trial_idx, subject_id=None,
        epoch=None, cmap='YlOrRd', plot_recon=True, plot_topo_psd=True, tag=''):
    model_type = 'MeSAE'  # only registered model (MeFSQ removed, docs/adr/0013)
    plugin  = MODEL_REGISTRY[model_type]
    checker = plugin.checker_cls()
    trainer = plugin.trainer_cls()
    return checker.check_finetune(
        config, output_dir, model, dataset, trial_idx,
        subject_id=subject_id, epoch=epoch, cmap=cmap,
        plot_recon=plot_recon, plot_topo_psd=plot_topo_psd,
        trainer=trainer, tag=tag,
    )


```

- [ ] **Step 3: Import the new pieces in `__main__`**

Replace:
```python
    from tools.analysis import _deep_merge, load_model, resolve_output_dir
```
with:
```python
    from tools.analysis import _deep_merge, load_model, resolve_output_dir
    from tools.analysis.snapshot import build_finetune_bundle
    from tools.panels import PanelContext, run_panels
```

- [ ] **Step 4: Migrate the per-class loop**

Replace:
```python
            t_idx = int(idxs[0])
            subject_id = int(ds.base_dataset.subject_data[t_idx].item())
            tag = f'_target{cls_idx}_{safe}_{status}'
            metrics = run(
                filtered, out, mdl, ds, t_idx, subject_id=subject_id, cmap=cmap,
                plot_recon=check_cfg.get('plot_recon', True),
                plot_topo_psd=check_cfg.get('plot_topo_psd', True),
                tag=tag,
            )
            metrics_str = '  '.join(f"{k}={v:.4f}" for k, v in metrics.items())
```
with:
```python
            t_idx = int(idxs[0])
            subject_id = int(ds.base_dataset.subject_data[t_idx].item())
            tag = f'_target{cls_idx}_{safe}_{status}'
            bundle, metrics = build_finetune_bundle(mdl, ds, t_idx, filtered, device,
                                                      subject_id=subject_id, tag=tag)
            panels = []
            if check_cfg.get('plot_recon', True):
                panels.append('recon_signal')
            if check_cfg.get('plot_topo_psd', True):
                panels += ['stamp_by_patch', 'stamp_gallery']
            panel_ctx = PanelContext(config=filtered, output_dir=out, device=device, args=args,
                                      model=mdl, dataset=ds, cmap=cmap, bundle=bundle)
            run_panels(panels, 'finetune', panel_ctx)
            metrics_str = '  '.join(f"{k}={v:.4f}" for k, v in metrics.items())
```

- [ ] **Step 5: Smoke — imports clean**

```bash
CUDA_VISIBLE_DEVICES='' /home/mamechin/anaconda3/envs/eeg_fm/bin/python -c "import analysis_finetune; print('ok')"
```

- [ ] **Step 6: Smoke — rerun the exact baseline command, compare**

```bash
/home/mamechin/anaconda3/envs/eeg_fm/bin/python analysis_finetune.py \
  --config /tmp/finetune_snapshot_overlay.json \
  --base-config output/baseline/BNCI2014001_intra_raw_band/artifacts/config.json \
  --dataset BNCI2014001 > /tmp/finetune_snapshot_after.log 2>&1
diff /tmp/finetune_snapshot_baseline.log /tmp/finetune_snapshot_after.log
find output/baseline/BNCI2014001_intra_raw_band/analysis -type f | sort > /tmp/finetune_snapshot_after_files.txt
diff /tmp/finetune_snapshot_baseline_files.txt /tmp/finetune_snapshot_after_files.txt
```
Expected: both `diff`s print nothing — same `[check] done: ...` lines (same `recon_mse`
values, deterministic) and the same set of output files.

- [ ] **Step 7: `git diff --stat` sanity check**

```bash
git diff --stat analysis_finetune.py
```
Expected: net negative or roughly neutral.

- [ ] **Step 8: Commit**

```bash
git add analysis_finetune.py
git commit -m "refactor: migrate analysis_finetune.py's per-class loop to the new panels

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18"
```

---

## Task 5: Migrate `train_pretrain.py`'s periodic snapshot

**Files:**
- Modify: `train_pretrain.py`

**Interfaces:**
- Consumes: `tools.analysis.snapshot.build_pretrain_bundle` (Task 1),
  `tools.panels.PanelContext`/`run_panels` (Task 2). In-process call, no CLI/argparse
  involved.

- [ ] **Step 1: Read the file's imports and the checker call site**

Read `train_pretrain.py` in full (or at minimum its imports at the top and lines
260-395) before editing — confirm the exact current text matches what's quoted below; if
it doesn't, stop and report BLOCKED rather than guessing.

- [ ] **Step 2: Add the new imports**

Replace:
```python
from model.factory import build_pretrain_from_config, optimizer_param_groups, MODEL_REGISTRY
from tools.analysis import pick_trial
```
with:
```python
from model.factory import build_pretrain_from_config, optimizer_param_groups, MODEL_REGISTRY
from tools.analysis import pick_trial
from tools.analysis.snapshot import build_pretrain_bundle
from tools.panels import PanelContext, run_panels
```

- [ ] **Step 3: Fix a stale comment naming two files this effort already renamed/deleted**

Replace:
```python
    # Separate assemble_trials=False dataset just for periodic snapshot viz -- val_dataset
    # itself stays assembled (assemble_trials=True, continuous windows) for real
    # train/val loss. A snapshot built from an assembled window mixes multiple real
    # trials with no single event to mark, so _lookup_event_onset (model/base_checker.py)
    # silently drops the recon_signal/stamp_by_patch onset line whenever it's handed one.
    # Same assemble_trials=False dataset check_model.py's own snapshot path already uses.
```
with:
```python
    # Separate assemble_trials=False dataset just for periodic snapshot viz -- val_dataset
    # itself stays assembled (assemble_trials=True, continuous windows) for real
    # train/val loss. A snapshot built from an assembled window mixes multiple real
    # trials with no single event to mark, so _lookup_event_onset
    # (tools/analysis/snapshot.py) silently drops the recon_signal/stamp_by_patch onset
    # line whenever it's handed one. Same assemble_trials=False dataset
    # analysis_pretrain.py's own snapshot path already uses.
```

- [ ] **Step 4: Drop the now-unused `checker` variable**

Replace:
```python
    model_type = train_params.get('model_type', 'MeSAE')
    entry      = MODEL_REGISTRY[model_type]
    trainer    = entry.trainer_cls()
    checker    = entry.checker_cls()
```
with:
```python
    model_type = train_params.get('model_type', 'MeSAE')
    entry      = MODEL_REGISTRY[model_type]
    trainer    = entry.trainer_cls()
```

- [ ] **Step 5: Migrate the periodic snapshot call**

Replace:
```python
        if epoch == total_epochs if viz_every_n == 'last' else (viz_every_n > 0 and epoch % viz_every_n == 0):
            for topo_trial_idx, topo_subject_id in viz_targets:
                try:
                    checker.check_pretrain(
                        config, vis_dir, model, viz_dataset,
                        topo_trial_idx, subject_id=topo_subject_id, epoch=epoch,
                        cmap=config.get('training_params', {}).get('visualize_params', {}).get('cmap', 'YlOrRd'),
                    )
                except Exception as e:
                    logger.warning(f"  Topomap viz failed (epoch {epoch}, subject={topo_subject_id}, trial_idx={topo_trial_idx}): {e}")
```
with:
```python
        if epoch == total_epochs if viz_every_n == 'last' else (viz_every_n > 0 and epoch % viz_every_n == 0):
            for topo_trial_idx, topo_subject_id in viz_targets:
                try:
                    bundle, _metrics = build_pretrain_bundle(
                        model, viz_dataset, topo_trial_idx, config, device,
                        subject_id=topo_subject_id, epoch=epoch)
                    panel_ctx = PanelContext(
                        config=config, output_dir=vis_dir, device=device, args=None,
                        model=model, dataset=viz_dataset, bundle=bundle,
                        cmap=config.get('training_params', {}).get('visualize_params', {}).get('cmap', 'YlOrRd'))
                    run_panels(['recon_signal', 'stamp_by_patch', 'stamp_gallery'], 'pretrain', panel_ctx)
                except Exception as e:
                    logger.warning(f"  Topomap viz failed (epoch {epoch}, subject={topo_subject_id}, trial_idx={topo_trial_idx}): {e}")
```
(This keeps the exact same try/except shape as today — the only thing inside it changes.
`device` must already be a variable in scope here — confirm by reading the file; it's used
earlier in `main()` to move the model/data to GPU, e.g. `model.to(device)` around where
`model.enter_tokenizer_phase()` is called.)

- [ ] **Step 6: Smoke — imports clean**

```bash
CUDA_VISIBLE_DEVICES='' /home/mamechin/anaconda3/envs/eeg_fm/bin/python -c "import train_pretrain; print('ok')"
```

- [ ] **Step 7: Smoke — a short real training run exercises the periodic snapshot**

Build a throwaway config in the scratchpad (NOT `config/config.json`) — copy
`config/config.json`, set `training_params.pretrain.epochs` to a small number (e.g. 3),
`training_params.pretrain.model_name` to a throwaway name (e.g.
`smoke/snapshot_migration_test`), and `training_params.visualize_params.pretrain.every_n_epochs`
to `1` (or whatever key `viz_every_n` reads — confirm the exact config key by reading
`train_pretrain.py`'s own parsing of `viz_targets`/`viz_every_n` before writing the
overlay) so the snapshot fires every epoch on a run short enough to finish in a couple of
minutes. Use a small `subject_to_use` list (1-2 subjects) on one small pretrain dataset.
Run it:
```bash
/home/mamechin/anaconda3/envs/eeg_fm/bin/python train_pretrain.py --config /tmp/<throwaway>.json
```
Expected: training completes all epochs without crashing, and
`output/smoke/snapshot_migration_test/visualization/` (or wherever `vis_dir` resolves for
this run) contains `recon_signal`/`stamp_by_patch`/`stamp_gallery`-shaped PNG files from
at least one epoch, with no `Topomap viz failed` warning in the log. Delete
`output/smoke/snapshot_migration_test/` afterward.

- [ ] **Step 8: `git diff --stat` sanity check**

```bash
git diff --stat train_pretrain.py
```
Expected: a small diff (import lines, the `checker` line removed, the stale-comment fix,
the one call-site block replaced) — not a rewrite.

- [ ] **Step 9: Commit**

```bash
git add train_pretrain.py
git commit -m "refactor: migrate train_pretrain.py's periodic snapshot to the new panels

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18"
```

---

## Task 6: Delete the retired class hierarchy and dead code

**Files:**
- Delete: `model/base_checker.py`
- Modify: `model/MeSAE/plugin.py`
- Modify: `model/base_plugin.py`
- Modify: `tools/viz/panels.py`

**Interfaces:** none produced — this is cleanup, nothing downstream depends on what's
deleted here (verified by Tasks 3-5 having already moved every real caller off it).

- [ ] **Step 1: Delete `model/base_checker.py`**

```bash
git rm model/base_checker.py
```

- [ ] **Step 2: Delete `MeSAEChecker` from `model/MeSAE/plugin.py`**

Read the file first. Delete the entire `MeSAEChecker` class (from `class MeSAEChecker(BaseEpochChecker):`
through the blank lines right before `class MeSAECodebookChecker(BaseCodebookChecker):`) —
this also removes `_run_reconstruction_sae` if it's now unused (it was already moved into
`tools/analysis/snapshot.py` as `_run_reconstruction` in Task 1 — confirm no other caller
in this file still references `_run_reconstruction_sae` before deleting it; `git grep -n
_run_reconstruction_sae` should show only its own definition and the deleted
`MeSAEChecker.run_reconstruction` override, both going away together).

Also delete the now-unused imports at the top of the file: `from model.base_checker import
BaseEpochChecker`, and any of `tools.viz.panels`/`tools.viz.codebook`/`tools.viz.extract`
names that were ONLY used by `MeSAEChecker` (check each import against what
`MeSAECodebookChecker` and `MeSAEPlotter`, which stay, still use — `git grep` each
imported name's other uses in this file before removing it from the import list; don't
remove a name still used elsewhere in the file).

Replace the module-bottom `BasePlugin(...)` call:
```python
PLUGIN = BasePlugin(
    build=build_model,
    finetune_cls=build_finetune,
    trainer_cls=MeSAETrainer,
    checker_cls=MeSAEChecker,
    plotter_cls=MeSAEPlotter,
    codebook_checker_cls=MeSAECodebookChecker,
```
with:
```python
PLUGIN = BasePlugin(
    build=build_model,
    finetune_cls=build_finetune,
    trainer_cls=MeSAETrainer,
    plotter_cls=MeSAEPlotter,
    codebook_checker_cls=MeSAECodebookChecker,
```
(Read the line after `codebook_checker_cls=MeSAECodebookChecker,` — it should be a closing
`)` — leave it as-is.)

- [ ] **Step 3: Remove `checker_cls` from `model/base_plugin.py`**

Read the file first. Replace:
```python
from model.base_trainer import BaseTrainer
from model.base_checker import BaseEpochChecker
from model.base_codebook_checker import BaseCodebookChecker
from model.base_plotter import BasePlotter


@dataclass(frozen=True)
class BasePlugin:
    """One instance per model_type, built at the bottom of model/<Name>/plugin.py and
    registered in model/factory.py's MODEL_REGISTRY. Bundles everything a new model needs
    to plug into the shared train_pretrain.py/train_finetune.py loops."""
    build: Callable            # (bp: dict, num_channels: int) -> nn.Module, the pretrain backbone
    finetune_cls: Callable     # (backbone, num_channels, num_classes, **finetune params) -> classifier
    trainer_cls: Type[BaseTrainer]
    checker_cls: Type[BaseEpochChecker]
    plotter_cls: Type[BasePlotter]
```
with:
```python
from model.base_trainer import BaseTrainer
from model.base_codebook_checker import BaseCodebookChecker
from model.base_plotter import BasePlotter


@dataclass(frozen=True)
class BasePlugin:
    """One instance per model_type, built at the bottom of model/<Name>/plugin.py and
    registered in model/factory.py's MODEL_REGISTRY. Bundles everything a new model needs
    to plug into the shared train_pretrain.py/train_finetune.py loops."""
    build: Callable            # (bp: dict, num_channels: int) -> nn.Module, the pretrain backbone
    finetune_cls: Callable     # (backbone, num_channels, num_classes, **finetune params) -> classifier
    trainer_cls: Type[BaseTrainer]
    plotter_cls: Type[BasePlotter]
```

- [ ] **Step 4: Delete the three dead functions from `tools/viz/panels.py`**

Read the file first. Delete `plot_topo_psd_filter` (from `def plot_topo_psd_filter(...)`
through the blank lines right before `def _log_pow(x):`), `plot_stamp_panel` (from `def
plot_stamp_panel(...)` through the blank lines right before `def plot_attn_topo(...)`),
and `plot_attn_topo` (from `def plot_attn_topo(...)` through the blank lines right before
`def plot_event_stamp_dynamics(...)`). Keep `_log_pow`, `_log_signed`, `plot_stamp_by_patch`,
`plot_stamp_gallery`, `plot_event_stamp_dynamics` — all still used (the first four by the
new panels via Task 2's imports, `plot_event_stamp_dynamics` by `MeSAECodebookChecker`,
sub-project C's concern, untouched here).

Update the module docstring (lines 1-7), which currently reads:
```python
"""
Shared epoch-snapshot panels (topo_psd_filter, attn_topo) reused by
BaseEpochChecker._render_snapshot (MeFSQ Experts or MeSAE stamps — viz/extract.py's
extract_head_*/extract_filter_* already return the shared PsdResult/SpectraResult
dataclasses). Keeps the two training-phase methods producing the exact same panel format
instead of drifting — see model/base_checker.py and docs/adr/0004-model-plugin-base-classes.md.
"""
```
Replace with:
```python
"""
Plotting functions for the stamp_by_patch/stamp_gallery panels (tools/panels/) and the
event_stamp_dynamics codebook diagnostic (model/MeSAE/plugin.py's MeSAECodebookChecker).
Pure rendering only -- calculation lives in tools/analysis/ and tools/viz/extract.py.
"""
```

- [ ] **Step 5: Smoke — full import sweep**

```bash
CUDA_VISIBLE_DEVICES='' /home/mamechin/anaconda3/envs/eeg_fm/bin/python -c "
import tools.analysis, tools.analysis.snapshot, tools.viz.panels, tools.viz.codebook, tools.viz.extract
import tools.panels, tools.panels.panel_profile, tools.panels.panel_recon_signal, tools.panels.panel_stamp_by_patch, tools.panels.panel_stamp_gallery
import analysis_pretrain, analysis_finetune, train_pretrain, train_finetune
import model.factory, model.MeSAE.plugin, model.base_plugin, model.base_codebook_checker
print('ok')
"
```
Expected: `ok`, no `ImportError`/`ModuleNotFoundError`/`AttributeError`.

- [ ] **Step 6: Confirm no live references to the deleted code remain**

```bash
git grep -n "BaseEpochChecker\|checker_cls\b\|has_attn_topo\|plot_attn_topo\|_render_stamp_panel\|plot_stamp_panel\|plot_topo_psd_filter\|_run_reconstruction_sae" -- '*.py'
```
Expected: no output (only `codebook_checker_cls` survives anywhere, a different,
unrelated field this task doesn't touch — the grep pattern above doesn't match it).

- [ ] **Step 7: Rerun every real-checkpoint smoke from Tasks 3-5 once more, end to end**

```bash
/home/mamechin/anaconda3/envs/eeg_fm/bin/python analysis_pretrain.py --config config/analysis.json \
  --checkpoint output/pretrain/mesae_v10_all_share/checkpoint/last.pth \
  --analysis snapshot --dataset BNCI2014001 --subject 1 --trial 0
/home/mamechin/anaconda3/envs/eeg_fm/bin/python analysis_finetune.py \
  --config /tmp/finetune_snapshot_overlay.json \
  --base-config output/baseline/BNCI2014001_intra_raw_band/artifacts/config.json \
  --dataset BNCI2014001
```
Expected: both exit 0 with the same output shape as Tasks 3/4's own smoke tests — proves
deleting the old class hierarchy didn't silently break the new panels (e.g. via a stale
import that only Task 6's own files happened to still provide).

**Known, disclosed behavior change from this migration (not a regression to fix, just to
verify doesn't surprise you):** in `analysis_pretrain.py`/`analysis_finetune.py`, a
`stamp_by_patch`/`stamp_gallery` extraction failure now raises normally instead of being
caught and logged (today's `_render_snapshot`'s `try/except` around `extract_psd` is
gone, replaced by each panel independently calling its own extraction function with no
wrapping try). `recon_signal` was never protected either way. `train_pretrain.py`'s
resilience is unchanged (its own `try/except` still wraps the whole call). If either CLI
smoke command above raises where the baseline didn't, that's a real regression — but a
clean run proves this doesn't bite for the checkpoint/trial this plan's smoke tests use.

- [ ] **Step 8: `git diff --stat` sanity check**

```bash
git diff --stat --cached
```
Expected: `model/base_checker.py` deleted, `model/MeSAE/plugin.py` shows a substantial
deletion (the whole `MeSAEChecker` class), `model/base_plugin.py` and `tools/viz/panels.py`
show small targeted edits.

- [ ] **Step 9: Commit**

```bash
git add -A model tools/viz/panels.py
git commit -m "refactor: delete retired BaseEpochChecker/MeSAEChecker and dead attn_topo/stamp_panel code

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18"
```
