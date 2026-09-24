"""
Shared orchestration helpers for the analysis scripts (analysis_pretrain.py,
analysis_finetune.py): config loading/merging, model loading, dataset/trial/subject
selection, output-dir resolution. Nothing here renders a plot — see viz/ for that.
"""

import os
import json
import copy
import random
import numpy as np
import torch


def lookup_event_onset_sample(config: dict, ds_name: str):
    """This dataset's own datas/<ds_name>/metadata.json data_metadata.event_onset_sample
    (an int sample count, or None if absent) -- a per-dataset property of which loader
    pre-event shift was applied, not a per-run setting, so it lives in the dataset's own
    metadata rather than being copy-pasted into every config that references it (see
    that file's sibling event_onset_sample_note for the per-dataset rationale). Finds
    ds_name's dataset_path by searching every dataset_params.<mode> entry (pretrain,
    finetune, ...) for a matching key. Returns None if the dataset isn't in config at
    all, or its metadata.json has no event_onset_sample -- `is not None`, not truthiness,
    at every call site: an onset of literal 0 is a real, legitimate value, not "not
    configured"."""
    dataset_path = None
    for mode_params in config.get('dataset_params', {}).values():
        if ds_name in mode_params:
            dataset_path = mode_params[ds_name].get('dataset_path')
            break
    if dataset_path is None:
        return None
    meta_path = os.path.join(dataset_path, 'metadata.json')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    dm = meta.get('data_metadata', {})
    if dm.get('event_onset_seconds') is not None:
        # MOABB-backed datasets store seconds (rate-independent), see IO/loader.py's
        # write_moabb_metadata; convert with this run's sample rate.
        fs = config.get('preprocess_params', {}).get('sample_freq', 200)
        return int(round(dm['event_onset_seconds'] * fs))
    return dm.get('event_onset_sample')


def event_onset_patch(config: dict, ds_name: str):
    """Event onset on a patch-index x-axis (patch n covers samples
    [n*stride, n*stride+patch_len), plotted at its centre): (onset - patch_len/2) / stride,
    or None when the dataset has no event (continuous windows)."""
    onset = lookup_event_onset_sample(config, ds_name)
    if onset is None:
        return None
    pp = config.get('preprocess_params', {})
    L = pp.get('patch_length', 50)
    stride = pp.get('patch_stride', L)
    return (onset - L / 2) / stride


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_config(path: str) -> dict:
    """Load config. If it contains 'base_config', deep-merge on top of it."""
    with open(path, 'r') as f:
        cfg = json.load(f)
    if 'base_config' in cfg:
        base_path = cfg.pop('base_config')
        with open(base_path, 'r') as f:
            base = json.load(f)
        cfg = _deep_merge(base, cfg)
    return cfg


def resolve_output_path(config: dict, mode: str = 'pretrain') -> str:
    """Return this run's path segment under output/ -- training_params.<mode>.output_path
    if set, else '<model_name>/pretrain' for a pretrain run (pretrain artifacts always
    nest under pretrain/, so a backbone's later finetune/ runs never share a level with
    them) or plain model_name for a finetune run. output_path is separate from
    model_name (a clean identity string, e.g. logged in train_pretrain.py) so a path can
    carry a subfolder without model_name itself doing double duty as a path."""
    tp = config['training_params'][mode]
    if 'output_path' in tp:
        return tp['output_path']
    return f"{tp['model_name']}/pretrain" if mode == 'pretrain' else tp['model_name']


def resolve_output_dir(config: dict, *sub_dirs: str, mode: str = 'pretrain') -> str:
    """Return output/{output_path}/{sub_dirs...} and create it."""
    path = os.path.join('output', resolve_output_path(config, mode=mode), *sub_dirs)
    os.makedirs(path, exist_ok=True)
    return path


def resolve_finetune_analysis_dir(config: dict, dataset_name: str) -> str:
    """Return analysis_finetune.py's output dir for one dataset and create it.

    For a baseline-matrix run (training_params.finetune.output_path ==
    '<backbone>/finetune/<head>/<dataset>_<mode>', see ADR 0017) this is
    output/<backbone>/finetune/analysis/<head>/<dataset>_<mode>/<dataset_name>/ -- one
    shared analysis/ root directly under finetune/ (sibling to every head's own run dirs),
    instead of nested inside each individual run, so every run's snapshots land in one
    place. Falls back to resolve_output_dir's plain output/<output_path>/analysis/
    <dataset_name>/ for any output_path that isn't shaped like a baseline-matrix run (no
    literal '/finetune/' marker -- e.g. an ad-hoc one-off run)."""
    output_path = resolve_output_path(config, mode='finetune')
    marker = '/finetune/'
    if marker in output_path:
        backbone, rest = output_path.split(marker, 1)   # rest = '<head>/<dataset>_<mode>'
        path = os.path.join('output', backbone, 'finetune', 'analysis', rest, dataset_name)
    else:
        path = os.path.join('output', output_path, 'analysis', dataset_name)
    os.makedirs(path, exist_ok=True)
    return path


def select_subject_dataset(config: dict, subject=None, dataset_name: str = None, mode: str = 'pretrain'):
    """
    Resolve (dataset_name, subject_id) using:
      1. CLI arguments (highest priority)
      2. The dataset_params entry that has trial_to_use set
      3. First key of dataset_params as last resort
    """
    ds_params = config['dataset_params'][mode]
    if dataset_name is None:
        for ds, ds_cfg in ds_params.items():
            if 'trial_to_use' in ds_cfg:
                dataset_name = ds
                break
        if dataset_name is None:
            dataset_name = next(iter(ds_params))
    if subject is None:
        ds_cfg = ds_params[dataset_name]
        subs = ds_cfg.get('subject_to_use', [])
        subject = subs[0] if subs and subs[0] != 'all' else None
    if subject is None:
        meta_path = os.path.join(ds_params[dataset_name]['dataset_path'], 'metadata.json')
        with open(meta_path, 'r') as f:
            all_subs = list(json.load(f).get('data_structure', {}).keys())
        subject = int(all_subs[0]) if str(all_subs[0]).isdigit() else random.choice(all_subs)
    return dataset_name, subject


def filter_config_to_subject(config: dict, dataset_name: str, subject, mode: str = 'pretrain') -> dict:
    """Return config copy with dataset_params[mode] reduced to one dataset + subject."""
    cfg = copy.deepcopy(config)
    orig = cfg['dataset_params'][mode][dataset_name]
    cfg['dataset_params'][mode] = {
        dataset_name: {**orig, 'subject_to_use': [subject]}
    }
    return cfg


def load_model(config: dict, checkpoint: str, device: torch.device, mode: str = 'pretrain'):
    """Build model from config, load checkpoint with strict=False, return eval.
    mode='finetune' rebuilds the full FinetuneModel (frozen backbone + head) from the head
    checkpoint's own head_config and backbone_checkpoint path (no shape inference; a missing
    file raises)."""
    if mode == 'finetune':
        from model.factory import load_finetune_checkpoint
        return load_finetune_checkpoint(config, checkpoint, device)

    from model.factory import build_pretrain_from_config

    model = build_pretrain_from_config(config, mode=mode).to(device)

    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [ckpt] {len(missing)} missing keys (fresh init), e.g. {missing[0]}")
        if unexpected:
            print(f"  [ckpt] {len(unexpected)} unexpected keys, e.g. {unexpected[0]}")
        # phase flags (spatial/temporal/active blocks) restored by MeSAE's load post-hook
    else:
        print(f"  WARNING: checkpoint not found at {checkpoint!r}, using random weights.")

    model.eval()
    return model


def pick_trial(dataset, subject_id, trial: int = None, dataset_name: str = None):
    """
    Return (trial_idx, subject_id). Picks randomly if trial is None.
    dataset_name: disambiguates subject_id across multi-dataset training runs (see
    build_dataset_from_config / dataset_params.pretrain having multiple entries) —
    subject IDs are per-source-dataset, not globally unique (e.g. subject "1" can exist
    in both BETA_3s and BCICIV1_Train), so without this a collision silently picks
    whichever dataset's matching subject happened first in the concatenated tensor.
    """
    sub_data = dataset.base_dataset.subject_data
    try:
        sid = type(sub_data[0].item())(subject_id)
        indices = (sub_data == sid).nonzero(as_tuple=True)[0]
        if dataset_name is not None and indices.numel() > 0:
            names = dataset.base_dataset.dataset_names
            keep = torch.tensor([names[i] == dataset_name for i in indices.tolist()], dtype=torch.bool)
            indices = indices[keep]
    except Exception:
        indices = torch.arange(len(dataset))

    if len(indices) == 0:
        raise ValueError(f"No trials found for subject {subject_id!r}" + (f", dataset {dataset_name!r}" if dataset_name else ""))

    idx = indices[trial % len(indices)].item() if trial is not None else random.choice(indices).item()
    actual_subject = dataset.base_dataset.subject_data[idx].item()
    return idx, actual_subject


def setup_mne_info(dataset, fs=200.0):
    """Build MNE Info from dataset electrode coordinates."""
    import mne

    base   = dataset.base_dataset if hasattr(dataset, 'base_dataset') else dataset
    coords = np.array(base.coords, dtype=float).copy()
    names  = base.channel_names

    radii = np.sqrt(np.sum(coords[:, :2] ** 2, axis=1))
    max_r = radii.max()
    if max_r > 0:
        coords *= 0.06 / max_r

    try:
        std = mne.channels.make_standard_montage('standard_1020')
        std_pos = std.get_positions()['ch_pos']
        std_keys = {k.upper(): k for k in std_pos}
    except Exception:
        std_pos, std_keys = {}, {}

    montage_pos, valid_names = {}, []
    for i, name in enumerate(names):
        upper = name.upper()
        if upper in std_keys:
            montage_pos[name] = std_pos[std_keys[upper]]
            valid_names.append(name)
        elif np.any(coords[i] != 0):
            montage_pos[name] = coords[i]
            valid_names.append(name)

    info = mne.create_info(ch_names=valid_names, sfreq=fs, ch_types='eeg')
    try:
        dig = mne.channels.make_dig_montage(ch_pos=montage_pos, coord_frame='head')
        info.set_montage(dig)
    except Exception as e:
        print(f"[setup_mne_info] montage warning: {e}")
    return info
