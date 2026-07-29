"""
Shared infrastructure for the viz package.

Every viz module imports from here instead of duplicating
dataset filtering, model loading, and trial selection logic.
"""

import os
import json
import copy
import random
import numpy as np
import torch


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


def resolve_output_dir(config: dict, *sub_dirs: str, mode: str = 'pretrain') -> str:
    """Return output/{model_name}/{sub_dirs...} and create it."""
    model_name = config['training_params'][mode]['model_name']
    path = os.path.join('output', model_name, *sub_dirs)
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
    """Build model from config, load checkpoint with strict=False, return eval."""
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
        # spatial_active/temporal_active are plain attributes, not restored by load_state_dict
        model.enable_spatial()
        if hasattr(model, 'enable_temporal'):
            model.enable_temporal()
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
