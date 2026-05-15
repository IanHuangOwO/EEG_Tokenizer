"""
Shared utilities for the analysis package.

Every analysis module imports from here instead of duplicating
dataset filtering, model loading, and trial selection logic.
"""

import os
import json
import copy
import random
import torch


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base. override wins on scalar conflicts."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_config(path: str) -> dict:
    """
    Load analysis config. If it contains 'base_config', that file is loaded
    first and the analysis config is deep-merged on top of it.
    """
    with open(path, 'r') as f:
        cfg = json.load(f)
    if 'base_config' in cfg:
        base_path = cfg.pop('base_config')
        with open(base_path, 'r') as f:
            base = json.load(f)
        cfg = _deep_merge(base, cfg)
    return cfg


def resolve_output_dir(config: dict, *sub_dirs: str) -> str:
    """
    Return output/{model_name}/visualization/{sub_dirs...} and create it.
    """
    model_name = config['training_params']['model_name']
    path = os.path.join('output', model_name, 'visualization', *sub_dirs)
    os.makedirs(path, exist_ok=True)
    return path


def select_subject_dataset(config: dict, subject=None, dataset_name: str = None):
    """
    Resolve (dataset_name, subject_id) using:
      1. CLI arguments (highest priority)
      2. The dataset_params entry that has trial_to_use set (the explicitly selected one)
      3. First key of dataset_params as last resort

    Returns (dataset_name, subject_id).
    """
    if dataset_name is None:
        # Prefer the dataset that has trial_to_use explicitly set
        for ds, ds_cfg in config['dataset_params'].items():
            if 'trial_to_use' in ds_cfg:
                dataset_name = ds
                break
        if dataset_name is None:
            dataset_name = next(iter(config['dataset_params']))
    if subject is None:
        ds_cfg = config['dataset_params'][dataset_name]
        subs = ds_cfg.get('subject_to_use', [])
        subject = subs[0] if subs and subs[0] != 'all' else None
    if subject is None:
        meta_path = os.path.join(config['dataset_params'][dataset_name]['dataset_path'], 'metadata.json')
        with open(meta_path, 'r') as f:
            all_subs = list(json.load(f).get('data_structure', {}).keys())
        subject = int(all_subs[0]) if str(all_subs[0]).isdigit() else random.choice(all_subs)
    return dataset_name, subject


def filter_config_to_subject(config: dict, dataset_name: str, subject) -> dict:
    """
    Return a config copy with dataset_params reduced to exactly one
    dataset + subject, so build_dataset_from_config loads only that data.
    """
    cfg = copy.deepcopy(config)
    orig = cfg['dataset_params'][dataset_name]
    cfg['dataset_params'] = {
        dataset_name: {**orig, 'subject_to_use': [subject]}
    }
    return cfg


def load_model(config: dict, checkpoint: str, device: torch.device):
    """
    Build model from config, load checkpoint with strict=False and
    consistent warnings, return in eval mode.
    """
    from model.factory import build_model_from_config

    # Infer n_fft from checkpoint's freq_mask shape, fall back to config arithmetic
    n_fft = None
    if checkpoint and os.path.exists(checkpoint):
        sd = torch.load(checkpoint, map_location='cpu', weights_only=False)
        sd = sd.get('model_state_dict', sd)
        if 'freq_mask_0' in sd:
            n_fft = (sd['freq_mask_0'].shape[0] - 1) * 2

    if n_fft is None:
        pp = config['model_params']['AttnVQ']['preprocess']
        n_fft = (pp['trial_length'] // pp['patch_length']) * pp['patch_length']

    model = build_model_from_config(config, n_fft_trial=n_fft).to(device)

    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [ckpt] {len(missing)} missing keys (fresh init), e.g. {missing[0]}")
        if unexpected:
            print(f"  [ckpt] {len(unexpected)} unexpected keys, e.g. {unexpected[0]}")
    else:
        print(f"  WARNING: checkpoint not found at {checkpoint!r}, using random weights.")

    model.eval()
    return model


def pick_trial(dataset, subject_id, trial: int = None):
    """
    Return (trial_idx, subject_id) for the given subject.
    If trial is None, picks randomly. Handles int/str subject_id mismatch.
    """
    sub_data = dataset.base_dataset.subject_data
    try:
        # Normalise subject_id to the same type stored in the tensor
        sid = type(sub_data[0].item())(subject_id)
        indices = (sub_data == sid).nonzero(as_tuple=True)[0]
    except Exception:
        indices = torch.arange(len(dataset))

    if len(indices) == 0:
        raise ValueError(f"No trials found for subject {subject_id!r}")

    idx = indices[trial % len(indices)].item() if trial is not None else random.choice(indices).item()
    actual_subject = dataset.base_dataset.subject_data[idx].item()
    return idx, actual_subject
