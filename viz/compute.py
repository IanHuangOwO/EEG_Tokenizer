"""
Model-forward utilities: patch reconstruction pipeline.
"""

import torch
import numpy as np


@torch.no_grad()
def get_detailed_outputs(model, x, coords, time_idx=None):
    B, C, N, L = x.shape
    out = model(x, coords=coords, time_idx=time_idx, bool_masked_pos=None)
    return out.recon.reshape(B, C, N * L)


@torch.no_grad()
def _run_reconstruction(model, dataset, trial_idx, device):
    x_patches, coords, mask, time_indices, _, _ = dataset[trial_idx]
    C, N, L = x_patches.shape
    T_total = N * L
    fs = dataset.base_dataset.config['preprocess_params']['target_freq']

    x_in      = x_patches.unsqueeze(0).to(device)
    coords_in = coords.unsqueeze(0).to(device)
    t_in      = time_indices.unsqueeze(0).to(device)

    recon_flat = get_detailed_outputs(model, x_in, coords_in, t_in)

    raw   = x_patches.reshape(C, T_total).cpu().numpy()
    recon = recon_flat[0].cpu().numpy()

    return {
        'raw':    raw,
        'recon':  recon,
        'coords': coords.numpy(),
        'T': T_total, 'N': N, 'L': L, 'fs': fs,
    }
