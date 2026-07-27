"""
Finetune-stage epoch snapshot — same three-panel format as check_epoch_pretrain.py, and
same MeFSQ/MeSAE dispatch (inferred from model.backbone):
  recon_signal     raw vs backbone-recon time series (model.backbone, unmasked)
  topo_psd_filter  raw / full-recon / per-unit (Expert or Filter) topo + PSD grid, sorted
                    by contribution (model.backbone, same extract_* functions as pretrain)
  attn_topo        per-unit channel-attention topography, from the finetune head's own
                    per-channel-over-unit attention (PerChannelHeadAttn's attn_h),
                    transposed to [Unit, Channel] so it plots through the exact same
                    shared code as the pretrain script's own per-unit channel attention —
                    a different attention (the classifier's, not the tokenizer's), same
                    shape/visual language, so the two training-phase scripts stay in sync
                    instead of drifting into separate formats.
"""
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch

from viz.draw import (
    project_coords_2d,
    extract_head_psd, extract_head_spectra,
    extract_filter_psd, extract_filter_spectra,
    plot_topo_psd_filter, plot_attn_topo,
)
from viz.train import visualize_reconstruction


def _patchify(x, patch_len):
    """x: [C, T] -> x_patches [1, C, P, L], time_idx [1, P]."""
    C, T = x.shape
    P = T // patch_len
    x_patches = x[:, :P * patch_len].reshape(C, P, patch_len).unsqueeze(0)
    time_idx = torch.arange(P, dtype=torch.long).unsqueeze(0)
    return x_patches, time_idx


def _build_pad_mask(valid_channels, valid_length, P, patch_len):
    """Mirrors FinetuneCollate: a patch counts valid only if fully inside the real
    (non-padded) length. Returns [1, C, P] bool."""
    valid_length = valid_length.item() if torch.is_tensor(valid_length) else valid_length
    n_valid_patches = min(valid_length // patch_len, P)
    patch_valid = torch.arange(P) < n_valid_patches                    # [P]
    return (valid_channels.unsqueeze(-1) & patch_valid.unsqueeze(0)).unsqueeze(0)  # [1, C, P]


@torch.no_grad()
def run(config, output_dir, model, dataset, trial_idx, subject_id=None, epoch=None,
        patch_len=None, cmap='YlOrRd'):
    """
    model:   the MeFSQFinetune instance — model.backbone reused for recon_signal/topo_psd_filter,
             model(...) itself (the full finetune forward, PerChannelHeadAttn included) for attn_topo.
    dataset: a FinetuneDataset (yields raw [C,T] trials, not pre-patched).
    """
    backbone = model.backbone
    model_type = 'MeFSQ' if hasattr(backbone, 'n_routed_experts') else 'MeSAE'
    unit_label = 'Expert' if model_type == 'MeFSQ' else 'Filter'
    device = next(model.parameters()).device
    pp = config.get('preprocess_params', {})
    patch_len = patch_len or pp.get('patch_length', 100)
    epoch_tag = f'_ep{epoch:04d}' if epoch is not None else ''

    x_raw, coords, label, valid_channels, valid_length = dataset[trial_idx]
    x_patches, time_idx = _patchify(x_raw, patch_len)
    x_in = x_patches.to(device)
    c_in = coords.unsqueeze(0).to(device)
    t_in = time_idx.to(device)
    vc_in = valid_channels.unsqueeze(0).to(device)
    pad_mask = _build_pad_mask(valid_channels, valid_length, x_patches.shape[2], patch_len).to(device)

    pos2d = project_coords_2d(coords.numpy())
    channel_names = dataset.base_dataset.channel_names
    C, N, L = x_patches.shape[1], x_patches.shape[2], patch_len

    was_training = model.training
    model.eval()

    viz_dir = os.path.join(output_dir, 'recon')
    os.makedirs(viz_dir, exist_ok=True)

    # finetune head's own attention (classifier's per-channel distribution over units) —
    # used to rank AND plot attn_topo below.
    _, attn_h, attn_n, attn_c = model(x_in, c_in, time_idx=t_in, pad_mask=pad_mask)
    attn_np = attn_h[0].detach().cpu().numpy()  # [C, H_total]
    importance = attn_np.sum(axis=0)            # [H_total] — total attention each unit received

    out = backbone(x_in, c_in, time_idx=t_in, bool_masked_pos=None, valid_channels=vc_in)
    raw_cnl   = x_patches[0].numpy()                                   # [C, N, L]
    recon_cnl = out.recon[0].reshape(C, N, L).detach().cpu().numpy()   # [C, N, L]

    # ── recon_signal ─────────────────────────────────────────────────────────
    raw_t   = torch.from_numpy(raw_cnl.reshape(1, C, N * L))
    recon_t = torch.from_numpy(recon_cnl.reshape(1, C, N * L))
    visualize_reconstruction(
        None, (raw_t, recon_t), epoch,
        output_dir=viz_dir,
        channel_names=channel_names,
        subject_id=subject_id, trial_idx=trial_idx,
        mask=None, patch_len=patch_len,  # FinetuneDataset trials aren't masked
    )

    # ── topo_psd_filter ──────────────────────────────────────────────────────
    try:
        if model_type == 'MeFSQ':
            psd_list, _, _, _ = extract_head_psd(backbone, x_in, c_in, t_in)
        else:
            psd_list, _, _, _ = extract_filter_psd(backbone, x_in, c_in, t_in, vc_in)
        psd_ch_x = psd_list[0]  # [C, H_total]

        fs = pp.get('target_freq')
        l_freq, h_freq = pp.get('l_freq'), pp.get('h_freq')
        if model_type == 'MeFSQ':
            psd_x, freqs, _ = extract_head_spectra(backbone, x_in, c_in, t_in, fs=fs, freq_resolution=0.2)
        else:
            psd_x, freqs, _ = extract_filter_spectra(backbone, x_in, c_in, t_in, vc_in, fs=fs, freq_resolution=0.2)

        n_fft = max(L, int(round(fs / 0.2))) if fs else L
        fft_raw   = np.fft.rfft(raw_cnl,   n=n_fft, axis=-1)
        fft_recon = np.fft.rfft(recon_cnl, n=n_fft, axis=-1)
        psd_raw   = (fft_raw.real**2   + fft_raw.imag**2  ).mean(axis=1)  # [C, F]
        psd_recon = (fft_recon.real**2 + fft_recon.imag**2).mean(axis=1)  # [C, F]

        if l_freq is not None and h_freq is not None:
            band = (freqs >= l_freq) & (freqs <= h_freq)
            freqs     = freqs[band]
            psd_x     = psd_x[:, :, band]
            psd_raw   = psd_raw[:, band]
            psd_recon = psd_recon[:, band]

        raw_power   = (raw_cnl   ** 2).mean(axis=(1, 2))  # [C]
        recon_power = (recon_cnl ** 2).mean(axis=(1, 2))  # [C]

        out_path = os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_topo_psd_filter.png")
        plot_topo_psd_filter(
            out_path, pos2d, raw_power, recon_power, psd_raw, psd_recon,
            psd_ch_x, psd_x, freqs, importance, cmap=cmap,
            subject_id=subject_id, trial_idx=trial_idx, epoch_tag=f"{epoch_tag} [finetune]",
            unit_label=unit_label, l_freq=l_freq, h_freq=h_freq,
        )
        print(f"  [epoch] -> {out_path}")
    except Exception as e:
        print(f"  [epoch] topo_psd_filter failed: {e}")

    # ── attn_topo ────────────────────────────────────────────────────────────
    try:
        attn = attn_np.T  # [H_total, C] — classifier's channel weight per Expert
        out_path = os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_attn_topo.png")
        plot_attn_topo(
            out_path, pos2d, attn, importance, channel_names,
            valid_channels=valid_channels.numpy(),
            subject_id=subject_id, trial_idx=trial_idx, epoch_tag=f"{epoch_tag} [finetune]",
            unit_label=unit_label,
        )
        print(f"  [epoch] -> {out_path}")
    except Exception as e:
        print(f"  [epoch] attn_topo failed: {e}")

    model.train(was_training)
