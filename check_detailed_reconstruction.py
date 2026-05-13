import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import math


from IO.dataset import build_dataset_from_config
from model.factory import build_model_from_config

def get_psd(sig, fs):
    from scipy.signal import welch
    f, p = welch(sig, fs=fs, nperseg=min(len(sig), 256))
    return f, p

def spectrum_to_time(real, imag, mask, n_fft, n_samples):
    """Converts a partial spectrum (masked) to time domain."""
    B_total = real.shape[0]
    full_fft = torch.zeros((B_total, n_fft // 2 + 1), dtype=torch.complex64, device=real.device)
    full_fft[:, mask] = torch.complex(real.float(), imag.float())
    recon = torch.fft.irfft(full_fft, n=n_fft, dim=-1, norm='ortho')
    return recon[:, :n_samples]

@torch.no_grad()
def get_detailed_outputs(model, x, coords, time_idx):
    """
    Extracts Stage-wise and Head-wise temporal reconstructions.
    x: [B, C, N, L]
    """
    B, C, N, L = x.shape
    T_total = N * L
    
    # 1. Forward through embedding and encoder
    z = model.embed(x, coords=coords, time_idx=time_idx)
    
    needed_stages = {h_cfg["stage_idx"] for h_cfg in model.decoder_heads_config}
    stage_outputs = {}
    for i, block in enumerate(model.encoder.blocks):
        z = block(z)
        if i in needed_stages:
            stage_outputs[i] = z

    stage_recons = [] 
    stage_heads_recons = [] 
    stage_names = []

    for i, h_cfg in enumerate(model.decoder_heads_config):
        z_stage = stage_outputs[h_cfg["stage_idx"]]
        B_s, C_s, N_s, D_s = z_stage.shape
        z_flat = z_stage.reshape(B_s * C_s, N_s, D_s)
        
        vq_head = model.vq_heads[i]
        decoder = model.decoders[i]
        
        v_q, _, _, _ = vq_head(z_flat)
        D_dim = vq_head.A.shape[0]
        H, r = vq_head.num_heads, vq_head.r
        M = torch.einsum('dhr,hdf->hrf', vq_head.A.view(D_dim, H, r), decoder.W)
        h_out = torch.einsum('bnhr,hrf->bnhf', v_q, M) / math.sqrt(H)
        
        h_real, h_imag = h_out.chunk(2, dim=-1)
        mask = getattr(model, f'freq_mask_{i}')
        
        s_real = h_real.sum(dim=(1, 2))
        s_imag = h_imag.sum(dim=(1, 2))
        b_real, b_imag = decoder.bias.chunk(2, dim=-1)
        s_real = s_real + b_real
        s_imag = s_imag + b_imag
        
        s_recon = spectrum_to_time(s_real, s_imag, mask, model.n_fft, T_total)
        stage_recons.append(s_recon.reshape(B, C, T_total))
        
        h_real_sum = h_real.sum(dim=1)
        h_imag_sum = h_imag.sum(dim=1)
        heads_recon_list = []
        for h in range(H):
            hr = spectrum_to_time(h_real_sum[:, h], h_imag_sum[:, h], mask, model.n_fft, T_total)
            heads_recon_list.append(hr)
        heads_recon = torch.stack(heads_recon_list, dim=1)
        stage_heads_recons.append(heads_recon.reshape(B, C, H, T_total))
        
        f_min, f_max = h_cfg["freq_range"]
        stage_names.append(f"Stage {i} ({f_min}-{f_max}Hz)")

    return stage_recons, stage_heads_recons, stage_names

def main():
    parser = argparse.ArgumentParser(description='Detailed EEG Reconstruction Analysis')
    parser.add_argument('--config', type=str, default='config/config.json')
    parser.add_argument('--checkpoint', type=str, default='output/attnvq_v06/tokenizer/best_tokenizer.pth')
    parser.add_argument('--output_dir', type=str, default=None) # Default to None to trigger auto-derivation
    parser.add_argument('--subject', type=int, default=None)
    parser.add_argument('--trial', type=int, default=None)
    parser.add_argument('--channel', type=str, default=None)
    args = parser.parse_args()

    # --- Auto-derive Output Directory from Checkpoint ---
    if args.output_dir is None:
        # Expected path: output/MODEL_NAME/tokenizer/best_tokenizer.pth
        # Target path:   output/MODEL_NAME/visualization/detailed_recon
        ckpt_parts = os.path.normpath(args.checkpoint).split(os.sep)
        if 'output' in ckpt_parts:
            out_idx = ckpt_parts.index('output')
            if len(ckpt_parts) > out_idx + 1:
                model_name = ckpt_parts[out_idx + 1]
                args.output_dir = os.path.join('output', model_name, 'visualization', 'detailed_recon')
        
        if args.output_dir is None:
            args.output_dir = 'output/visualization/detailed_recon'

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Pick one dataset at random if no subject specified, then zero out all others
    ds_names = list(config['dataset_params'].keys())
    chosen_ds = random.choice(ds_names) if args.subject is None else None

    for ds_name, ds_args in config['dataset_params'].items():
        meta_path = os.path.join(ds_args['dataset_path'], 'metadata.json')
        with open(meta_path, 'r') as f_meta:
            meta = json.load(f_meta)
        all_subs = list(meta.get('data_structure', {}).keys())

        if args.subject is not None:
            if str(args.subject) in all_subs or args.subject in all_subs:
                config['dataset_params'][ds_name]['subject_to_use'] = [args.subject]
            else:
                config['dataset_params'][ds_name]['subject_to_use'] = []
        else:
            if ds_name == chosen_ds:
                random_sub = random.choice(all_subs)
                config['dataset_params'][ds_name]['subject_to_use'] = [random_sub]
                args.subject = int(random_sub) if str(random_sub).isdigit() else random_sub
            else:
                config['dataset_params'][ds_name]['subject_to_use'] = []

    dataset = build_dataset_from_config(config, mode='tokenizer')
    if len(dataset) == 0: return
    fs = dataset.base_dataset.config['model_params']['AttnVQ']['preprocess']['target_freq']
    n_fft = dataset.base_dataset.fft_params['n_fft']
    model = build_model_from_config(config, n_fft_trial=n_fft).to(device)
    
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    
    model.eval()
    sub_indices = (dataset.base_dataset.subject_data == args.subject).nonzero(as_tuple=True)[0]
    if len(sub_indices) == 0: return
    trial_idx = sub_indices[args.trial % len(sub_indices)].item() if args.trial is not None else random.choice(sub_indices).item()

    x_patches, coords, time_indices, label, _ = dataset[trial_idx]
    C, N, L = x_patches.shape
    T_total = N * L
    ch_idx = dataset.base_dataset.channel_names.index(args.channel) if args.channel in dataset.base_dataset.channel_names else 0
    args.channel = dataset.base_dataset.channel_names[ch_idx]

    print(f"Analyzing Subject {args.subject}, Trial {trial_idx}, Channel {args.channel}")

    x_in = x_patches.unsqueeze(0).to(device)
    coords_in = coords.unsqueeze(0).to(device)
    time_idx_in = time_indices.unsqueeze(0).to(device)
    
    stage_recons, stage_heads_recons, stage_names = get_detailed_outputs(model, x_in, coords_in, time_idx_in)
    num_stages = len(stage_recons)
    full_recon_ch = torch.stack(stage_recons).sum(dim=0)[0, ch_idx].cpu().numpy()
    raw_ch = x_patches[ch_idx].reshape(-1).cpu().numpy()
    stage_recons_ch = [sr[0, ch_idx].cpu().numpy() for sr in stage_recons]
    time_vec = np.arange(T_total) / fs

    # === PRE-COMPUTE BAND-SPECIFIC RAW SIGNALS ===
    raw_fft_full = torch.fft.rfft(torch.from_numpy(raw_ch).to(device), n=model.n_fft, norm='ortho')
    raw_bands_ch = []
    for i in range(num_stages):
        mask_i = getattr(model, f'freq_mask_{i}')
        raw_fft_i = torch.zeros_like(raw_fft_full); raw_fft_i[mask_i] = raw_fft_full[mask_i]
        raw_band_i = torch.fft.irfft(raw_fft_i, n=model.n_fft, norm='ortho')[:T_total].cpu().numpy()
        raw_bands_ch.append(raw_band_i)

    # Combined Mask and Freqs
    full_mask = torch.zeros(model.n_fft // 2 + 1, dtype=torch.bool, device=device)
    for i in range(num_stages): full_mask |= getattr(model, f'freq_mask_{i}')
    full_mask_cpu = full_mask.cpu().numpy()
    freqs = torch.fft.rfftfreq(model.n_fft, d=1.0/fs).cpu().numpy()
    freqs_act = freqs[full_mask_cpu]
    extent = [freqs_act[0], freqs_act[-1], 0, 1]

    # --- Unified Super-Dashboard Layout ---
    num_p_show = max(2, int(N * 0.2))
    p_indices = np.linspace(0, N - 1, num_p_show, dtype=int)
    
    fig_final = plt.figure(figsize=(12 + 4 * num_p_show, 28), constrained_layout=True)
    gs_master = fig_final.add_gridspec(1, 1 + num_p_show, width_ratios=[3] + [1.0]*num_p_show, wspace=0.03)
    
    def plot_analysis_column(gs_slot, raw_sig, recon_sig, band_raws, band_recons, title_prefix, is_trial=False, 
                             vmins_m=None, vmaxs_m=None, vmins_p=None, vmaxs_p=None):
        h_ratios = [1.2]*(num_stages+1) + [0.8]*3 + [0.8]*3
        gs_col = gs_slot.subgridspec(num_stages + 1 + 3 + 3, 1, height_ratios=h_ratios, hspace=0.05)
        
        # 1. Temporal Section
        for i in range(num_stages + 1):
            ax = fig_final.add_subplot(gs_col[i, 0])
            if i == 0:
                ax.plot(raw_sig, color='black', alpha=0.2, linewidth=0.8)
                ax.plot(recon_sig, color='blue', alpha=0.7, linewidth=1.2, linestyle=':')
                ax.set_title(f"{title_prefix} FULL: RAW vs TOTAL", fontsize=10, fontweight='bold')
            else:
                s_idx = i - 1
                ax.plot(band_raws[s_idx], color='black', alpha=0.3, linewidth=0.8)
                ax.plot(band_recons[s_idx], color='C'+str(s_idx), alpha=0.8, linewidth=1.2, linestyle=':')
                ax.set_title(f"{title_prefix}: {stage_names[s_idx].split(' ')[0]} vs Raw content", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(True, alpha=0.05)

        # FFT Analysis
        sig_fft = torch.fft.rfft(torch.from_numpy(raw_sig).to(device), n=model.n_fft, norm='ortho')
        rec_fft = torch.fft.rfft(torch.from_numpy(recon_sig).to(device), n=model.n_fft, norm='ortho')
        raw_mag = torch.abs(sig_fft).cpu().numpy()[full_mask_cpu]
        rec_mag = torch.abs(rec_fft).cpu().numpy()[full_mask_cpu]
        raw_phase = torch.angle(sig_fft).cpu().numpy()[full_mask_cpu]
        rec_phase = torch.angle(rec_fft).cpu().numpy()[full_mask_cpu]

        # 2. Magnitude Section
        mag_datas = [20*np.log10(raw_mag+1e-8), 20*np.log10(rec_mag+1e-8), (raw_mag-rec_mag)**2]
        mag_titles = ["Mag Raw", "Mag Recon", "Mag Error"]
        mag_cmaps = ['jet', 'jet', 'magma']
        for i in range(3):
            ax = fig_final.add_subplot(gs_col[num_stages + 1 + i, 0])
            im = ax.imshow(mag_datas[i][None, :], aspect='auto', extent=extent, cmap=mag_cmaps[i], 
                           vmin=vmins_m[i] if vmins_m else None, vmax=vmaxs_m[i] if vmaxs_m else None)
            ax.set_yticks([]); ax.set_ylabel(mag_titles[i], rotation=0, labelpad=35, va='center', fontsize=8)
            if is_trial: fig_final.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([])

        # 3. Phase Section
        phase_err = np.abs(np.angle(np.exp(1j * (rec_phase - raw_phase))))
        phase_datas = [raw_phase, rec_phase, phase_err]
        phase_titles = ["Phase Raw", "Phase Recon", "Phase Error"]
        phase_cmaps = ['twilight', 'twilight', 'magma'] # Unified Error Cmap
        for i in range(3):
            ax = fig_final.add_subplot(gs_col[num_stages + 1 + 3 + i, 0])
            im = ax.imshow(phase_datas[i][None, :], aspect='auto', extent=extent, cmap=phase_cmaps[i],
                           vmin=vmins_p[i] if vmins_p else None, vmax=vmaxs_p[i] if vmaxs_p else None)
            ax.set_yticks([]); ax.set_ylabel(phase_titles[i], rotation=0, labelpad=35, va='center', fontsize=8)
            if is_trial: fig_final.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if i < 2: ax.set_xticks([])
            else: ax.set_xlabel("Frequency (Hz)", fontsize=8)

    # Pre-calculate trial limits
    t_sig_fft = torch.fft.rfft(torch.from_numpy(raw_ch).to(device), n=model.n_fft, norm='ortho')
    t_rec_fft = torch.fft.rfft(torch.from_numpy(full_recon_ch).to(device), n=model.n_fft, norm='ortho')
    t_raw_mag = torch.abs(t_sig_fft).cpu().numpy()[full_mask_cpu]
    t_rec_mag = torch.abs(t_rec_fft).cpu().numpy()[full_mask_cpu]
    t_raw_db = 20*np.log10(t_raw_mag+1e-8)
    t_rec_db = 20*np.log10(t_rec_mag+1e-8)
    t_err_m = (t_raw_mag - t_rec_mag)**2
    t_vm_m = [min(t_raw_db.min(), t_rec_db.min()), min(t_raw_db.min(), t_rec_db.min()), 0]
    t_vx_m = [max(t_raw_db.max(), t_rec_db.max()), max(t_raw_db.max(), t_rec_db.max()), t_err_m.max()]
    
    t_raw_ph = torch.angle(t_sig_fft).cpu().numpy()[full_mask_cpu]
    t_rec_ph = torch.angle(t_rec_fft).cpu().numpy()[full_mask_cpu]
    t_err_p = np.abs(np.angle(np.exp(1j * (t_rec_ph - t_raw_ph))))
    t_vm_p = [-np.pi, -np.pi, 0]
    t_vx_p = [np.pi, np.pi, np.pi]

    plot_analysis_column(gs_master[0, 0], raw_ch, full_recon_ch, raw_bands_ch, stage_recons_ch, "TRIAL", is_trial=True,
                         vmins_m=t_vm_m, vmaxs_m=t_vx_m, vmins_p=t_vm_p, vmaxs_p=t_vx_p)
    
    # Pre-calculate patch limits for consistency
    p_raw_mags, p_recon_mags, p_raw_phases, p_recon_phases = [], [], [], []
    x_patch_ch = x_patches[ch_idx].cpu().numpy()
    full_recon_patch_ch = full_recon_ch.reshape(N, L)
    for p in p_indices:
        p_r_f = torch.fft.rfft(torch.from_numpy(x_patch_ch[p]).to(device), n=model.n_fft, norm='ortho')
        p_rec_f = torch.fft.rfft(torch.from_numpy(full_recon_patch_ch[p]).to(device), n=model.n_fft, norm='ortho')
        p_raw_mags.append(torch.abs(p_r_f).cpu().numpy()[full_mask_cpu])
        p_recon_mags.append(torch.abs(p_rec_f).cpu().numpy()[full_mask_cpu])
        p_raw_phases.append(torch.angle(p_r_f).cpu().numpy()[full_mask_cpu])
        p_recon_phases.append(torch.angle(p_rec_f).cpu().numpy()[full_mask_cpu])
    
    p_raw_db = [20*np.log10(m+1e-8) for m in p_raw_mags]
    p_rec_db = [20*np.log10(m+1e-8) for m in p_recon_mags]
    p_err_m = [(m1-m2)**2 for m1, m2 in zip(p_raw_mags, p_recon_mags)]
    p_vm_m = [min(np.min(p_raw_db), np.min(p_rec_db)), min(np.min(p_raw_db), np.min(p_rec_db)), 0]
    p_vx_m = [max(np.max(p_raw_db), np.max(p_rec_db)), max(np.max(p_raw_db), np.max(p_rec_db)), np.max(p_err_m)]
    p_vm_p, p_vx_p = [-np.pi, -np.pi, 0], [np.pi, np.pi, np.pi]

    for p_col, p_idx in enumerate(p_indices):
        p_band_raws = [rb.reshape(N, L)[p_idx] for rb in raw_bands_ch]
        p_band_recons = [sr.reshape(N, L)[p_idx] for sr in stage_recons_ch]
        plot_analysis_column(gs_master[0, 1 + p_col], x_patch_ch[p_idx], full_recon_patch_ch[p_idx], p_band_raws, p_band_recons, f"P{p_idx}",
                             vmins_m=p_vm_m, vmaxs_m=p_vx_m, vmins_p=p_vm_p, vmaxs_p=p_vx_p)

    plt.suptitle(f"Unified 3-Domain Reconstruction Dashboard (Sub {args.subject}, Trial {trial_idx}, Channel {args.channel})", fontsize=24, fontweight='bold')
    plt.savefig(os.path.join(args.output_dir, f"sub{args.subject}_trial{trial_idx}_full_dashboard.png"), dpi=150, bbox_inches='tight', pad_inches=0.1)

    # --- Complete Head Analysis Grid ---
    max_h = max(shr.shape[2] for shr in stage_heads_recons)
    fig_heads = plt.figure(figsize=(28 * num_stages, 2.5 * max_h), constrained_layout=True)
    gs_heads = fig_heads.add_gridspec(max_h, 2 * num_stages, wspace=0.05, hspace=0.1)
    for s_idx in range(num_stages):
        heads_data_s = stage_heads_recons[s_idx][0, ch_idx].cpu().numpy()
        mask_s = getattr(model, f'freq_mask_{s_idx}').cpu().numpy()
        h_cfg = model.decoder_heads_config[s_idx]
        all_h_db = []
        for h in range(heads_data_s.shape[0]):
            h_f = torch.fft.rfft(torch.from_numpy(heads_data_s[h]).to(device), n=model.n_fft, norm='ortho')
            all_h_db.append(20 * np.log10(torch.abs(h_f).cpu().numpy()[mask_s] + 1e-8))
        s_vmin, s_vmax = np.min(all_h_db), np.max(all_h_db)
        last_im_s = None
        for h in range(heads_data_s.shape[0]):
            ax_t = fig_heads.add_subplot(gs_heads[h, 2 * s_idx])
            ax_t.plot(time_vec, heads_data_s[h], color='teal', linewidth=0.8, linestyle=':') # Dotted for consistency
            if h == 0: ax_t.set_title(f"S{s_idx} TEMP", fontsize=16, fontweight='bold')
            ax_t.grid(True, alpha=0.1); ax_t.set_xticks([])
            ax_s = fig_heads.add_subplot(gs_heads[h, 2 * s_idx + 1])
            im_h = ax_s.imshow(all_h_db[h][None, :], aspect='auto', extent=[h_cfg["freq_range"][0], h_cfg["freq_range"][1], 0, 1], cmap='viridis', vmin=s_vmin, vmax=s_vmax)
            ax_s.set_yticks([])
            if h == 0: ax_s.set_title(f"S{s_idx} SPEC", fontsize=16, fontweight='bold')
            if h == heads_data_s.shape[0] - 1: ax_s.set_xlabel("Hz", fontsize=12)
            else: ax_s.set_xticks([])
            last_im_s = im_h
        if last_im_s is not None:
            ax_cb = fig_heads.add_subplot(gs_heads[:, 2 * s_idx + 1]); ax_cb.axis('off')
            cb = fig_heads.colorbar(last_im_s, ax=ax_cb, fraction=0.04, pad=0.08, location='right')
            cb.set_label('dB', fontsize=14)
    plt.suptitle(f"Global Feature Map (Sub {args.subject}, Trial {trial_idx})", fontsize=32, fontweight='bold')
    plt.savefig(os.path.join(args.output_dir, f"sub{args.subject}_trial{trial_idx}_heads_all_stages.png"), dpi=150, bbox_inches='tight', pad_inches=0.1)
    print(f"All plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
