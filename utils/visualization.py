import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import torch
import mne
import random

# --- Helper Functions ---

def _get_subject_indices(dataset, subject_id):
    """Returns indices for a specific subject."""
    subject_mask = (dataset.subject_data == subject_id)
    return subject_mask.nonzero(as_tuple=True)[0]

def _compute_avg_psd(dataset, indices, fs, resolution=0.2):
    """Computes average PSD for a set of trial indices."""
    ffts = []
    for idx in indices:
        x, _ = dataset[idx.item()]
        if isinstance(x, torch.Tensor): x = x.numpy()
        
        nfft = int(fs / resolution)
        nperseg = min(x.shape[-1], nfft)
        freqs, psd = scipy.signal.welch(x, fs=fs, nperseg=nperseg, nfft=nfft, axis=-1)
        ffts.append(psd)
    return np.mean(ffts, axis=0), freqs

def _setup_mne_info(dataset, fs):
    """Creates MNE Info and Montage using coordinates from the dataset object."""
    import mne

    montage_pos = {}
    valid_ch_names = []

    # 1. Try to load standard 10-20 montage for comparison
    try:
        std_montage = mne.channels.make_standard_montage('standard_1020')
        std_positions = std_montage.get_positions()['ch_pos']
        std_keys = {k.upper(): k for k in std_positions.keys()}
    except:
        std_keys = {}
        std_positions = {}

    # dataset.channel_names is list of length Nc
    if hasattr(dataset, 'coords') and dataset.coords is not None:
        coords = dataset.coords
    elif hasattr(dataset, 'all_coords') and len(dataset.all_coords) > 0:
        coords = dataset.all_coords[0]
    else:
        coords = np.zeros((len(dataset.channel_names), 3))

    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    coords = coords.copy()
    names = dataset.channel_names

    for i, name in enumerate(names):
        # A. Case-insensitive match to standard 10-20 (BEST)
        upper_name = name.upper()
        if upper_name in std_keys:
            montage_pos[name] = std_positions[std_keys[upper_name]]
            valid_ch_names.append(name)
            continue
            
        # B. Fallback to Metadata coordinates
        pos = coords[i]
        if np.any(pos != 0):
            # Dynamic scaling: Dial uses radius ~0.5, BETA ~0.5. Standard MNE head ~0.095.
            # We scale such that the max radius of the whole set becomes ~0.095
            # but only if the coordinates are in 'large' units.
            radii = np.sqrt(np.sum(coords[:, :2]**2, axis=1))
            current_max_r = np.max(radii)
            
            if current_max_r > 0.2: # Likely 0-1 scale
                scale_factor = 0.095 / current_max_r
                pos = pos * scale_factor
                
            montage_pos[name] = pos
            valid_ch_names.append(name)
        else:
            print(f"Skipping channel {name} in Topomap due to missing coordinates.")

    # Create info
    info = mne.create_info(ch_names=valid_ch_names, sfreq=fs, ch_types='eeg')

    try:
        if montage_pos:
            dig_montage = mne.channels.make_dig_montage(ch_pos=montage_pos, coord_frame='head')
            info.set_montage(dig_montage)
    except Exception as e:
        print(f"Error setting custom montage: {e}")

    return info
# --- Main Visualization Functions ---

def visualize_masking(pretrain_dataset, subject_id, output_dir='output/visualization', trial_idx=None):
    """
    Visualizes the masking strategy using a grid of individual patch plots.
    Rows = All Channels, Cols = Patches.
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"Generating Masking Grid Visualization for Subject {subject_id}...")

    indices = _get_subject_indices(pretrain_dataset.base_dataset, subject_id)
    if len(indices) == 0: return
    
    if trial_idx is not None:
        idx = indices[trial_idx % len(indices)].item()
    else:
        idx = random.choice(indices).item()
    x_patches, coords, mask, time_indices, y, _ = pretrain_dataset[idx]
    
    # Get coordinates: (C, 3)
    coords_idx = pretrain_dataset.base_dataset.trial_to_coords_idx[idx]
    
    C, P, T_patch = x_patches.shape
    mask_2d = mask.reshape(C, P)
    
    # Dynamically scale height based on channel count (64 channels needs a very tall plot)
    fig_height = max(10, C * 0.8)
    fig, axes = plt.subplots(C, P, figsize=(P * 2.5, fig_height), squeeze=False)
    
    for ch in range(C):
        # Calculate shared Y-limits for this channel across all patches
        ch_min = x_patches[ch].min().item()
        ch_max = x_patches[ch].max().item()
        # Add small margin
        margin = (ch_max - ch_min) * 0.1 if ch_max > ch_min else 0.1
        ylim = (ch_min - margin, ch_max + margin)

        for p in range(P):
            ax = axes[ch, p]
            is_masked = mask_2d[ch, p].item()
            patch_data = x_patches[ch, p].numpy()
            
            color = 'red' if is_masked else 'blue'
            ax.plot(patch_data, color=color, linewidth=0.8)
            ax.set_ylim(ylim) # Synchronize Y-axis for this channel row
            
            if is_masked:
                ax.set_facecolor('#fff0f0') 
            else:
                ax.set_facecolor('#f0faff') 
                
            ax.set_xticks([])
            ax.set_yticks([])
            
            if ch == 0:
                ax.set_title(f"P{p}", fontsize=10)
            if p == 0:
                ch_name = pretrain_dataset.base_dataset.channel_names[ch]
                ax.set_ylabel(ch_name, fontsize=8, rotation=0, labelpad=25, verticalalignment='center')

    fig.suptitle(f"Subject {subject_id} Masking Pattern (Red=Masked, Blue=Visible)\nAll {C} Channels", fontsize=16)
    plt.tight_layout(rect=[0, 0.01, 1, 0.98])
    
    save_path = os.path.join(output_dir, f'Subject_{subject_id}_Masking_Grid.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

def visualize_raw_eeg(dataset, subject_id, output_dir='output/visualization', fs=None, patch_len_pts=None, trial_idx=None):
    """
    Plots a grid of patches for Raw EEG. Rows=Channels, Cols=Patches.
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"Generating Patched Raw EEG Grid for Subject {subject_id}...")
    
    if fs is None:
        model_type = dataset.config.get('training_params', {}).get('model_type', 'AttnVQ')
        fs = dataset.config.get('model_params', {}).get(model_type, {}).get('preprocess', {}).get('target_freq', 200)
    if patch_len_pts is None:
        model_type = dataset.config.get('training_params', {}).get('model_type', 'AttnVQ')
        patch_len_pts = dataset.config.get('model_params', {}).get(model_type, {}).get('preprocess', {}).get('patch_length', 200)

    indices = _get_subject_indices(dataset, subject_id)
    if len(indices) == 0: return

    if trial_idx is not None:
        idx = indices[trial_idx % len(indices)].item()
    else:
        idx = random.choice(indices).item()
    x_raw, label = dataset[idx]
    if isinstance(x_raw, torch.Tensor): x_raw = x_raw.numpy()
    
    C, T = x_raw.shape
    num_patches = T // patch_len_pts
    x_patches = x_raw[:, :num_patches * patch_len_pts].reshape(C, num_patches, patch_len_pts)
    
    fig_height = max(10, C * 0.8)
    fig, axes = plt.subplots(C, num_patches, figsize=(num_patches * 2.5, fig_height), squeeze=False)
    
    for ch in range(C):
        ch_min, ch_max = x_patches[ch].min(), x_patches[ch].max()
        margin = (ch_max - ch_min) * 0.1 if ch_max > ch_min else 0.1
        ylim = (ch_min - margin, ch_max + margin)

        for p in range(num_patches):
            ax = axes[ch, p]
            ax.plot(x_patches[ch, p], color='black', linewidth=0.8)
            ax.set_ylim(ylim)
            ax.set_xticks([]); ax.set_yticks([])
            if ch == 0: ax.set_title(f"P{p}")
            if p == 0:
                ch_name = dataset.channel_names[ch] if hasattr(dataset, 'channel_names') else f"Ch{ch+1}"
                ax.set_ylabel(ch_name, fontsize=8, rotation=0, labelpad=25, verticalalignment='center')

    fig.suptitle(f"Subject {subject_id} | Class {label} - Raw EEG Patches", fontsize=16)
    plt.tight_layout(rect=[0, 0.01, 1, 0.98])
    save_path = os.path.join(output_dir, f'Subject_{subject_id}_Patches_Raw.png')
    plt.savefig(save_path, dpi=150); plt.close(fig)
    print(f"Saved: {save_path}")

def visualize_amplitude_phase(dataset, subject_id, output_dir='output/visualization', patch_len_pts=None, trial_idx=None):
    """
    Plots a grid showing Amplitude (rows) and Phase (optional side-by-side or separate).
    We'll do Amplitude only for the grid to keep it readable, or side-by-side columns.
    Let's do: Cols = [P0 Amp, P0 Phase, P1 Amp, P1 Phase, ...]
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"Generating Paired Amp/Phase Grid for Subject {subject_id}...")
    
    if patch_len_pts is None:
        model_type = dataset.config.get('training_params', {}).get('model_type', 'AttnVQ')
        patch_len_pts = dataset.config.get('model_params', {}).get(model_type, {}).get('preprocess', {}).get('patch_length', 200)

    indices = _get_subject_indices(dataset, subject_id)
    if len(indices) == 0: return

    if trial_idx is not None:
        idx = indices[trial_idx % len(indices)].item()
    else:
        idx = random.choice(indices).item()
    x_raw, label = dataset[idx]
    if isinstance(x_raw, np.ndarray): x_raw = torch.from_numpy(x_raw)
    
    C, T = x_raw.shape
    num_patches = T // patch_len_pts
    x_patches = x_raw[:, :num_patches * patch_len_pts].reshape(C, num_patches, patch_len_pts)
    
    fft_data = torch.fft.rfft(x_patches, norm='ortho')
    amp = torch.abs(fft_data).numpy()
    phase = torch.angle(fft_data).numpy()
    
    fig_height = max(10, C * 0.8)
    fig, axes = plt.subplots(C, num_patches * 2, figsize=(num_patches * 5, fig_height), squeeze=False)
    
    for ch in range(C):
        amp_min, amp_max = amp[ch].min(), amp[ch].max()
        a_margin = (amp_max - amp_min) * 0.1 if amp_max > amp_min else 0.1
        amp_ylim = (amp_min - a_margin, amp_max + a_margin)
        
        phase_ylim = (-np.pi - 0.2, np.pi + 0.2)

        for p in range(num_patches):
            # Amp
            ax_a = axes[ch, p*2]
            ax_a.plot(amp[ch, p], color='purple', linewidth=0.8)
            ax_a.set_ylim(amp_ylim); ax_a.set_xticks([]); ax_a.set_yticks([])
            ax_a.set_facecolor('#faf0fa')
            
            # Phase
            ax_p = axes[ch, p*2 + 1]
            ax_p.plot(phase[ch, p], color='green', linewidth=0.8)
            ax_p.set_ylim(phase_ylim); ax_p.set_xticks([]); ax_p.set_yticks([])
            ax_p.set_facecolor('#f0faf0')
            
            if ch == 0:
                ax_a.set_title(f"P{p} Amp")
                ax_p.set_title(f"P{p} Phase")
            if p == 0:
                ch_name = dataset.channel_names[ch] if hasattr(dataset, 'channel_names') else f"Ch{ch+1}"
                ax_a.set_ylabel(ch_name, fontsize=8, rotation=0, labelpad=25, verticalalignment='center')

    fig.suptitle(f"Subject {subject_id} | Class {label} - FFT Amplitude & Phase Grid", fontsize=16)
    plt.tight_layout(rect=[0, 0.01, 1, 0.98])
    save_path = os.path.join(output_dir, f'Subject_{subject_id}_Paired_AmpPhase.png')
    plt.savefig(save_path, dpi=150); plt.close(fig)
    print(f"Saved: {save_path}")

def visualize_real_imaginary(dataset, subject_id, output_dir='output/visualization', patch_len_pts=None, trial_idx=None):
    """
    Plots a grid: Cols = [P0 Real, P0 Imag, P1 Real, P1 Imag, ...]
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"Generating Paired Real/Imag Grid for Subject {subject_id}...")
    
    if patch_len_pts is None:
        model_type = dataset.config.get('training_params', {}).get('model_type', 'AttnVQ')
        patch_len_pts = dataset.config.get('model_params', {}).get(model_type, {}).get('preprocess', {}).get('patch_length', 200)

    indices = _get_subject_indices(dataset, subject_id)
    if len(indices) == 0: return

    if trial_idx is not None:
        idx = indices[trial_idx % len(indices)].item()
    else:
        idx = random.choice(indices).item()
    x_raw, label = dataset[idx]
    if isinstance(x_raw, np.ndarray): x_raw = torch.from_numpy(x_raw)
    
    C, T = x_raw.shape
    num_patches = T // patch_len_pts
    x_patches = x_raw[:, :num_patches * patch_len_pts].reshape(C, num_patches, patch_len_pts)
    
    fft_data = torch.fft.rfft(x_patches, norm='ortho')
    real_p = fft_data.real.numpy()
    imag_p = fft_data.imag.numpy()
    
    fig_height = max(10, C * 0.8)
    fig, axes = plt.subplots(C, num_patches * 2, figsize=(num_patches * 5, fig_height), squeeze=False)
    
    for ch in range(C):
        r_min, r_max = real_p[ch].min(), real_p[ch].max()
        i_min, i_max = imag_p[ch].min(), imag_p[ch].max()
        # Use shared Y limit for Real/Imag within same channel for comparability
        v_min, v_max = min(r_min, i_min), max(r_max, i_max)
        margin = (v_max - v_min) * 0.1 if v_max > v_min else 0.1
        ylim = (v_min - margin, v_max + margin)

        for p in range(num_patches):
            # Real
            ax_r = axes[ch, p*2]
            ax_r.plot(real_p[ch, p], color='blue', linewidth=0.8)
            ax_r.set_ylim(ylim); ax_r.set_xticks([]); ax_r.set_yticks([])
            ax_r.set_facecolor('#f0f4fa')
            
            # Imag
            ax_i = axes[ch, p*2 + 1]
            ax_i.plot(imag_p[ch, p], color='darkorange', linewidth=0.8)
            ax_i.set_ylim(ylim); ax_i.set_xticks([]); ax_i.set_yticks([])
            ax_i.set_facecolor('#faf6f0')
            
            if ch == 0:
                ax_r.set_title(f"P{p} Real")
                ax_i.set_title(f"P{p} Imag")
            if p == 0:
                ch_name = dataset.channel_names[ch] if hasattr(dataset, 'channel_names') else f"Ch{ch+1}"
                ax_r.set_ylabel(ch_name, fontsize=8, rotation=0, labelpad=25, verticalalignment='center')

    fig.suptitle(f"Subject {subject_id} | Class {label} - FFT Real & Imaginary Grid", fontsize=16)
    plt.tight_layout(rect=[0, 0.01, 1, 0.98])
    save_path = os.path.join(output_dir, f'Subject_{subject_id}_Paired_RealImag.png')
    plt.savefig(save_path, dpi=150); plt.close(fig)
    print(f"Saved: {save_path}")

def visualize_psd_grid(dataset, subject_id, config, output_dir='output/visualization'):
    """
    Plots a grid of Average PSD Heatmaps for all classes of a subject.
    Top-Down (Channel 1 at the top).
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    indices = _get_subject_indices(dataset, subject_id)
    if len(indices) == 0: 
        print(f"No indices found for Subject {subject_id}")
        return
    
    # Dataset-level labels
    if hasattr(dataset, 'labels'):
        subject_labels = dataset.labels[indices]
    elif hasattr(dataset, 'label_data'):
        subject_labels = dataset.label_data[indices]
    else:
        subject_labels = torch.zeros(len(indices))
        
    unique_labels = sorted(torch.unique(subject_labels).tolist())
    print(f"Generating PSD Grid for Subject {subject_id} ({len(unique_labels)} classes)...")
    
    # Config
    model_type = config.get('training_params', {}).get('model_type', 'AttnVQ')
    preprocess = config.get('model_params', {}).get(model_type, {}).get('preprocess', {})
    fs = preprocess.get('target_freq', 200)
    f_start = preprocess.get('l_freq', 0)
    f_end = preprocess.get('h_freq', 75)
    stim_map = config.get('data_metadata', {}).get('stimulus_hz', {})
    
    # Compute All Data first for Global Scaling
    all_psd_db = []
    plot_freqs = None
    freq_mask = None
    
    for label in unique_labels:
        local_idxs = (subject_labels == label).nonzero(as_tuple=True)[0]
        global_idxs = indices[local_idxs]
        avg_psd, freqs = _compute_avg_psd(dataset, global_idxs, fs)
        
        if freq_mask is None:
            freq_mask = (freqs >= f_start) & (freqs <= f_end)
            plot_freqs = freqs[freq_mask]
            
        all_psd_db.append(10 * np.log10(avg_psd[:, freq_mask] + 1e-12))
        
    if not all_psd_db:
        print("No PSD data computed.")
        return

    vmin = np.min([np.percentile(p, 5) for p in all_psd_db])
    vmax = np.max([np.percentile(p, 95) for p in all_psd_db])
    
    # Plot Grid
    cols = 8
    rows = int(np.ceil(len(unique_labels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*5), squeeze=False)
    fig.suptitle(f'Subject {subject_id} - Average PSD Heatmap (Top-Down Channels)\nRange: {f_start}-{f_end} Hz', fontsize=16)
    axes = axes.flatten()
    
    for i, label in enumerate(unique_labels):
        ax = axes[i]
        # origin='upper' + extent [f0, f1, Nc, 0] -> Ch1 (index 0) is at y=0 (top)
        im = ax.imshow(
            all_psd_db[i],
            aspect='auto',
            origin='upper',
            extent=[plot_freqs[0], plot_freqs[-1], dataset.Nc, 0],
            cmap='jet',
            vmin=vmin, vmax=vmax
        )
        
        stim_val = stim_map.get(str(int(label)), 0)
        try:
            target_hz = float(stim_val)
        except (ValueError, TypeError):
            target_hz = 0
            
        if target_hz > 0:
            title = f"Class {label}: {target_hz}Hz"
        else:
            title = f"Class {label}: {stim_val}" if stim_val else f"Class {label}"
            
        ax.set_title(title, fontsize=10)
        
        if target_hz > 0 and f_start <= target_hz <= f_end:
            ax.axvline(target_hz, color='white', linestyle='--', alpha=0.6)
            
        # Channel labels
        if i % cols == 0:
            ax.set_yticks(np.arange(0.5, dataset.Nc + 0.5, 1))
            ax.set_yticklabels(dataset.channel_names, fontsize=5)
        else:
            ax.set_yticks([])
            
        ax.set_xlabel("Hz", fontsize=8)
        
    # Shared Colorbar
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Power (dB)')
    
    # Cleanup empty slots
    for j in range(len(unique_labels), len(axes)): axes[j].axis('off')
    
    save_path = os.path.join(output_dir, f'Subject_{subject_id}_PSD_Grid.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

def visualize_topo_grid(dataset, subject_id, config, output_dir='output/visualization'):
    """
    Plots a grid of Topographic Maps (at Target Freq) for all classes.
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"Generating Topo Grid for Subject {subject_id}...")
    
    indices = _get_subject_indices(dataset, subject_id)
    if len(indices) == 0: return
    
    if hasattr(dataset, 'labels'):
        subject_labels = dataset.labels[indices]
    elif hasattr(dataset, 'label_data'):
        subject_labels = dataset.label_data[indices]
    else:
        subject_labels = torch.zeros(len(indices))
        
    unique_labels = sorted(torch.unique(subject_labels).tolist())
    
    # Config & Info
    model_type = config.get('training_params', {}).get('model_type', 'AttnVQ')
    preprocess = config.get('model_params', {}).get(model_type, {}).get('preprocess', {})
    fs = preprocess.get('target_freq', 200)
    stim_map = config.get('data_metadata', {}).get('stimulus_hz', {})
    info = _setup_mne_info(dataset, fs)
    
    # Get the channel names that were actually included in the MNE Info object
    channels_in_info = info.ch_names
    
    # Find the indices of these channels in the original dataset.channel_names
    original_channel_names = dataset.channel_names
    channel_indices_for_info = [original_channel_names.index(ch) for ch in channels_in_info]

    # Compute Data
    all_topo_vals = []
    valid_labels = [] # Only labels with defined targets
    
    for label in unique_labels:
        stim_val = stim_map.get(str(int(label)), 0)
        try:
            target_hz = float(stim_val)
        except (ValueError, TypeError):
            target_hz = 0
        
        # If no target frequency (Motor Imagery/Generic), use Alpha Band (8-13Hz)
        if target_hz <= 0:
            band_min, band_max = 8.0, 13.0
            title = f"Class {int(label)}: {stim_val} (8-13Hz)" if stim_val else f"Class {int(label)} (8-13Hz)"
        else:
            band_min, band_max = target_hz - 0.2, target_hz + 0.2
            title = f"{target_hz} Hz"
            
        valid_labels.append((label, title))
        local_idxs = (subject_labels == label).nonzero(as_tuple=True)[0]
        global_idxs = indices[local_idxs]
        
        avg_psd, freqs = _compute_avg_psd(dataset, global_idxs, fs)
        avg_psd_filtered = avg_psd[channel_indices_for_info, :]
        
        band = (freqs >= band_min) & (freqs <= band_max)
        power = np.mean(avg_psd_filtered[:, band], axis=1) if np.sum(band) > 0 else np.zeros(len(channels_in_info))
        all_topo_vals.append(power)
        
    if not all_topo_vals: return

    vmin = np.min(all_topo_vals)
    vmax = np.max(all_topo_vals)
    
    # Plot Grid
    cols = 8
    rows = int(np.ceil(len(valid_labels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3), squeeze=False)
    fig.suptitle(f'Subject {subject_id} - Spatial Power Distribution', fontsize=16)
    axes = axes.flatten()
    
    for i, (label, title) in enumerate(valid_labels):
        ax = axes[i]
        mne.viz.plot_topomap(
            all_topo_vals[i], 
            info, 
            axes=ax, 
            show=False, 
            cmap='jet', 
            contours=0, 
            sensors=True,
            extrapolate='local',
            res=64,
            vlim=(vmin, vmax)
        )
        ax.set_title(title, fontsize=10)
        
    # Shared Colorbar
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label='Power (linear)')
    
    # Cleanup
    for j in range(len(valid_labels), len(axes)): axes[j].axis('off')
    
    save_path = os.path.join(output_dir, f'Subject_{subject_id}_Topo_Grid.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

def visualize_band_time_series(dataset, subject_id, output_dir='output/visualization', fs=None, channel_label='Oz', trial_idx=None):
    """
    Plots time-domain signals for Raw, Delta, Theta, Alpha, Beta, Gamma for a specific channel.
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"Generating Band Time-Series for Subject {subject_id}, Channel {channel_label}...")

    if fs is None:
        model_type = dataset.config.get('training_params', {}).get('model_type', 'AttnVQ')
        fs = dataset.config.get('model_params', {}).get(model_type, {}).get('preprocess', {}).get('target_freq', 200)

    indices = _get_subject_indices(dataset, subject_id)
    if len(indices) == 0: return

    # Trial Selection
    if trial_idx is not None:
        idx = indices[trial_idx % len(indices)].item()
    else:
        idx = random.choice(indices).item()
    x_raw, label = dataset[idx] # (C, T)
    if isinstance(x_raw, torch.Tensor): x_raw = x_raw.numpy()

    # Find Channel Index
    ch_idx = -1
    # dataset.channel_names should exist
    clean_names = [n.strip().upper() for n in dataset.channel_names]
    target = channel_label.strip().upper()
    
    if target in clean_names:
        ch_idx = clean_names.index(target)
    else:
        print(f"Channel {channel_label} not found. Using first channel.")
        ch_idx = 0
        channel_label = dataset.channel_names[0]

    # Extract single channel data: Shape (1, T) for MNE filter
    data_1ch = x_raw[ch_idx:ch_idx+1, :].astype(np.float64)
    
    # Define Bands
    bands = {
        'Delta (0.5-4 Hz)': (0.5, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
        'Gamma (30-80 Hz)': (30, 80)
    }

    fig, axes = plt.subplots(len(bands) + 1, 1, figsize=(10, 12), sharex=True)
    time_vec = np.arange(data_1ch.shape[1]) / fs
    
    # Plot Raw
    axes[0].plot(time_vec, data_1ch[0], color='k', linewidth=0.8)
    axes[0].set_title(f"Raw Signal - {channel_label}")
    axes[0].set_ylabel("Amplitude")

    # Filter and Plot Bands
    for i, (name, (l_freq, h_freq)) in enumerate(bands.items()):
        ax = axes[i+1]
        # MNE filter_data
        # Use IIR filter for short signals to avoid length warnings
        filtered = mne.filter.filter_data(data_1ch, fs, l_freq, h_freq, method='iir', verbose=False)
        
        ax.plot(time_vec, filtered[0], linewidth=0.8)
        ax.set_title(name)
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle=':', alpha=0.6)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'Subject_{subject_id}_Bands_{channel_label}.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")