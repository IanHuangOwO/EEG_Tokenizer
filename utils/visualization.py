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
    
    montage_pos = {}
    valid_ch_names = []
    
    # dataset.coords is (Nc, 3)
    # dataset.channel_names is list of length Nc
    coords = dataset.coords.copy()
    names = dataset.channel_names
    
    # Scale coordinates to MNE standard radius (approx 0.095)
    radii = np.sqrt(np.sum(coords[:, :2]**2, axis=1))
    max_r = np.max(radii)
    if max_r > 0:
        scale_factor = 0.06 / max_r
        coords *= scale_factor
    
    for i, name in enumerate(names):
        pos = coords[i]
        
        # Check if coord is valid (not all zeros)
        if np.any(pos != 0):
            montage_pos[name] = pos
            valid_ch_names.append(name)
        else:
            # If [0,0,0], we still add it but maybe skip if it causes MNE issues.
            # Usually, it's better to include it at origin than skip if we want to plot.
            # But let's skip to be safe and avoid "overlapping channels" error in MNE.
            print(f"Skipping channel {name} in Topomap due to missing coordinates.")
            
    # Create info with only the channels that have valid positions
    info = mne.create_info(ch_names=valid_ch_names, sfreq=fs, ch_types='eeg')
    
    try:
        dig_montage = mne.channels.make_dig_montage(ch_pos=montage_pos, coord_frame='head')
        info.set_montage(dig_montage)
    except Exception as e:
        print(f"Error setting custom montage: {e}")
        
    return info

# --- Main Visualization Functions ---

def visualize_raw_eeg(dataset, subject_id, output_dir='output/visualization', fs=200):
    """
    Plots a single random trial's Raw EEG for a specific subject.
    Stacked Top-Down (Channel 1 at the top).
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"Generating Raw EEG Plot for Subject {subject_id}...")
    
    indices = _get_subject_indices(dataset, subject_id)
    if len(indices) == 0: return

    # Random Trial
    rand_idx = random.choice(indices).item()
    x_raw, label = dataset[rand_idx]
    if isinstance(x_raw, torch.Tensor): x_raw = x_raw.numpy()
    
    num_chans, num_pts = x_raw.shape
    time_vec = np.arange(num_pts) / fs
    
    fig, ax = plt.subplots(figsize=(12, 12))
    std_dev = np.std(x_raw)
    spacing = std_dev * 5 if std_dev > 0 else 1.0
    
    yticks, yticklabels = [], []
    for ch in range(num_chans):
        # Top-Down: Ch1 (ch=0) at the top
        offset = (num_chans - 1 - ch) * spacing
        ax.plot(time_vec, x_raw[ch] + offset, color='k', linewidth=0.3)
        
        # Show ALL labels
        yticks.append(offset)
        yticklabels.append(dataset.channel_names[ch] if hasattr(dataset, 'channel_names') else f"Ch{ch+1}")
            
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=6)
    ax.set_title(f"Subject {subject_id} | Class {label} | Raw EEG (Top-Down)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels")
    ax.margins(x=0)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'Subject_{subject_id}_Raw.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

def visualize_psd_grid(dataset, subject_id, config, output_dir='output/visualization'):
    """
    Plots a grid of Average PSD Heatmaps for all classes of a subject.
    Top-Down (Channel 1 at the top).
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"Generating PSD Grid for Subject {subject_id}...")
    
    indices = _get_subject_indices(dataset, subject_id)
    if len(indices) == 0: return
    
    subject_labels = dataset.label_data[indices]
    unique_labels = sorted(torch.unique(subject_labels).tolist())
    
    # Config
    model_type = config['training_params']['model_type']
    params = config['model_params'][model_type]
    fs = 200
    f_start = params.get('start_freq', 0)
    f_end = params.get('end_freq', 20)
    stim_map = config['data_metadata']['stimulus_hz']
    
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
        
    vmin = np.min([np.percentile(p, 5) for p in all_psd_db])
    vmax = np.max([np.percentile(p, 95) for p in all_psd_db])
    
    # Plot Grid
    cols = 8
    rows = int(np.ceil(len(unique_labels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*6)) # Larger figsize for labels
    fig.suptitle(f'Subject {subject_id} - Average PSD per Class (Top-Down Channels)', fontsize=16)
    axes = axes.flatten()
    
    for i, label in enumerate(unique_labels):
        ax = axes[i]
        im = ax.imshow(
            all_psd_db[i],
            aspect='auto',
            origin='lower',
            extent=[plot_freqs[0], plot_freqs[-1], dataset.Nc, 0],
            cmap='jet',
            vmin=vmin, vmax=vmax
        )
        
        target_hz = float(stim_map.get(str(label), 0))
        title = f"{label}: {target_hz}Hz" if target_hz > 0 else f"{label}"
        ax.set_title(title, fontsize=10)
        
        if target_hz > 0 and f_start <= target_hz <= f_end:
            ax.axvline(target_hz, color='white', linestyle='--')
            
        # Set ALL channel labels
        ax.set_yticks(np.arange(0.5, dataset.Nc + 0.5, 1))
        ax.set_yticklabels(dataset.channel_names, fontsize=5)
        ax.set_xlabel("Hz", fontsize=8)
        
    # Shared Colorbar
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Power (dB)')
    
    # Cleanup
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
    
    subject_labels = dataset.label_data[indices]
    unique_labels = sorted(torch.unique(subject_labels).tolist())
    
    # Config & Info
    fs = 200
    stim_map = config['data_metadata']['stimulus_hz']
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
        target_hz = float(stim_map.get(str(label), 0))
        if target_hz <= 0: continue
        
        valid_labels.append((label, target_hz))
        local_idxs = (subject_labels == label).nonzero(as_tuple=True)[0]
        global_idxs = indices[local_idxs]
        
        avg_psd, freqs = _compute_avg_psd(dataset, global_idxs, fs)
        
        # Filter avg_psd to only include channels present in info
        avg_psd_filtered = avg_psd[channel_indices_for_info, :]
        
        # Extract Band Power
        band = (freqs >= target_hz - 0.2) & (freqs <= target_hz + 0.2)
        power = np.mean(avg_psd_filtered[:, band], axis=1) if np.sum(band) > 0 else np.zeros(len(channels_in_info))
        all_topo_vals.append(power)
        
    if not all_topo_vals: return

    vmin = np.min(all_topo_vals)
    vmax = np.max(all_topo_vals)
    
    # Plot Grid
    cols = 8
    rows = int(np.ceil(len(valid_labels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    fig.suptitle(f'Subject {subject_id} - Target Freq Topomaps', fontsize=16)
    axes = axes.flatten()
    
    for i, (label, target_hz) in enumerate(valid_labels):
        ax = axes[i]
        mne.viz.plot_topomap(
            all_topo_vals[i], 
            info, 
            axes=ax, 
            show=False, 
            cmap='jet', 
            contours=0, 
            sensors=False,
            extrapolate='head',
            res=32,
            vlim=(vmin, vmax)
        )
        ax.set_title(f"{target_hz} Hz", fontsize=10)
        
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

def visualize_band_time_series(dataset, subject_id, output_dir='output/visualization', fs=200, channel_label='Oz'):
    """
    Plots time-domain signals for Raw, Delta, Theta, Alpha, Beta, Gamma for a specific channel.
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"Generating Band Time-Series for Subject {subject_id}, Channel {channel_label}...")

    indices = _get_subject_indices(dataset, subject_id)
    if len(indices) == 0: return

    # Random Trial
    rand_idx = random.choice(indices).item()
    x_raw, label = dataset[rand_idx] # (C, T)
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