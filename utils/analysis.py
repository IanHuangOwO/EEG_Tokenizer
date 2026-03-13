import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import mne
import torch

# --- 1. Geometry Helpers ---

def _setup_mne_info(dataset, fs=1000):
    """
    Creates MNE Info and Montage using coordinates from the dataset object.
    Scales coordinates to match MNE standard head size (~0.095m).
    """
    base_ds = dataset.base_dataset if hasattr(dataset, 'base_dataset') else dataset
    
    montage_pos = {}
    valid_ch_names = []
    
    # dataset.coords is (Nc, 3)
    coords = base_ds.coords.copy()
    names = base_ds.channel_names
    
    # Scale coordinates to MNE standard radius (approx 0.095)
    # This prevents the 'tiny head' issue where sensors are at radius 0.5
    radii = np.sqrt(np.sum(coords[:, :2]**2, axis=1))
    max_r = np.max(radii)
    if max_r > 0:
        scale_factor = 0.06 / max_r
        coords *= scale_factor
    
    for i, name in enumerate(names):
        pos = coords[i]
        # Skip M1/M2 and invalid coords
        # if name.upper() in ['M1', 'M2']:
        #     continue
            
        if np.any(pos != 0):
            montage_pos[name] = pos
            valid_ch_names.append(name)
            
    info = mne.create_info(ch_names=valid_ch_names, sfreq=fs, ch_types='eeg')
    
    try:
        # Using MNE default head radius implicitly now
        dig_montage = mne.channels.make_dig_montage(ch_pos=montage_pos, coord_frame='head')
        info.set_montage(dig_montage)
    except Exception as e:
        print(f"Error setting custom montage: {e}")
        
    return info

def find_neighbors_from_dataset(dataset, threshold=None):
    """
    Computes a dictionary mapping each channel to its neighbors based on distance in dataset.coords.
    """
    base_ds = dataset.base_dataset if hasattr(dataset, 'base_dataset') else dataset
    coords = base_ds.coords # (Nc, 3)
    names = base_ds.channel_names
    
    if threshold is None:
        span = np.max(coords) - np.min(coords)
        threshold = span * 0.25
        print(f"Estimated neighbor threshold: {threshold:.4f}")

    neighbors = {}
    for i, c1 in enumerate(names):
        p1 = coords[i]
        n_list = []
        for j, c2 in enumerate(names):
            if i == j: continue
            p2 = coords[j]
            dist = np.sqrt(np.sum((p1 - p2)**2))
            if dist <= threshold: 
                n_list.append(c2)
        neighbors[c1] = n_list
        
    return neighbors

# --- 2. Core Analysis Logic ---

def calculate_uniqueness_metrics(df, config, neighbors):
    """
    Calculates percentage-based metrics for each channel across Scales AND Heads:
    1. Unique_Count_Pct: (Unique Tokens in Head / Total Tokens in Head) * 100
    2. Unique_Weight_Pct: (Sum of weights of unique tokens / Total weight sum of head) * 100
    """
    channels = df['Channel'].unique()
    
    # Extract params from config
    model_type = config['training_params']['model_type']
    params = config['model_params'][model_type]
    in_scales = params.get('in_scales', 3)
    vq_head_num = params.get('vq_head_num', 8)
    vq_head_top_k = params.get('vq_head_top_k', 8)
    
    results = []
    
    for s in range(in_scales):
        scale_label = f'S{s}'
        df_s = df[df['Scale'] == scale_label]
        
        for h in range(vq_head_num):
            head_label = f'H{h}'
            
            channel_to_tokens = {}
            channel_to_weights = {}
            
            idx_cols = [f'idx_H{h}_K{k}' for k in range(vq_head_top_k)]
            wt_cols = [f'weight_H{h}_K{k}' for k in range(vq_head_top_k)]

            for ch in channels:
                ch_rows = df_s[df_s['Channel'] == ch]
                if len(ch_rows) == 0: continue
                
                row = ch_rows.iloc[0]
                channel_to_tokens[ch] = row[idx_cols].values
                channel_to_weights[ch] = row[wt_cols].values

            for c1 in channel_to_tokens.keys():
                c1_tokens = channel_to_tokens[c1]
                c1_weights = channel_to_weights[c1]
                
                tokens_c1_set = set(c1_tokens)
                tokens_n_set = set()
                
                ch_neighbors = neighbors.get(c1, [])
                for n in ch_neighbors:
                    if n in channel_to_tokens:
                        tokens_n_set.update(channel_to_tokens[n])
                    
                unique_tokens = tokens_c1_set - tokens_n_set
                
                # 1. Percentage of tokens that are unique
                unique_count_pct = (len(unique_tokens) / vq_head_top_k) * 100
                
                # 2. Percentage of weight attributed to unique tokens
                unique_weight_sum = sum(w for t, w in zip(c1_tokens, c1_weights) if t in unique_tokens)
                total_weight_sum = sum(c1_weights)
                unique_weight_pct = (unique_weight_sum / total_weight_sum) * 100 if total_weight_sum > 0 else 0
                
                results.append({
                    'Channel': c1,
                    'Scale': scale_label,
                    'Head': head_label,
                    'Unique_Count_Pct': unique_count_pct,
                    'Unique_Weight_Pct': unique_weight_pct
                })
            
    return pd.DataFrame(results)

# --- 3. Visualization ---

def _plot_uniqueness_metric_grid(res_df, info, metric_name, label, model_name, output_dir):
    """Internal helper to plot S x Head grid for a metric."""
    unique_scales = sorted(res_df['Scale'].unique())
    unique_heads = sorted(res_df['Head'].unique(), key=lambda x: int(x[1:]))
    
    n_rows = len(unique_scales)
    n_cols = len(unique_heads)
    mne_channels = info.ch_names
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.5))
    if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
    elif n_rows == 1: axes = axes[np.newaxis, :]
    elif n_cols == 1: axes = axes[:, np.newaxis]

    # Use fixed 0-100 scale for percentages to make cross-plot comparison easy
    vmin, vmax = 0, 100

    im = None
    for r, scale in enumerate(unique_scales):
        for c, head in enumerate(unique_heads):
            data_subset = res_df[(res_df['Scale'] == scale) & (res_df['Head'] == head)]
            
            ch_val_dict = dict(zip(data_subset['Channel'], data_subset[metric_name]))
            data_values = np.array([ch_val_dict.get(ch, 0.0) for ch in mne_channels])
            
            ax = axes[r, c]
            im, _ = mne.viz.plot_topomap(
                data_values, 
                info, 
                axes=ax, 
                cmap='viridis',
                show=False, 
                sphere=None, # Use MNE default (0.095)
                extrapolate='head',
                vlim=(vmin, vmax)
            )
            if r == 0:
                ax.set_title(f'Head {head}', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'Scale {scale}', fontsize=10, labelpad=20)

    # Add Colorbar
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    if im:
        fig.colorbar(im, cax=cbar_ax, label=label)

    plt.suptitle(f"{label} Grid (Scale x Head) - {model_name}", fontsize=16, y=0.98)
    
    filename = f"{metric_name.lower()}_grid_topomap.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

def visualize_weighted_uniqueness_topo(config, dataset, csv_path=None, output_dir=None):
    """
    Generates MNE Topographical grids (Scale x Head) for:
    1. Unique Token Allocation (%)
    2. Unique Token Contribution (Weight %)
    """
    model_name = config['training_params']['model_name']
    
    if csv_path is None:
        csv_path = f"./output/{model_name}/visualization/neighbor_codes_long.csv"
    if output_dir is None:
        output_dir = f"./output/{model_name}/visualization"
        
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # 1. Setup Info and Neighbors
    info = _setup_mne_info(dataset)
    neighbors = find_neighbors_from_dataset(dataset)
    
    # 2. Load Data and Analyze
    df = pd.read_csv(csv_path)
    
    # Rule out M1 and M2
    # df = df[~df['Channel'].str.upper().isin(['M1', 'M2'])]
    
    res_df = calculate_uniqueness_metrics(df, config, neighbors)
    
    # 3. Plot Unique Token Count Percentage Grid
    _plot_uniqueness_metric_grid(
        res_df, info, 
        metric_name='Unique_Count_Pct', 
        label='Unique Token Allocation (%)', 
        model_name=model_name, 
        output_dir=output_dir
    )
    
    # 4. Plot Unique Token Weight Percentage Grid
    _plot_uniqueness_metric_grid(
        res_df, info, 
        metric_name='Unique_Weight_Pct', 
        label='Unique Token Contribution (Weight %)', 
        model_name=model_name, 
        output_dir=output_dir
    )
