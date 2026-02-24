import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import mne
import torch

# --- 1. Geometry Helpers ---

def get_channel_coordinates(ch_name):
    """
    Returns 2D grid coordinates for a given channel name to help find neighbors.
    Uses a custom mapping for standard EEG positions.
    """
    ch = ch_name.upper()
    if ch == 'M1': return -5, -2.5
    if ch == 'M2': return 5, -2.5
    
    match = re.match(r'([A-Z]+)(\d+|Z)', ch)
    if not match: return 0, 0
    
    row, col = match.groups()
    y_map = {
        'FP': 4, 'AF': 3, 'F': 2, 'FC': 1, 'FT': 1, 'C': 0, 'T': 0, 
        'CP': -1, 'TP': -1, 'P': -2, 'PO': -3, 'O': -4, 'CB': -4.5
    }
    y = y_map.get(row, 0)
    
    if col == 'Z':
        x = 0
    else:
        col_num = int(col)
        sign = 1 if col_num % 2 == 0 else -1
        x = sign * ((col_num + 1) // 2) * 1.5 
        if row in ['T', 'FT', 'TP']:
            x = sign * 5
            
    return x, y

def find_neighbors(channels, threshold=2.8):
    """
    Computes a dictionary mapping each channel to its neighbors based on distance threshold.
    """
    coords = {ch: get_channel_coordinates(ch) for ch in channels}
    neighbors = {}
    
    for c1 in channels:
        x1, y1 = coords[c1]
        n_list = []
        for c2 in channels:
            if c1 == c2: continue
            x2, y2 = coords[c2]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist <= threshold: 
                n_list.append(c2)
        neighbors[c1] = n_list
        
    return neighbors

# --- 2. Core Analysis Logic ---

def calculate_weighted_uniqueness(df, config):
    """
    Calculates the 'Weighted Uniqueness' percentage for each channel across scales.
    Weighted Uniqueness: The sum of weights of tokens chosen by a channel that were NOT chosen by any of its neighbors.
    """
    channels = df['Channel'].unique()
    neighbors = find_neighbors(channels)
    
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
        
        # Mapping for fast lookup
        channel_to_tokens = {}
        channel_to_weights = {}
        
        idx_cols = [f'idx_H{h}_K{k}' for h in range(vq_head_num) for k in range(vq_head_top_k)]
        wt_cols = [f'weight_H{h}_K{k}' for h in range(vq_head_num) for k in range(vq_head_top_k)]

        for ch in channels:
            row = df_s[df_s['Channel'] == ch].iloc[0]
            channel_to_tokens[ch] = row[idx_cols].values
            channel_to_weights[ch] = row[wt_cols].values

        for c1 in channels:
            c1_tokens = channel_to_tokens[c1]
            c1_weights = channel_to_weights[c1]
            
            tokens_c1_set = set(c1_tokens)
            tokens_n_set = set()
            for n in neighbors[c1]:
                tokens_n_set.update(channel_to_tokens[n])
                
            unique_tokens = tokens_c1_set - tokens_n_set
            
            # Calculate Weight %
            unique_weight_sum = sum(w for t, w in zip(c1_tokens, c1_weights) if t in unique_tokens)
            total_weight_sum = sum(c1_weights)
            weight_pct = (unique_weight_sum / total_weight_sum) * 100 if total_weight_sum > 0 else 0
            
            results.append({
                'Channel': c1,
                'Scale': scale_label,
                'Weight_Pct': weight_pct
            })
            
    return pd.DataFrame(results)

# --- 3. Visualization ---

def visualize_weighted_uniqueness_topo(config, csv_path=None, output_dir=None):
    """
    Generates MNE Topographical Heatmaps for weighted uniqueness across all scales.
    """
    model_name = config['training_params']['model_name']
    
    if csv_path is None:
        csv_path = f"./output/{model_name}/visualization/neighbor_codes_with_weights.csv"
    
    if output_dir is None:
        output_dir = f"./output/{model_name}/visualization"
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    res_df = calculate_weighted_uniqueness(df, config)
    channels = df['Channel'].unique()

    # Map custom/variant channel names to standard 10-05 system so MNE recognizes them
    channel_mapping = {
        'CB1': 'PO9', 'CB2': 'PO10',
        'M1': 'TP9',  'M2': 'TP10',
        'FPZ': 'Fpz', 'CPZ': 'CPz', 'PZ': 'Pz', 'OZ': 'Oz', 'FCZ': 'FCz', 'CZ': 'Cz', 'POZ': 'POz', 'FZ':'Fz'
    }
    standardized_channels = [channel_mapping.get(ch.upper(), ch) for ch in channels]

    # Setup MNE info
    info = mne.create_info(ch_names=standardized_channels, sfreq=1000, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1005')
    info.set_montage(montage, match_case=False, on_missing='ignore')

    # Plot setup
    unique_scales = res_df['Scale'].unique()
    n_scales = len(unique_scales)
    fig, axes = plt.subplots(1, n_scales, figsize=(5 * n_scales + 1, 5))
    if n_scales == 1: axes = [axes]

    im = None
    for i, scale in enumerate(unique_scales):
        scale_data = res_df[res_df['Scale'] == scale]
        
        # Create a dictionary to safely map values back to the exact channel order
        ch_val_dict = dict(zip(scale_data['Channel'], scale_data['Weight_Pct']))
        data_values = np.array([ch_val_dict[ch] for ch in channels])
        
        # Generate the Topomap
        im, _ = mne.viz.plot_topomap(
            data_values, 
            info, 
            axes=axes[i], 
            cmap='viridis',
            show=False, 
            sphere='eeglab',
            extrapolate='local'
        )
        axes[i].set_title(f'Scale {scale}: Weighted Uniqueness', fontsize=12)

    # Add Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    if im:
        fig.colorbar(im, cax=cbar_ax, label='Unique Token Allocation (Weighted %)')

    plt.suptitle(f"MNE Topographical Heatmap of Unique Tokens ({model_name})", fontsize=16, y=1.05)
    
    save_path = os.path.join(output_dir, 'weighted_uniqueness_topomap.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")
