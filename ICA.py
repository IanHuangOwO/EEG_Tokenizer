import os
import json
import torch
import numpy as np
import mne
from mne.preprocessing import ICA
from IO.dataset import build_dataset_from_config
from model.factory import build_preprocessing_from_config
from utils.analysis import _setup_mne_info

def run_ica_on_subject(subject_id=None, n_components=20):
    """
    Loads EEG data for a subject and performs ICA using MNE.
    """
    # 1. Load Configuration
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    # 2. Setup Dataset
    # We load the data without preprocessing (transform=None) to apply MNE filters
    config['dataset_params']['subjects'] = [subject_id] if subject_id else [config['dataset_params']['subjects'][0]]
    config['dataset_params']['trials_to_use'] = 1 # Just analyze one trial for speed
    
    # Load Metadata for Sample Frequency
    dataset_path = config['dataset_params']['dataset_path']
    with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    config['data_metadata'] = meta['data_metadata']
    
    fs = config['data_metadata']['Sample_Frequency']
    
    # Build dataset in 'base' mode (full trials)
    dataset = build_dataset_from_config(config, transform=None, mode='base')
    if len(dataset) == 0:
        print(f"No data found for subject {subject_id}")
        return

    # 3. Prepare MNE Object
    # Get a single trial (Channels, Time)
    data, label = dataset[0]
    data_np = data.numpy().astype(np.float64) # (Nc, Time)
    
    # Create MNE Info and set montage using project's existing utility
    info = _setup_mne_info(dataset, fs=fs)
    
    # Note: _setup_mne_info might have filtered out M1/M2 or channels with (0,0,0) coords.
    # We need to ensure the data matches the info.ch_names
    valid_indices = [i for i, name in enumerate(dataset.channel_names) if name in info.ch_names]
    data_np = data_np[valid_indices, :]
    
    # Create RawArray
    raw = mne.io.RawArray(data_np, info)
    
    # 4. Pre-ICA Filtering (High-pass is essential for ICA stability)
    print("Applying High-pass filter (1.0 Hz) for ICA...")
    raw.filter(l_freq=1.0, h_freq=None)
    
    # 5. Run ICA
    print(f"Fitting ICA with {n_components} components...")
    ica = ICA(n_components=n_components, method='fastica', random_state=42, max_iter='auto')
    ica.fit(raw)
    
    # 6. Visualization
    model_name = config['training_params'].get('model_name', 'default_run')
    viz_dir = f"output/{model_name}/visualization/ICA"
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"Generating ICA visualizations in: {viz_dir}")
    
    # Plot Component Topographies
    fig_topo = ica.plot_components(show=False)
    
    # MNE plot_components can return a single figure or a list of figures
    if isinstance(fig_topo, list):
        for i, fig in enumerate(fig_topo):
            fig.savefig(os.path.join(viz_dir, f'sub-{subject_id}_ica_components_{i}.png'))
    else:
        fig_topo.savefig(os.path.join(viz_dir, f'sub-{subject_id}_ica_components.png'))
    
    # Plot Component Properties (e.g., first 5)
    # ica.plot_properties(raw, picks=range(min(5, n_components)), show=False)
    
    # 7. (Optional) Reconstruct and Compare
    # You can manually exclude components here if you find artifacts
    # ica.exclude = [0, 1] # Example: Eye blinks
    # raw_clean = ica.apply(raw.copy())
    
    print("ICA Analysis Complete.")
    return ica, raw

if __name__ == "__main__":
    # Example: Run on subject 36 (common in this project)
    # Increase n_components if you have many channels
    ica_obj, raw_obj = run_ica_on_subject(subject_id=36, n_components=40)
