import os
import json
import torch
import numpy as np
import mne
from mne.preprocessing import ICA
from IO.dataset import build_dataset_from_config
from utils.analysis import _setup_mne_info

def run_ica_on_subject(subject_id=None, n_components=20):
    """
    Loads EEG data for a subject and performs ICA using MNE.
    """
    # 1. Load Configuration
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    # --- Multi-Dataset Selection ---
    target_ds = list(config['dataset_params'].keys())[0]
    for ds_name, ds_args in config['dataset_params'].items():
        if subject_id in ds_args.get('subject_to_use', []):
            target_ds = ds_name
            break
            
    # Load metadata to get sample frequency before building dataset
    data_root = config['dataset_params'][target_ds]['dataset_path']
    with open(os.path.join(data_root, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    fs = meta['data_metadata']['Sample_Frequency']
    
    config['dataset_params'] = {
        target_ds: {
            'dataset_path': data_root,
            'subject_to_use': [subject_id],
            'channels_to_use': config['dataset_params'][target_ds].get('channels_to_use', ["all"])
        }
    }
    
    # 2. Setup Dataset
    dataset = build_dataset_from_config(config, transform=None, mode='base')
    if len(dataset) == 0:
        print(f"No data found for subject {subject_id}")
        return

    # 3. Prepare MNE Object
    data, label = dataset[0]
    data_np = data.numpy().astype(np.float64) # (Nc, Time)
    info = _setup_mne_info(dataset, fs=fs)
    valid_indices = [i for i, name in enumerate(dataset.channel_names) if name in info.ch_names]
    data_np = data_np[valid_indices, :]
    raw = mne.io.RawArray(data_np, info)
    
    # 4. Run ICA
    print("Applying High-pass filter (1.0 Hz) for ICA...")
    raw.filter(l_freq=1.0, h_freq=None)
    print(f"Fitting ICA with {n_components} components...")
    ica = ICA(n_components=n_components, method='fastica', random_state=42, max_iter='auto')
    ica.fit(raw)
    
    # 5. Visualization
    model_name = config['training_params'].get('model_name', 'default_run')
    viz_dir = f"output/{model_name}/visualization/ICA"
    os.makedirs(viz_dir, exist_ok=True)
    fig_topo = ica.plot_components(show=False)
    if isinstance(fig_topo, list):
        for i, fig in enumerate(fig_topo): fig.savefig(os.path.join(viz_dir, f'sub-{subject_id}_ica_components_{i}.png'))
    else:
        fig_topo.savefig(os.path.join(viz_dir, f'sub-{subject_id}_ica_components.png'))
    
    print("ICA Analysis Complete.")
    return ica, raw

if __name__ == "__main__":
    run_ica_on_subject(subject_id=36, n_components=40)
