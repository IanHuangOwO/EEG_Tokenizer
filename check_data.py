import json
import os
import random
import sys
sys.path.append(os.getcwd())

from IO.dataset import build_dataset_from_config
from model.LaBraM.preprocessing import LaBraMProcessing
from utils.visualization import visualize_raw_eeg, visualize_psd_grid, visualize_topo_grid, visualize_band_time_series

def run_viz():
    # Load Config
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    dataset_path = config['dataset_params']['dataset_path']
    with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    config['data_metadata'] = meta['data_metadata']
    
    # Dataset
    fs_orig = meta['data_metadata']['Sample_Frequency']
    transform = LaBraMProcessing(original_freq=fs_orig, target_freq=200, normalization_type='none')
    dataset = build_dataset_from_config(config, transform=transform)
    
    if len(dataset.subject_list) == 0: return
    
    # Pick Random Subject
    # subj = random.choice(dataset.subject_list)
    subj = 3
    print(f"Running visualizations for Subject {subj}...")
    
    # 1. Raw EEG (Single Trial)
    visualize_raw_eeg(dataset, subj)
    
    # 2. PSD Grid (All Classes)
    visualize_psd_grid(dataset, subj, config)
    
    # 3. Topo Grid (All Classes)
    visualize_topo_grid(dataset, subj, config)

    # 4. Band Time Series (Delta, Theta, Alpha, Beta, Gamma)
    # Using 'Oz' as it's a standard visual channel, or 'POz' if Oz is missing.
    # The function defaults to Oz or falls back safely.
    visualize_band_time_series(dataset, subj, channel_label='Oz')
    
if __name__ == "__main__":
    run_viz()
