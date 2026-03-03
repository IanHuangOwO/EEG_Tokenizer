import json
import os
import random
import sys
sys.path.append(os.getcwd())

from IO.dataset import build_dataset_from_config
from model.factory import build_preprocessing_from_config
from utils.visualization import visualize_raw_eeg, visualize_psd_grid, visualize_topo_grid, visualize_band_time_series

def run_viz():
    # Load Config
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    dataset_path = config['dataset_params']['dataset_path']
    with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    config['data_metadata'] = meta['data_metadata']
    
    # Model/Output Info
    model_name = config['training_params'].get('model_name', 'default_run')
    viz_dir = f"output/{model_name}/visualization/data_analysis"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Dataset
    transform = build_preprocessing_from_config(config)
    dataset = build_dataset_from_config(config, transform=transform, mode='base')
    
    if len(dataset.subject_list) == 0: return
    
    # Pick Random Subject
    # subj = random.choice(dataset.subject_list)
    subj = 36
    print(f"Running visualizations for Subject {subj}...")
    
    # 1. Raw EEG (Single Trial)
    visualize_raw_eeg(dataset, subj, output_dir=viz_dir)
    
    # 2. PSD Grid (All Classes)
    visualize_psd_grid(dataset, subj, config, output_dir=viz_dir)
    
    # 3. Topo Grid (All Classes)
    visualize_topo_grid(dataset, subj, config, output_dir=viz_dir)

    # 4. Band Time Series (Delta, Theta, Alpha, Beta, Gamma)
    # Using 'Oz' as it's a standard visual channel, or 'POz' if Oz is missing.
    # The function defaults to Oz or falls back safely.
    visualize_band_time_series(dataset, subj, channel_label='Oz', output_dir=viz_dir)
    
if __name__ == "__main__":
    run_viz()
