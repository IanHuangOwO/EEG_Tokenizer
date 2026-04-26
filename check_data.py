import json
import os
import sys
import argparse
sys.path.append(os.getcwd())

from IO.dataset import build_dataset_from_config
from utils.visualization import (
    visualize_raw_eeg, visualize_real_imaginary, visualize_amplitude_phase, visualize_psd_grid, visualize_topo_grid, visualize_band_time_series, visualize_masking)

def run_viz():
    parser = argparse.ArgumentParser(description='Check Raw Data')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name to visualize (e.g., BETA, Dial)')
    parser.add_argument('--subject', type=int, default=None, help='Subject ID to visualize')
    parser.add_argument('--all', action='store_true', help='Visualize all subjects in all datasets')
    args = parser.parse_args()

    # Load Config
    with open('config/config.json', 'r') as f:
        full_config = json.load(f)
    
    model_name = full_config['training_params'].get('model_name', 'default_run')
    base_viz_dir = f"output/{model_name}/visualization/data_analysis"

    # Determine tasks
    tasks = [] # List of (ds_name, subject_id)
    
    if args.all:
        for ds_name, ds_args in full_config['dataset_params'].items():
            for sub_id in ds_args.get('subject_to_use', []):
                tasks.append((ds_name, sub_id))
    else:
        # Determine datasets to process
        if args.dataset:
            if args.dataset not in full_config['dataset_params']:
                print(f"Dataset {args.dataset} not found in config.")
                return
            target_datasets = [args.dataset]
        else:
            target_datasets = list(full_config['dataset_params'].keys())

        # Determine subjects for each dataset
        for ds_name in target_datasets:
            ds_args = full_config['dataset_params'][ds_name]
            available_subs = ds_args.get('subject_to_use', [])
            if not available_subs:
                continue
            
            if args.subject:
                if args.subject in available_subs:
                    tasks.append((ds_name, args.subject))
            else:
                # Default to the first subject of the dataset
                tasks.append((ds_name, available_subs[0]))

        if not tasks:
            if args.subject:
                print(f"Subject {args.subject} not found in processed datasets.")
            else:
                print("No datasets or subjects found to process.")
            return

    print(f"Starting visualization for {len(tasks)} tasks...")

    for ds_name, subj in tasks:
        print(f"\n>>> Processing Dataset: {ds_name} | Subject: {subj} <<<")
        
        # Create a temporary config for this specific subject/dataset
        config = full_config.copy()
        config['dataset_params'] = {
            ds_name: {
                'dataset_path': full_config['dataset_params'][ds_name]['dataset_path'],
                'subject_to_use': [subj],
                'channels_to_use': full_config['dataset_params'][ds_name].get('channels_to_use', ["all"])
            }
        }
        
        # Dedicated output folder
        viz_dir = os.path.join(base_viz_dir, ds_name, f"sub-{subj}")
        os.makedirs(viz_dir, exist_ok=True)

        # Dataset loading
        try:
            base_dataset = build_dataset_from_config(config, transform=None, mode='base')
            # Trigger summary log
            build_dataset_from_config(config, transform=None, mode='tokenizer')
            # For masking visualization
            pretrain_ds = build_dataset_from_config(config, transform=None, mode='pretrain')
            
            if len(base_dataset) == 0: 
                print(f"No data found for {ds_name} subject {subj}")
                continue
            
            # Load metadata
            meta_path = os.path.join(full_config['dataset_params'][ds_name]['dataset_path'], 'metadata.json')
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            config['data_metadata'] = metadata.get('data_metadata', {})

            # 1. Raw EEG (Single Trial)
            visualize_raw_eeg(base_dataset, subj, output_dir=viz_dir)
            visualize_amplitude_phase(base_dataset, subj, output_dir=viz_dir)
            visualize_real_imaginary(base_dataset, subj, output_dir=viz_dir)
            
            # 2. Masking Visualization
            visualize_masking(pretrain_ds, subj, output_dir=viz_dir)

            # 3. PSD Grid (All Classes)
            visualize_psd_grid(base_dataset, subj, config, output_dir=viz_dir)
            
            # 4. Topo Grid (All Classes)
            visualize_topo_grid(base_dataset, subj, config, output_dir=viz_dir)
            
        except Exception as e:
            print(f"Error visualizing {ds_name} subject {subj}: {e}")
            continue
    
if __name__ == "__main__":
    run_viz()
