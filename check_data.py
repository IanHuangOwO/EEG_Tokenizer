import json
import os
import sys
import argparse
import shutil
sys.path.append(os.getcwd())

from IO.dataset import build_dataset_from_config
from utils.visualization import (
    visualize_raw_eeg, visualize_real_imaginary, visualize_amplitude_phase, visualize_psd_grid, visualize_topo_grid, visualize_band_time_series, visualize_masking)

def run_viz():
    # 1. Pre-parse to get the config path
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default='config/check_data.json')
    pre_args, _ = pre_parser.parse_known_args()

    # 2. Load Config
    if not os.path.exists(pre_args.config):
        print(f"Error: Config file not found at {pre_args.config}")
        return
        
    with open(pre_args.config, 'r') as f:
        full_config = json.load(f)
    
    # 3. Setup Main Parser with defaults from config
    parser = argparse.ArgumentParser(description='Check Raw Data')
    parser.add_argument('--config', type=str, default=pre_args.config, help='Path to check_data config file')
    
    # Metadata and output overrides
    cfg_model_name = full_config['training_params'].get('model_name', 'default_check')
    parser.add_argument('--model_name', type=str, default=cfg_model_name, help='Name for the check run (folder name)')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name to visualize')
    parser.add_argument('--subject', type=int, default=None, help='Subject ID override (single)')
    parser.add_argument('--trial_idx', type=int, default=None, help='Trial index to visualize (optional)')
    parser.add_argument('--all', action='store_true', help='Visualize all subjects in metadata')

    # Plot Selection (Defaults from config, or True if missing)
    plot_cfg = full_config.get('plot_selection', {})
    parser.add_argument('--plot_raw', type=bool, default=plot_cfg.get('raw', True), help='Plot raw EEG and metrics')
    parser.add_argument('--plot_masking', type=bool, default=plot_cfg.get('masking', True), help='Plot masking visualization')
    parser.add_argument('--plot_psd', type=bool, default=plot_cfg.get('psd', True), help='Plot PSD grid')
    parser.add_argument('--plot_topo', type=bool, default=plot_cfg.get('topo', True), help='Plot Topography grid')
    
    args = parser.parse_args()
    
    # Use the (possibly overridden) model_name for the directory
    base_viz_dir = os.path.join("output", "data_checks", args.model_name)
    os.makedirs(base_viz_dir, exist_ok=True)
    
    # Copy config for future reference
    shutil.copy(args.config, os.path.join(base_viz_dir, 'check_data_config.json'))

    # Determine datasets to process
    if args.dataset:
        if args.dataset not in full_config['dataset_params']:
            print(f"Dataset {args.dataset} not found in config.")
            return
        target_datasets = [args.dataset]
    else:
        target_datasets = list(full_config['dataset_params'].keys())

    # Determine subjects for each dataset by checking metadata.json
    tasks = [] 
    for ds_name in target_datasets:
        ds_args = full_config['dataset_params'][ds_name]
        ds_path = ds_args['dataset_path']
        meta_path = os.path.join(ds_path, 'metadata.json')
        
        if not os.path.exists(meta_path):
            print(f"Warning: metadata.json not found for {ds_name} at {ds_path}")
            continue
            
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        data_structure = metadata.get('data_structure', {})
        available_subs = sorted([int(k) for k in data_structure.keys()])
        
        if not available_subs:
            continue
        
        if args.all:
            for sub_id in available_subs:
                tasks.append((ds_name, sub_id))
        elif args.subject:
            if args.subject in available_subs:
                tasks.append((ds_name, args.subject))
        elif ds_args.get('subject_to_use'):
            # Support "all" or ["all"] in config
            sub_to_use = ds_args['subject_to_use']
            if sub_to_use == "all" or sub_to_use == ["all"]:
                for sub_id in available_subs:
                    tasks.append((ds_name, sub_id))
            else:
                # Use specific list from config
                for sub_id in sub_to_use:
                    if sub_id in available_subs:
                        tasks.append((ds_name, sub_id))
                    else:
                        print(f"Warning: Subject {sub_id} not found in {ds_name} metadata.")
        else:
            # Default to first one
            tasks.append((ds_name, available_subs[0]))

    if not tasks:
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
        
        viz_dir = os.path.join(base_viz_dir, ds_name, f"sub-{subj}")
        os.makedirs(viz_dir, exist_ok=True)

        try:
            # Load metadata
            meta_path = os.path.join(full_config['dataset_params'][ds_name]['dataset_path'], 'metadata.json')
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            config['data_metadata'] = metadata.get('data_metadata', {})

            # 1. Base Dataset (Raw EEG)
            if args.plot_raw or args.plot_psd or args.plot_topo:
                base_dataset = build_dataset_from_config(config, transform=None, mode='base')
                if len(base_dataset) == 0:
                    print(f"No data found for {ds_name} subject {subj}")
                    continue

                if args.plot_raw:
                    visualize_raw_eeg(base_dataset, subj, output_dir=viz_dir, trial_idx=args.trial_idx)
                    visualize_amplitude_phase(base_dataset, subj, output_dir=viz_dir, trial_idx=args.trial_idx)
                    visualize_real_imaginary(base_dataset, subj, output_dir=viz_dir, trial_idx=args.trial_idx)

                if args.plot_psd:
                    visualize_psd_grid(base_dataset, subj, config, output_dir=viz_dir)
                
                if args.plot_topo:
                    visualize_topo_grid(base_dataset, subj, config, output_dir=viz_dir)
            
            # 2. Masking Visualization
            if args.plot_masking:
                pretrain_ds = build_dataset_from_config(config, transform=None, mode='pretrain')
                visualize_masking(pretrain_ds, subj, output_dir=viz_dir, trial_idx=args.trial_idx)
            
        except Exception as e:
            print(f"Error visualizing {ds_name} subject {subj}: {e}")
            continue
            
        except Exception as e:
            print(f"Error visualizing {ds_name} subject {subj}: {e}")
            continue
    
if __name__ == "__main__":
    run_viz()
