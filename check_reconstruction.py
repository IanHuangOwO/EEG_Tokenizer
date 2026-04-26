import json
import os
import argparse
import torch
import numpy as np
import mne
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from IO.dataset import build_dataset_from_config
from model.factory import build_model_from_config
from utils.visualization import visualize_raw_eeg, visualize_psd_grid, visualize_topo_grid, visualize_band_time_series

class ReconstructedDatasetWrapper:
    def __init__(self, original_dataset, reconstructed_data):
        self.data = reconstructed_data
        self.labels = original_dataset.labels
        self.subject_data = original_dataset.subject_data
        self.Nc = original_dataset.Nc
        self.channel_names = original_dataset.channel_names
        self.coords = original_dataset.coords
        self.subject_list = original_dataset.subject_list
        self.datasets = [True] # Dummy to pass check_data check

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

def main():
    parser = argparse.ArgumentParser(description='Check Reconstructed Data')
    parser.add_argument('--config', type=str, default='config/config.json')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--subject', type=int, default=36, help='Subject ID to visualize')
    parser.add_argument('--mask_ratio', type=float, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # --- Multi-Dataset Selection ---
    target_ds = list(config['dataset_params'].keys())[0]
    for ds_name, ds_args in config['dataset_params'].items():
        if args.subject in ds_args.get('subject_to_use', []):
            target_ds = ds_name
            break
            
    config['dataset_params'] = {
        target_ds: {
            'dataset_path': config['dataset_params'][target_ds]['dataset_path'],
            'subject_to_use': [args.subject],
            'channels_to_use': config['dataset_params'][target_ds].get('channels_to_use', ["all"])
        }
    }
    
    mask_ratio = args.mask_ratio if args.mask_ratio is not None else config.get('model_params', {}).get(config['training_params'].get('model_type', 'AttnVQ'), {}).get('preprocess', {}).get('mask_ratio', 0.0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = config['training_params'].get('model_name', 'default_run')
    if args.checkpoint is None:
        args.checkpoint = f"output/{model_name}/tokenizer/best_tokenizer.pth"
    
    base_dataset = build_dataset_from_config(config, transform=None, mode='base')
    model = build_model_from_config(config).to(device)
    
    # Warmup
    model_type = config['training_params'].get('model_type', 'AttnVQ')
    patch_len = config.get('model_params', {}).get(model_type, {}).get('preprocess', {}).get('patch_length', 100)
    # Expected model input: [B, C, N, L]
    dummy_x = torch.randn(1, base_dataset.Nc, 1, patch_len).to(device)
    dummy_coords = base_dataset.coords.unsqueeze(0).to(device) # [1, Nc, 3]
    dummy_time = torch.zeros(1, 1).to(device)
    model.eval()
    with torch.no_grad(): model(dummy_x, dummy_coords, dummy_time)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    reconstructed_tensors = []
    trial_len = base_dataset[0][0].shape[-1]
    num_patches = trial_len // patch_len
    loader = DataLoader(base_dataset, batch_size=1, shuffle=False)
    coords_tensor = base_dataset.coords.float().to(device).unsqueeze(0)

    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Reconstructing"):
            x = x.to(device)
            B, C, T = x.shape
            # AttnVQ expects [B, C_in, N, L]. Since in_chans=1, we treat C as batch dim.
            x_reshaped = x.view(B, C, num_patches, patch_len) # [1, Nc, num_patches, 100]
            if mask_ratio > 0:
                mask = torch.rand(B, C, num_patches, device=device) < mask_ratio
                x_reshaped = x_reshaped.masked_fill(mask.unsqueeze(-1), 0.0)
            
            # Prepare for model: [Batch*Nc, 1, num_patches, patch_len]
            x_in = x_reshaped.reshape(B * C, 1, num_patches, patch_len)
            coords_in = coords_tensor.expand(B, -1, -1).reshape(B * C, 1, 3)
            time_idx_in = torch.zeros(B * C, num_patches).to(device)
            
            p1, p2, _, _, _ = model(x_in, coords_in, time_idx_in)
            
            # p1, p2 are [B*C, 1, num_patches, F]
            x_recon_patched = model.reconstruct(p1, p2, n_samples=T)
            # x_recon_patched: [B*C, 1, T]
            x_recon = x_recon_patched.reshape(B, C, T)
            reconstructed_tensors.append(x_recon.cpu())

    recon_data = torch.cat(reconstructed_tensors, dim=0)
    recon_dataset = ReconstructedDatasetWrapper(base_dataset, recon_data)
    
    viz_dir = f"output/{model_name}/visualization/reconstruction_analysis"
    os.makedirs(viz_dir, exist_ok=True)
    visualize_raw_eeg(recon_dataset, args.subject, output_dir=viz_dir)
    visualize_psd_grid(recon_dataset, args.subject, config, output_dir=viz_dir)
    visualize_topo_grid(recon_dataset, args.subject, config, output_dir=viz_dir)
    visualize_band_time_series(recon_dataset, args.subject, channel_label='Oz', output_dir=viz_dir)

if __name__ == "__main__":
    main()
