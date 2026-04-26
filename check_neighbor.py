import os
import json
import torch
import numpy as np
import pandas as pd
from IO.dataset import build_dataset_from_config
from model.factory import build_model_from_config, build_preprocessing_from_config
from utils.analysis import visualize_weighted_uniqueness_topo

def main():
    # 1. Load Config
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    # 2. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. Create Preprocessor and Dataset
    # We want one subject and minimal trials for this check
    config['dataset_params']['subject_to_use'] = [config['dataset_params']['subject_to_use'][20]]
    
    transform = build_preprocessing_from_config(config)
    
    dataset = build_dataset_from_config(config, transform=transform, mode='tokenizer')
    print(f"Dataset loaded: {len(dataset)} trials")

    # 4. Create Model
    model = build_model_from_config(config).to(device)
    
    # Check for checkpoint
    model_name = config['training_params']['model_name']
    checkpoint_dir = f"output/{model_name}/checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    
    # Define visualization directory within the model folder
    viz_dir = f"output/{model_name}/visualization"
    os.makedirs(viz_dir, exist_ok=True)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully.")
    else:
        print("--------------------------------------------------")
        print(f"WARNING: No checkpoint found at {checkpoint_path}")
        print("Using RANDOM initialized weights.")
        print("--------------------------------------------------")

    model.eval()

    # 5. Get a sample (One trial/patch)
    patch_idx = 0
    patch, coords, time_idx, _, _ = dataset[patch_idx]
    
    # Logic to find which Subject this belongs to
    # dataset is a TokenizerDataset, dataset.base_dataset is EEGDataset
    trial_idx = patch_idx // dataset.patches_per_trial
    subject_id = dataset.base_dataset.subject_data[trial_idx].item()
    
    print(f"Analyzing Patch {patch_idx} (Trial {trial_idx}) from Subject {subject_id}")
    
    # Add batch dimension and move to device
    patch = patch.unsqueeze(0).to(device)   # (1, Channels, 200)
    coords = coords.unsqueeze(0).to(device) # (1, Channels, 3)
    time_idx = torch.tensor([time_idx], device=device) # (1,)

    # 6. Forward Pass
    with torch.no_grad():
        # NOTE: get_indices now returns (indices, weights) tuple
        outputs = model.get_indices(patch, coords, time_idx)
        indices = outputs[0] # (B*C, L=1, S, H, K)
        weights = outputs[1] # (B*C, L=1, S, H, K)
    
    # Indices and Weights should be (B*C, L=1, S, H, K)
    # Long Format: rows = (B*C * S), columns = (Channel, Scale, idx_H0_K0...idx_H7_K7, weight_H0_K0...weight_H7_K7)
    
    # Squeeze L=1 dimension
    indices = indices.squeeze(1).cpu().numpy() # (BC, S, H, K)
    weights = weights.squeeze(1).cpu().numpy() # (BC, S, H, K)
    
    BC, S, H, K = indices.shape
    print(f"Indices shape: B*C={BC}, S={S}, H={H}, K={K}")

    # 7. Create DataFrame in Long Format
    channel_names = dataset.base_dataset.channel_names
    if len(channel_names) != BC:
        print(f"Warning: Channel names ({len(channel_names)}) mismatch data ({BC}). Using indices.")
        channel_names = [f"Ch{i}" for i in range(BC)]

    # Column names for one scale
    idx_columns = [f"idx_H{h}_K{k}" for h in range(H) for k in range(K)]
    weight_columns = [f"weight_H{h}_K{k}" for h in range(H) for k in range(K)]
    
    all_rows = []
    for s_idx in range(S):
        scale_label = f"S{s_idx}"
        for bc_idx in range(BC):
            row = {
                "Channel": channel_names[bc_idx],
                "Scale": scale_label
            }
            # Flatten H and K for this scale and channel
            idx_vals = indices[bc_idx, s_idx].flatten()
            wt_vals = weights[bc_idx, s_idx].flatten()
            
            for i, col in enumerate(idx_columns):
                row[col] = idx_vals[i]
            for i, col in enumerate(weight_columns):
                row[col] = wt_vals[i]
                
            all_rows.append(row)
            
    df = pd.DataFrame(all_rows)
    
    save_path = os.path.join(viz_dir, 'neighbor_codes_long.csv')
    df.to_csv(save_path, index=False)
    print(f"CSV saved to {save_path}")

    # 8. Run Topographical Analysis
    visualize_weighted_uniqueness_topo(config, dataset, csv_path=save_path, output_dir=viz_dir)

if __name__ == "__main__":
    main()
