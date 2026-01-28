import os
import json
import torch
import numpy as np
import pandas as pd
from IO.dataset import build_dataset_from_config
from model.factory import build_model_from_config, build_preprocessing_from_config

def main():
    # 1. Load Config
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    # 2. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. Create Preprocessor and Dataset
    # We want one subject and minimal trials for this check
    config['dataset_params']['subjects'] = [config['dataset_params']['subjects'][0]]
    config['dataset_params']['trials_to_use'] = 1
    
    # Load Metadata to get Sample_Frequency
    dataset_path = config['dataset_params']['dataset_path']
    with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    config['data_metadata'] = meta['data_metadata']
    
    transform = build_preprocessing_from_config(config)
    
    dataset = build_dataset_from_config(config, transform=transform)
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
    # TokenizerWrapperDataset returns (patch, coords, label)
    patch, coords, _ = dataset[0]
    
    # Add batch dimension and move to device
    patch = patch.unsqueeze(0).to(device)   # (1, Channels, 200)
    coords = coords.unsqueeze(0).to(device) # (1, Channels, 3)

    # 6. Forward Pass
    with torch.no_grad():
        # Model returns: pred_amp, pred_sin, pred_cos, vq_loss, indices
        outputs = model(patch, coords)
        indices = outputs[-1] # (S, B, N, Steps)
    
    indices = indices.squeeze(1).cpu().numpy() # (S, N, Steps)
    
    S, N, R = indices.shape
    print(f"Indices shape: Scales={S}, Channels={N}, Steps={R}")

    # 7. Create DataFrame and Save to CSV
    
    # Flatten indices: (Channels, S * R)
    # We want rows to be channels, columns to be scales/steps
    flat_indices = indices.transpose(1, 0, 2).reshape(N, S * R)
    
    # Create Column Names
    columns = []
    for s in range(S):
        for r in range(R):
            columns.append(f"S{s}_R{r}")
            
    df = pd.DataFrame(flat_indices, columns=columns)
    
    # Add Channel Names
    channel_names = dataset.base_dataset.channel_names
    # If channel names list is shorter or longer (unlikely), handle gracefully
    if len(channel_names) == N:
        df.insert(0, "Channel", channel_names)
    else:
        print(f"Warning: Number of channel names ({len(channel_names)}) does not match data channels ({N}). Using indices.")
        df.insert(0, "Channel_Idx", range(N))
        
    save_path = os.path.join(viz_dir, 'neighbor_codes.csv')
    df.to_csv(save_path, index=False)
    print(f"CSV saved to {save_path}")

if __name__ == "__main__":
    main()
