import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from IO.dataset import build_dataset_from_config
from model.factory import build_model_from_config
from model.NeuroRVQ.preprocessing import NeuroRVQProcessing
from tqdm import tqdm

def calculate_perplexity():
    # 1. Load Config
    config_path = 'config/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    m_params = config['model_params']['NeuroRVQ']
    train_params = config['training_params']
    model_type = train_params.get('model_type', 'NeuroRVQ')
    model_name = train_params.get('model_name', 'neurorvq_v1')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Setup Dataset (using a small subset for speed)
    print("Loading dataset for usage analysis...")
    dataset_path = config['dataset_params']['dataset_path']
    with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    config['data_metadata'] = meta['data_metadata']
    
    fs_orig = meta['data_metadata']['Sample_Frequency']
    transform = NeuroRVQProcessing(
        original_freq=fs_orig, 
        target_freq=200, 
        l_freq=0.1, 
        h_freq=80.0, 
        normalization_type='zscore'
    )
    
    base_dataset = build_dataset_from_config(config, transform=transform)
    # We only need about 100 trials to get a stable usage estimate
    data_loader = DataLoader(base_dataset, batch_size=16, shuffle=True)

    # 3. Initialize Model and Load Checkpoint via Factory
    model = build_model_from_config(config).to(device)

    ckpt_path = f'output/checkpoints/tokenizer/{model_name}/tokenizer_best.pth'
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    else:
        print(f"No checkpoint found at {ckpt_path}. Analyzing RANDOM weights.")

    model.eval()

    # 4. Track Usage
    n_layers = m_params['num_codebooks']
    vocab_size = m_params['vocab_size']
    
    # Track counts for Layer 0 across all scales (or just Scale 0 for simplicity)
    # We'll focus on Scale 0, but collect usage for all RVQ layers within it
    usage_counts = [torch.zeros(vocab_size, device=device) for _ in range(n_layers)]
    
    # We need to hook into the RVQ layers to get indices
    indices_list = []

    def hook_fn(module, input, output):
        # output of ResidualVQ is (quantized, loss, indices)
        # indices shape: (B, N, n_layers)
        indices_list.append(output[2].detach())

    # Attach hook to the first scale's RVQ
    handle = model.rvqs[0].register_forward_hook(hook_fn)

    print("Running inference to collect usage statistics...")
    coords = torch.from_numpy(base_dataset.coords).float().to(device)
    
    # Limit to 50 batches for analysis
    max_batches = 50
    with torch.no_grad():
        for i, (batch_x, _) in enumerate(tqdm(data_loader)):
            if i >= max_batches: break
            
            # EEG signal: (B, Channels, Time)
            # We need to take a 1-second patch to match tokenizer expectation
            patch = batch_x[..., :200].to(device)
            
            # Forward pass (triggers hook)
            model(patch, coords)
            
            # Process collected indices
            if indices_list:
                indices = indices_list.pop() # (B, N, n_layers)
                for l in range(n_layers):
                    l_indices = indices[..., l].flatten()
                    usage_counts[l].put_(l_indices, torch.ones_like(l_indices, dtype=torch.float), accumulate=True)

    handle.remove()

    # 5. Calculate Perplexity
    perplexities = []
    print("\n--- Perplexity Report (Scale 0) ---")
    for l in range(n_layers):
        counts = usage_counts[l]
        total = counts.sum()
        if total == 0:
            perplexities.append(0.0)
            continue
            
        probs = counts / total
        # Filter out zeros to avoid log(0)
        probs = probs[probs > 0]
        entropy = -torch.sum(probs * torch.log(probs))
        perplexity = torch.exp(entropy).item()
        perplexities.append(perplexity)
        
        print(f"Layer {l+1}: {perplexity:.2f} / {vocab_size} ({(perplexity/vocab_size)*100:.1f}%)")

    # 6. Visualization
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_layers + 1), perplexities, color='teal', alpha=0.8)
    plt.axhline(y=vocab_size, color='r', linestyle='--', label='Max Possible')
    plt.title("Codebook Perplexity per Layer (Effective Vocabulary Size)")
    plt.xlabel("RVQ Layer")
    plt.ylabel("Perplexity")
    plt.xticks(range(1, n_layers + 1))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    output_path = f'output/visualization/{model_name}/codebook_perplexity.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    calculate_perplexity()
