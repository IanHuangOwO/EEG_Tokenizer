import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure we can import from local modules
sys.path.append(os.getcwd())

from IO.dataset import build_dataset_from_config
from model.factory import build_model_from_config, build_preprocessing_from_config

def get_latent_embeddings(model, x, coords):
    """
    Runs the encoder part of the model to get the combined quantized embeddings.
    """
    model.eval()
    with torch.no_grad():
        # 1. Extract Features
        # Note: This logic assumes Neuro/Recurrent architectures. LaBraM is different.
        if hasattr(model, 'patch_embed'): # LaBraM
             x_emb = model.patch_embed(x)
             h = model.encoder(x_emb)
             z = model.encode_task_layer(h)
             z_q, _, _ = model.quantize(z)
             return z_q
             
        # NeuroRVQ / RecurrentVQ / RecurrentFSQ
        ms_features = model.temporal_encoder(x) 
        spatial_emb = model.spatial_mlp(coords) 
        
        S = model.in_scales
        B, N, _ = x.shape
        
        # Combine scales
        h_all = torch.stack(ms_features, dim=0) + spatial_emb.unsqueeze(0) 
        h_all = h_all.view(S * B, N, -1)
        
        h_encoded = model.transformer_encoder(h_all)
        h_scales = h_encoded.view(S, B, N, -1)
        
        # Quantize
        if hasattr(model, 'fsq'):
             # RecurrentFSQ
             all_z_q, _, _ = model.fsq(h_scales)
        elif hasattr(model, 'rvq'):
             # RecurrentVQ
             all_z_q, _, _ = model.rvq(h_scales)
        else:
             # NeuroRVQ
             all_z_q = []
             for i in range(S):
                 z_q, _, _ = model.rvqs[i](h_scales[i])
                 all_z_q.append(z_q)
             all_z_q = torch.stack(all_z_q, dim=0)
            
        # Fusion
        z_fused = torch.sum(all_z_q, dim=0)
        
    return z_fused

def main():
    # 1. Load Config
    config_path = 'config/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    train_params = config['training_params']
    model_name = train_params.get('model_name', 'default_run')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_SAMPLES = 2000
    
    # 2. Setup Dataset (using a small subset for speed)
    print("Loading dataset for usage analysis...")
    dataset_path = config['dataset_params']['dataset_path']
    with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    config['data_metadata'] = meta['data_metadata']
    
    transform = build_preprocessing_from_config(config)
    
    tokenizer_dataset = build_dataset_from_config(config, transform=transform, mode='tokenizer')
    data_loader = DataLoader(tokenizer_dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=4)

    # 3. Initialize Model and Load Checkpoint via Factory
    model = build_model_from_config(config).to(device)

    ckpt_path = f'output/{model_name}/checkpoints/best_model.pth'
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        print(f"No checkpoint found at {ckpt_path}. Analyzing RANDOM weights.")
        
    model.eval()

    # 4. Extract Embeddings
    all_embeddings = []
    all_labels = []
    
    print("Extracting embeddings...")
    total_tokens = 0
    
    for batch_x, batch_coords, batch_y in tqdm(data_loader):
        batch_x = batch_x.to(device)
        batch_coords = batch_coords.to(device)
        
        # Get embeddings: (B, N, D)
        embeddings = get_latent_embeddings(model, batch_x, batch_coords)
        
        # Flatten channels: (B*N, D)
        B, N, D = embeddings.shape
        embeddings_flat = embeddings.view(-1, D).cpu().numpy()
        
        # Repeat labels for each channel: (B) -> (B, N) -> (B*N)
        labels_expanded = batch_y.unsqueeze(1).repeat(1, N).view(-1).numpy()
        
        all_embeddings.append(embeddings_flat)
        all_labels.append(labels_expanded)
        
        total_tokens += B * N
        if total_tokens >= MAX_SAMPLES:
            break
            
    X = np.concatenate(all_embeddings, axis=0)
    Y = np.concatenate(all_labels, axis=0)
    
    # Trim to max
    if len(X) > MAX_SAMPLES:
        X = X[:MAX_SAMPLES]
        Y = Y[:MAX_SAMPLES]
        
    print(f"Collected {len(X)} tokens. Running t-SNE...")
    
    # 5. Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, verbose=1)
    X_embedded = tsne.fit_transform(X)
    
    # 6. Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y, cmap='nipy_spectral', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Class Label')
    plt.title(f't-SNE of {model_name} Latent Space ({len(X)} tokens)')
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.tight_layout()
    
    viz_dir = f'output/{model_name}/visualization'
    os.makedirs(viz_dir, exist_ok=True)
    output_path = os.path.join(viz_dir, 'tsne_latent_space.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    main()
