import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure we can import from local modules
sys.path.append(os.getcwd())

from IO.dataset import build_dataset_from_config, TokenizerWrapperDatasetWithLabels
from model.factory import build_model_from_config
from model.NeuroRVQ.preprocessing import NeuroRVQProcessing

def get_latent_embeddings(model, x, coords):
    """
    Runs the encoder part of NeuroRVQTokenizer to get the combined quantized embeddings.
    Matches the updated 'Late Fusion' architecture.
    """
    model.eval()
    with torch.no_grad():
        # 1. Extract Multi-Scale Features
        ms_features = model.temporal_encoder(x) 
        spatial_emb = model.spatial_mlp(coords) # (B, N, D)
        
        all_z_q = []
        
        # 2. Encode each scale separately
        for i, feat in enumerate(ms_features):
            h = feat + spatial_emb
            h = model.transformer_encoder(h)
            
            # RVQ for this specific scale
            z_q, _, _ = model.rvqs[i](h)
            all_z_q.append(z_q)
            
        # 3. Combine Scales using summation
        z_fused = torch.sum(torch.stack(all_z_q, dim=0), dim=0)
        
    return z_fused

def main():
    # 1. Load Config
    config_path = 'config/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    m_params = config['model_params']['NeuroRVQ']
    train_params = config['training_params']
    model_type = train_params.get('model_type', 'NeuroRVQ')
    model_name = train_params.get('model_name', 'neurorvq_v1')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_SAMPLES = 2000
    
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
    tokenizer_dataset = TokenizerWrapperDatasetWithLabels(base_dataset, patch_len=200) # patch_len = 1 sec
    data_loader = DataLoader(tokenizer_dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=4)

    # 3. Initialize Model and Load Checkpoint via Factory
    model = build_model_from_config(config).to(device)

    ckpt_path = f'output/checkpoints/tokenizer/{model_name}/tokenizer_best.pth'
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
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
    plt.title(f't-SNE of NeuroRVQ Latent Space ({len(X)} tokens)')
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.tight_layout()
    
    output_path = f'output/visualization/{model_name}/tsne_input_embeddings.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    main()
