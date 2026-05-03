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

def extract_features(model, x, coords):
    """
    Extracts both Transformer Encoder output and VQ results.
    Returns: (enc_out, vq_out) - both (S, B, C, D) or similar
    """
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'attnvq') and hasattr(model, 'spatial_temporal_encoder'):
            # AttnVQ Path
            h_scales_projected = model.spatial_temporal_encoder(x, coords)
            
            h_enc = []
            for i in range(model.patch_len):
                z = h_scales_projected[i]
                enc = model.scale_encoders[i]
                z = enc(z)
                h_enc.append(z)
            
            h_enc = torch.stack(h_enc, dim=0) # (S, B, C, D)
            h_vq, _, _, _ = model.attnvq(h_enc) # (S, B, C, D)
            
            return h_enc, h_vq
        
        # Fallback for other models (NeuroRVQ, RecurrentVQ, etc.)
        # This is a bit more complex as they have different structures.
        # For now, let's focus on supporting AttnVQ specifically as requested.
        print(f"Warning: Model type {type(model).__name__} not fully supported for dual-tSNE. Using default extraction.")
        
        # Simplified fallback logic
        if hasattr(model, 'temporal_encoder'):
             ms_features = model.temporal_encoder(x) 
             spatial_emb = model.spatial_mlp(coords) 
             S = model.patch_len
             B, N, _ = x.shape
             h_all = torch.stack(ms_features, dim=0) + spatial_emb.unsqueeze(0) 
             h_all = h_all.view(S * B, N, -1)
             h_encoded = model.transformer_encoder(h_all)
             h_enc = h_encoded.view(S, B, N, -1)
             
             if hasattr(model, 'rvq'):
                 h_vq, _, _ = model.rvq(h_enc)
             elif hasattr(model, 'fsq'):
                 h_vq, _, _ = model.fsq(h_enc)
             else:
                 h_vq_list = []
                 for i in range(S):
                     z_q, _, _ = model.rvqs[i](h_enc[i])
                     h_vq_list.append(z_q)
                 h_vq = torch.stack(h_vq_list, dim=0)
             return h_enc, h_vq
             
        return None, None

def run_and_save_tsne(X, Y, title, output_path):
    print(f"Running t-SNE for: {title} ({len(X)} tokens)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, verbose=0)
    X_embedded = tsne.fit_transform(X)
    
    # Save Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y, cmap='nipy_spectral', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Class Label')
    plt.title(title)
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

def main():
    # 1. Load Config
    config_path = 'config/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    train_params = config['training_params']
    model_name = train_params.get('model_name', 'default_run')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_SAMPLES = 10000 # Increased for better manifold visualization
    
    # 2. Setup Dataset
    print("Loading dataset for t-SNE analysis...")
    dataset_path = config['dataset_params']['dataset_path']
    with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    config['data_metadata'] = meta['data_metadata']
    
    transform = build_preprocessing_from_config(config)
    tokenizer_dataset = build_dataset_from_config(config, transform=transform, mode='tokenizer')
    data_loader = DataLoader(tokenizer_dataset, batch_size=32, shuffle=True, num_workers=0)

    # 3. Initialize Model
    model = build_model_from_config(config).to(device)

    ckpt_path = f'output/{model_name}/checkpoints/best_model.pth'
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(sd, strict=False)
    else:
        print(f"No checkpoint found at {ckpt_path}. Analyzing RANDOM weights.")
        
    model.eval()

    # 4. Extract Embeddings
    enc_acc = []
    vq_acc = []
    labels_acc = []
    
    print("Extracting features...")
    total_tokens = 0
    
    for batch_x, batch_coords, batch_y in tqdm(data_loader):
        batch_x = batch_x.to(device)
        batch_coords = batch_coords.to(device)
        
        # Get embeddings: (S, B, C, D)
        h_enc, h_vq = extract_features(model, batch_x, batch_coords)
        if h_enc is None: break
        
        # Flatten across S, B, C for t-SNE
        # Current logic: Combine all scales into one t-SNE space
        S, B, C, D = h_enc.shape
        
        enc_flat = h_enc.permute(1, 2, 0, 3).reshape(-1, D).cpu().numpy() # (B*C*S, D)
        vq_flat = h_vq.permute(1, 2, 0, 3).reshape(-1, D).cpu().numpy()   # (B*C*S, D)
        
        # Repeat labels for each scale and channel: (B) -> (B*C*S)
        labels_expanded = batch_y.view(B, 1, 1).expand(B, C, S).reshape(-1).numpy()
        
        enc_acc.append(enc_flat)
        vq_acc.append(vq_flat)
        labels_acc.append(labels_expanded)
        
        total_tokens += B * C * S
        if total_tokens >= MAX_SAMPLES:
            break
            
    X_enc = np.concatenate(enc_acc, axis=0)[:MAX_SAMPLES]
    X_vq = np.concatenate(vq_acc, axis=0)[:MAX_SAMPLES]
    Y = np.concatenate(labels_acc, axis=0)[:MAX_SAMPLES]
    
    # 5. Run t-SNE and Save
    viz_dir = f'output/{model_name}/visualization'
    os.makedirs(viz_dir, exist_ok=True)
    
    # Encoder Output t-SNE
    run_and_save_tsne(
        X_enc, Y, 
        f't-SNE: Transformer Encoder Output ({model_name})', 
        os.path.join(viz_dir, 'tsne_encoder_output.png')
    )
    
    # VQ Results t-SNE
    run_and_save_tsne(
        X_vq, Y, 
        f't-SNE: VQ Results ({model_name})', 
        os.path.join(viz_dir, 'tsne_vq_results.png')
    )

if __name__ == '__main__':
    main()
