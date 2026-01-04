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

from IO.dataset import build_dataset_from_config
from model.NeuroRVQ.modeling_tokenizer import NeuroRVQTokenizer
from model.NeuroRVQ.preprocessing import NeuroRVQProcessing

class TokenizerWrapperDatasetWithLabels(Dataset):
    """
    Wraps the standard EEGDataset to yield 1-second patches (200 samples)
    instead of full trials, INCLUDING LABELS and dynamic coordinates.
    """
    def __init__(self, base_dataset, patch_len=200):
        self.base_dataset = base_dataset
        self.patch_len = patch_len
        
        # Check first sample to determine length
        if len(self.base_dataset) > 0:
            sample_x, _ = self.base_dataset[0]
            self.total_len = sample_x.shape[-1]
            self.patches_per_trial = self.total_len // patch_len
        else:
            self.total_len = 0
            self.patches_per_trial = 0
        
        # Pre-convert coords to tensor
        self.coords_tensor = torch.from_numpy(base_dataset.coords).float()
        
        print(f"Tokenizer Dataset: {len(self.base_dataset)} trials, {self.patches_per_trial} patches each.")

    def __len__(self):
        return len(self.base_dataset) * self.patches_per_trial

    def __getitem__(self, index):
        trial_idx = index // self.patches_per_trial
        patch_offset = (index % self.patches_per_trial) * self.patch_len
        
        x, y = self.base_dataset[trial_idx]
        patch = x[:, patch_offset : patch_offset + self.patch_len]
        
        return patch, self.coords_tensor, y

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
    config_path = 'config/config.json'
    ckpt_path = 'output/checkpoints/tokenizer/neurorvq/tokenizer_best.pth'
    output_dir = 'output/visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    m_params = config['model_params']['NeuroRVQ']
    
    # 2. Load Dataset
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
    tokenizer_dataset = TokenizerWrapperDatasetWithLabels(base_dataset, patch_len=200)
    
    # Limit to N samples for t-SNE (it's slow on large N)
    MAX_SAMPLES = 2000 
    data_loader = DataLoader(tokenizer_dataset, batch_size=32, shuffle=True, num_workers=0)

    # 3. Initialize Model
    print(f"Initializing NeuroRVQ Tokenizer for {base_dataset.Nc} channels...")
    
    model = NeuroRVQTokenizer(
        embed_dim=m_params['embed_dim'],
        enc_depth=m_params['enc_depth'],
        enc_heads=m_params['enc_heads'],
        dec_depth=m_params['dec_depth'],
        vocab_size=m_params['vocab_size'],
        freq_resolution=m_params['freq_resolution'],
        min_freq=m_params['min_freq'],
        max_freq=m_params['max_freq']
    )
    
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        
        # Map old rvqs.0 weights to new rvq if needed
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("rvqs.0."):
                new_state_dict[k.replace("rvqs.0.", "rvq.")] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print(f"Checkpoint not found at {ckpt_path}. Running with random weights.")

    model.to(device)
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
    
    save_path = os.path.join(output_dir, 'tsne_input_embeddings.png')
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == '__main__':
    main()
