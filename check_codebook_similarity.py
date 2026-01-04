import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from model.NeuroRVQ.modeling_tokenizer import NeuroRVQTokenizer

def check_codebook():
    # 1. Load Config and Model
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    m_params = config['model_params']['NeuroRVQ']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Pass ALL params to match the checkpoint's expected head size
    model = NeuroRVQTokenizer(
        embed_dim=m_params['embed_dim'],
        enc_depth=m_params['enc_depth'],
        enc_heads=m_params['enc_heads'],
        dec_depth=m_params['dec_depth'],
        vocab_size=m_params['vocab_size'],
        freq_resolution=m_params['freq_resolution'],
        min_freq=m_params['min_freq'],
        max_freq=m_params['max_freq']
    ).to(device)

    # Load checkpoint
    ckpt_path = 'output/checkpoints/tokenizer/neurorvq/tokenizer_best.pth'
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location=device)
        
        # Smart Map: If names changed from rvqs.0.layers.0 to rvq.layers.0
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("rvqs.0."):
                new_state_dict[k.replace("rvqs.0.", "rvq.")] = v
            else:
                new_state_dict[k] = v
        
        # Load with strict=False to ignore architecture-wide mismatches
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded with partial match. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
    else:
        print("No checkpoint found. Showing INITIAL state.")

    # 2. Extract First Codebook
    # NeuroRVQ -> self.rvq (ResidualVQ) -> self.layers[0] (VectorQuantizer) -> self.embedding
    codebook = model.rvq.layers[0].embedding.weight.data # (vocab_size, embed_dim)
    
    # 3. Create a Random Gaussian Vector
    random_vec = torch.randn(1, m_params['embed_dim']).to(device)
    
    # 4. Compute Cosine Similarity for Random Vector
    similarity = F.cosine_similarity(random_vec, codebook)
    similarity = similarity.cpu().numpy()

    # 5. Multi-Layer Correlation Analysis
    cb1 = model.rvq.layers[0].embedding.weight.data
    cb1_norm = F.normalize(cb1, p=2, dim=1)
    
    # 5. Full Inter-Layer Correlation Matrix
    all_cb_norms = []
    print("Normalizing all 8 codebooks...")
    for i in range(8):
        cb = model.rvq.layers[i].embedding.weight.data
        all_cb_norms.append(F.normalize(cb, p=2, dim=1))
    
    corr_matrix = np.zeros((8, 8))
    
    print("Computing 8x8 correlation matrix...")
    for i in range(8):
        for j in range(8):
            if i == j:
                # Self-correlation (average of abs off-diagonal)
                sample_indices = torch.randperm(m_params['vocab_size'])[:512]
                self_sim = torch.matmul(all_cb_norms[i][sample_indices], all_cb_norms[i][sample_indices].t())
                # Mask out the diagonal (which is 1.0)
                mask = ~torch.eye(512, dtype=torch.bool)
                corr_matrix[i, j] = torch.mean(torch.abs(self_sim[mask])).item()
            else:
                # Cross-correlation
                sample_indices = torch.randperm(m_params['vocab_size'])[:512]
                cross_sim = torch.matmul(all_cb_norms[i][sample_indices], all_cb_norms[j][sample_indices].t())
                corr_matrix[i, j] = torch.mean(torch.abs(cross_sim)).item()

    # 6. Visualization
    plt.figure(figsize=(12, 10))
    
    im = plt.imshow(corr_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, label='Avg Abs Cosine Similarity')
    
    # Annotate values
    for i in range(8):
        for j in range(8):
            plt.text(j, i, f"{corr_matrix[i, j]:.3f}", ha="center", va="center", 
                     color="white" if corr_matrix[i, j] < 0.2 else "black")
            
    plt.title("Inter-Codebook Correlation Matrix (8x8 Layers)")
    plt.xlabel("Codebook Layer Index")
    plt.ylabel("Codebook Layer Index")
    plt.xticks(range(8), range(1, 9))
    plt.yticks(range(8), range(1, 9))

    plt.tight_layout()
    output_path = 'output/visualization/codebook_full_matrix.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Analysis saved to {output_path}")

if __name__ == "__main__":
    check_codebook()
