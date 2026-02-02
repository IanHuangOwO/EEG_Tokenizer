import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Multi-Scale Temporal Encoder ---

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels) 
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += res
        x = self.act(x)
        return x

class MultiScaleTemporalEncoder(nn.Module):
    """
    Extracts features at multiple time-scales and fuses them via a Top-Down Bridge.
    """
    def __init__(self, in_chans=1, embed_dim=200, base_filters=8, num_scales=4, input_length=200):
        super().__init__()
        self.num_scales = num_scales
        
        # --- 1. The Bottom-Up Pathway ---
        self.stem = nn.Sequential(
            nn.Conv1d(in_chans, base_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.GELU()
        )
        
        self.layers = nn.ModuleList()
        self.projections = nn.ModuleList()
        
        current_filters = base_filters
        current_t = input_length 
        
        for i in range(num_scales):
            stride = 1 if i == 0 else 2
            out_filters = current_filters if i == 0 else current_filters * 2
            
            self.layers.append(ResBlock1D(current_filters, out_filters, stride=stride))
            
            current_filters = out_filters
            current_t = (current_t - 1) // stride + 1
            
            # Simplified Single Linear Projection
            input_dim = current_filters * current_t
            
            self.projections.append(nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            ))

        # --- 2. The Top-Down Pathway ---
        self.adapters = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim)
            for _ in range(num_scales - 1)
        ])
        
        # Learnable Gates
        self.gates = nn.Parameter(torch.zeros(num_scales - 1)) 

    def forward(self, x):
        B, N, T = x.shape
        x = x.view(B * N, 1, T)
        x = self.stem(x)
        
        # Phase 1: Extract Raw Scales (Bottom-Up)
        raw_features = []
        feat = x
        for layer, proj in zip(self.layers, self.projections):
            feat = layer(feat)
            flat = feat.flatten(1) 
            emb = proj(flat).view(B, N, -1) 
            raw_features.append(emb)
        
        # Phase 2: Fuse Scales (Top-Down)
        context = raw_features[-1] 
        refined_features = [None] * self.num_scales
        refined_features[-1] = context
        
        for i in range(self.num_scales - 2, -1, -1):
            target = raw_features[i]
            context_proj = self.adapters[i](context)
            gate = torch.sigmoid(self.gates[i])
            fused = target + (gate * context_proj)
            refined_features[i] = fused
            context = fused

        return refined_features

# --- 2. Transformer Encoder ---

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                       dim_feedforward=int(embed_dim * mlp_ratio), 
                                       dropout=drop_rate, activation='gelu', batch_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

# --- 3. AttnVQ ---

class AttnVQ(nn.Module):
    """
    AttnVQ: Replaces Iterative RVQ with Parallel Sparse Attention.
    
    Mechanics:
    1. Similarity: Dot Product (Attention)
    2. Selection: Top-K
    3. Reconstruction: Weighted Sum (Softmax weights)
    4. Regularization: Orthogonal Loss
    """
    def __init__(self, num_scales, n_e, e_dim, top_k=8, temperature=1.0):
        super().__init__()
        self.num_scales = num_scales
        self.n_e = n_e
        self.e_dim = e_dim
        self.top_k = top_k
        
        # Learnable Temperature (controls how "sharp" the weights are)
        # Start at 1.0, let the model tune it.
        self.temperature = nn.Parameter(torch.ones(num_scales, 1, 1) * temperature)

        # Separate Codebooks for each scale: (S, N_E, D)
        self.embedding = nn.Parameter(torch.empty(num_scales, n_e, e_dim))
        nn.init.xavier_uniform_(self.embedding)

    def forward(self, z):
        # z: (S, B, N, D)
        S, B, N, D = z.shape
        
        # Flatten Batch and Channels: (S, B*N, D)
        z_flat = z.view(S, B * N, D)
        
        # --- 1. Attention Scores (Dot Product) ---
        # (S, B*N, D) @ (S, D, N_E) -> (S, B*N, N_E)
        logits = torch.matmul(z_flat, self.embedding.transpose(1, 2))
        
        # --- 2. Top-K Selection (Sparse) ---
        # indices: (S, B*N, K)
        # top_vals: (S, B*N, K)
        top_vals, indices = torch.topk(logits, k=self.top_k, dim=-1)
        
        # --- 3. Soft-Weighted Mixing ---
        # Apply Softmax to get weights summing to 1.0 (over the K selected)
        weights = F.softmax(top_vals / self.temperature, dim=-1) # (S, B*N, K)
        
        # --- 4. Reconstruction ---
        # We need to select from self.embedding (S, N_E, D) using indices (S, B*N, K)
        
        # Efficient Gather Strategy:
        # 1. View embedding as (S * N_E, D)
        # 2. Adjust indices to point to the flattened embedding
        scale_offsets = torch.arange(S, device=z.device).view(S, 1, 1) * self.n_e
        flat_indices = (indices + scale_offsets).view(-1, self.top_k) # (S*B*N, K)
        flat_embedding = self.embedding.view(-1, D) # (S*N_E, D)
        
        # Gather: (S*B*N, K, D)
        selected_vectors = F.embedding(flat_indices, flat_embedding)
        
        # Reshape back to (S, B*N, K, D)
        selected_vectors = selected_vectors.view(S, B * N, self.top_k, D)
        
        # Weighted Sum: 
        # Weights: (S, B*N, K) -> (S, B*N, 1, K)
        # Vectors: (S, B*N, K, D)
        # Matmul: (1, K) @ (K, D) -> (1, D)
        z_q = torch.matmul(weights.unsqueeze(2), selected_vectors).squeeze(2) # (S, B*N, D)
        
        # --- 5. Losses ---
        # A. Commitment Loss (Standard VQ)
        # Move encoder output (z) towards the weighted mixture (z_q)
        # And move the mixture towards the encoder (to train codebook)
        loss_commit = F.mse_loss(z_q.detach(), z_flat) + 0.25 * F.mse_loss(z_q, z_flat.detach())
        
        # B. Orthogonal Regularization Loss (CRITICAL)
        loss_ortho = self.orthogonal_loss()
        
        # Combine losses
        total_loss = loss_commit + (0.1 *loss_ortho)

        # Reshape outputs
        z_q = z_q.view(S, B, N, D)
        indices = indices.view(S, B, N, self.top_k) # Save these Top-K indices for storage

        return z_q, total_loss, indices

    def orthogonal_loss(self, threshold=0.1):
        """
        Forces codebook vectors within each scale to be quasi-orthogonal.
        Uses a threshold hinge loss to allow for packing when N_E > D.
        """
        # 1. Normalize: (S, N_E, D)
        vectors_norm = F.normalize(self.embedding, p=2, dim=-1)
        
        # 2. Gram Matrix: (S, N_E, N_E)
        gram = torch.matmul(vectors_norm, vectors_norm.transpose(1, 2))
        
        # 3. Off-diagonal Mask
        I = torch.eye(self.n_e, device=gram.device).unsqueeze(0)
        off_diag = gram * (1 - I)
        
        # 4. Thresholded Loss
        # Penalize only if similarity > threshold
        loss = F.relu(off_diag.abs() - threshold).pow(2).mean()
        
        return loss

# --- 4. Main Tokenizer Model (AttnVQTokenizer) ---

class AttnVQTokenizer(nn.Module):
    def __init__(
        self,
        in_chans=1,
        embed_dim=200,
        enc_depth=4,
        enc_heads=10,
        enc_mlp_ratio=4.,
        dec_depth=3, 
        dec_heads=10,
        dec_mlp_ratio=4.,
        num_scales=4,
        top_k=8,
        vocab_size=512,
        freq_resolution=1.0,
        min_freq=0.0,
        max_freq=100.0,
        fs=200.0,
        input_length=200 
    ):
        super().__init__()
        
        self.num_scales = num_scales
        self.embed_dim = embed_dim
        self.freq_resolution = freq_resolution
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.fs = fs
        self.input_length = input_length 
        
        self.fft_dim = int(round((max_freq - min_freq) / freq_resolution)) + 1 
        self.n_fft = int(self.fs / freq_resolution)
        
        # Precompute Frequency Mask
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0/self.fs)
        mask = (freqs >= self.min_freq - 1e-5) & (freqs <= self.max_freq + 1e-5)
        self.register_buffer('freq_mask', mask)
        self.register_buffer('freq_indices', torch.where(mask)[0])
        
        # 1. Temporal Encoder
        self.temporal_encoder = MultiScaleTemporalEncoder(
            in_chans, embed_dim, num_scales=num_scales, input_length=input_length
        )
        
        # Fixed Spatial Embeddings
        self.spatial_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        nn.init.zeros_(self.spatial_mlp[-1].weight)
        nn.init.zeros_(self.spatial_mlp[-1].bias)
        
        # 2. Transformer Encoder
        self.transformer_encoder = TransformerEncoder(embed_dim, enc_depth, enc_heads, mlp_ratio=enc_mlp_ratio)
        
        # 3. AttnVQ Module
        self.attnvq = AttnVQ(
            num_scales=num_scales,
            n_e=vocab_size,
            e_dim=embed_dim,
            top_k=top_k
        )
        
        # 4. Decoder
        self.transformer_decoder = TransformerEncoder(embed_dim, dec_depth, dec_heads, mlp_ratio=dec_mlp_ratio) 
        
        self.head_amp = nn.Linear(embed_dim, self.fft_dim)
        self.head_sin = nn.Linear(embed_dim, self.fft_dim)
        self.head_cos = nn.Linear(embed_dim, self.fft_dim)

    def forward(self, x, coords):
        B, N, T = x.shape
        S = self.num_scales
        
        # 1. Extract Multi-Scale Features
        ms_features = self.temporal_encoder(x) 
        spatial_emb = self.spatial_mlp(coords) 
        
        # 2. Shared Transformer Pass
        h_all = torch.stack(ms_features, dim=0) + spatial_emb.unsqueeze(0) # (S, B, N, D)
        h_all = h_all.view(S * B, N, -1)
        
        h_encoded = self.transformer_encoder(h_all)
        h_scales = h_encoded.view(S, B, N, -1)
        
        # 3. AttnVQ Pass
        all_z_q, vq_loss, top_k_indices = self.attnvq(h_scales)
            
        # 4. Latent Fusion
        z_fused = torch.sum(all_z_q, dim=0)
        
        # 5. Decoder
        dec_h = self.transformer_decoder(z_fused)
        
        pred_amp = self.head_amp(dec_h)
        pred_sin = self.head_sin(dec_h)
        pred_cos = self.head_cos(dec_h)
        
        return pred_amp, pred_sin, pred_cos, vq_loss, top_k_indices

    def get_loss(self, x, pred_amp, pred_sin, pred_cos, x_fft=None):
        # x: (B, N, 200)
        if x_fft is None:
            x_fft = torch.fft.rfft(x, n=self.n_fft, dim=-1) # (B, N, n_fft//2 + 1)
        
        target_fft = x_fft[..., self.freq_mask]
        
        # Check shapes
        if target_fft.shape[-1] != pred_amp.shape[-1]:
             min_len = min(target_fft.shape[-1], pred_amp.shape[-1])
             target_fft = target_fft[..., :min_len]
             pred_amp = pred_amp[..., :min_len]
             pred_sin = pred_sin[..., :min_len]
             pred_cos = pred_cos[..., :min_len]
             
        gt_amp = torch.abs(target_fft)
        gt_phase = torch.angle(target_fft)
        
        # 1. Log-Amplitude Loss
        target_log_amp = torch.log1p(gt_amp)
        loss_amp = F.mse_loss(pred_amp, target_log_amp)
        
        # 2. Phase Loss
        target_sin = torch.sin(gt_phase)
        target_cos = torch.cos(gt_phase)
        loss_phase = F.mse_loss(pred_sin, target_sin) + F.mse_loss(pred_cos, target_cos)
        
        # 3. Temporal Loss
        x_recon = self.reconstruct(pred_amp, pred_sin, pred_cos, n_samples=x.shape[-1])
        loss_temp = F.l1_loss(x_recon, x)
        
        # Total Reconstruction Loss
        # Note: You must add the 'vq_loss' from forward() to this in your training loop!
        total_recon_loss = loss_amp + loss_phase + loss_temp
        
        return total_recon_loss, loss_amp, loss_phase, loss_temp

    def reconstruct(self, pred_amp, pred_sin, pred_cos, n_samples=200, x=None):
        """
        Reconstructs the time-domain signal from predicted coefficients.
        """
        # 1. Recover Amplitude
        amp = torch.exp(pred_amp) - 1
        amp = torch.clamp(amp, min=0) 
        
        # 2. Normalize Phase
        pred_norm = torch.sqrt(pred_cos**2 + pred_sin**2 + 1e-8)
        norm_cos = pred_cos / pred_norm
        norm_sin = pred_sin / pred_norm

        # 3. Form Complex Coefficients
        real = amp * norm_cos
        imag = amp * norm_sin
        z_pred = torch.complex(real, imag)
        
        # 4. Embed into Full Spectrum
        full_fft_dim = self.n_fft // 2 + 1
        full_z = torch.zeros((z_pred.shape[0], z_pred.shape[1], full_fft_dim), dtype=z_pred.dtype, device=z_pred.device)
        
        indices = self.freq_indices
        count = min(len(indices), z_pred.shape[-1])
        full_z[..., indices[:count]] = z_pred[..., :count]
        
        # 5. Inverse Real FFT
        x_recon_padded = torch.fft.irfft(full_z, n=self.n_fft, dim=-1)
        
        # 6. Crop to original length
        x_recon = x_recon_padded[..., :n_samples]
        
        return x_recon

    # --- Analysis Helpers ---

    def get_codebooks(self):
        """
        Returns a list of (tensor, name) tuples for analysis.
        Extracts embedding from AttnVQ.
        Name format: "S{scale}"
        """
        codebooks = []
        # embedding: (S, N_E, D)
        for s in range(self.num_scales):
            cb = self.attnvq.embedding[s].detach().cpu()
            name = f"S{s}"
            codebooks.append((cb, name))
                    
        return codebooks

    def get_indices(self, x, coords):
        """
        Runs forward pass and returns indices for analysis.
        Returns: (Batch*N, Depth=1, Scales, Heads=1, Top-K) flat tensor structure logic.
        AttnVQ indices: (S, B, N, K)
        """
        # Run forward pass (ignoring gradients)
        with torch.no_grad():
            _, _, _, _, indices = self.forward(x, coords)
        
        # Indices: (S, B, N, K)
        # Permute to (B, N, S, K)
        indices = indices.permute(1, 2, 0, 3) 
        
        # Flatten Batch and N -> (Batch*N, S, K)
        indices_flat = indices.reshape(-1, indices.shape[2], indices.shape[3])
        
        # Reshape to 5D for compatibility: (Batch*N, L=1, S, H=1, K)
        indices_flat = indices_flat.unsqueeze(1).unsqueeze(3)
        
        return indices_flat