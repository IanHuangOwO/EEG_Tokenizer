import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleTemporalEncoder(nn.Module):
    """
    Fast Parallel Multi-Scale Encoder.
    Different kernel sizes and strides per scale.
    Features are fused via a lightweight parameter-free FPN (CumSum).
    Heavier interaction is left to the Transformer.
    """
    def __init__(self, in_chans=1, embed_dim=200, base_filters=8, num_scales=4, input_length=200):
        super().__init__()
        self.num_scales = num_scales
        self.branches = nn.ModuleList()
        self.projections = nn.ModuleList()
        
        # Kernel sizes: 3, 7, 15, 31, 63...
        kernel_sizes = [3, 7, 15, 31, 63]
        
        for i in range(num_scales):
            # Scale-specific parameters
            k = kernel_sizes[min(i, len(kernel_sizes)-1)]
            stride = 2**i # 1, 2, 4, 8...
            padding = (k - 1) // 2
            
            # Simple Conv Block
            # We use a single robust convolution per scale
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_chans, base_filters * (2**i), kernel_size=k, stride=stride, padding=padding, bias=False),
                nn.BatchNorm1d(base_filters * (2**i)),
                nn.GELU()
            ))
            
            # Calculate output length for projection
            # L_out = floor((L_in + 2*pad - dilation*(kernel-1) - 1)/stride + 1)
            # With stride S and padding P=(K-1)/2:
            # L_out ~ L_in / S
            # Exact math:
            out_len = (input_length + 2*padding - k) // stride + 1
            # Adjust for integer division quirks if needed, but standard padding usually keeps alignment
            # For strict alignment, we calculate input_dim dynamically or force padding
            # Here using calculated dimension:
            flat_dim = (base_filters * (2**i)) * out_len
            
            self.projections.append(nn.Sequential(
                nn.Linear(flat_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            ))

    def forward(self, x):
        B, N, T = x.shape
        x = x.view(B * N, 1, T)
        
        raw_features = []
        
        # Parallel Execution (Python loop matches layer list, but ops are independent)
        for branch, proj in zip(self.branches, self.projections):
            feat = branch(x)
            flat = feat.flatten(1)
            emb = proj(flat).view(B, N, -1)
            raw_features.append(emb)
            
        # Lightweight FPN: Cumulative Sum from Coarse to Fine
        # Stack: (S, B, N, D)
        # Flip (S-1 to 0), Cumsum, Flip back
        # This allows coarse scales (high stride) to inject info into fine scales (low stride)
        # cheaply without learnable parameters.
        ms_features = torch.stack(raw_features, dim=0)
        ms_features = ms_features.flip(0).cumsum(dim=0).flip(0)
        
        # Return as list for compatibility with rest of pipeline
        return list(ms_features.unbind(0))

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
    AttnVQ: Multi-Head Sparse Attention VQ.
    
    Mechanics:
    1. Similarity: Dot Product (Attention) per Head
    2. Selection: Top-K per Head
    3. Reconstruction: Weighted Sum (Softmax weights)
    4. Regularization: Orthogonal Loss per Head
    """
    def __init__(self, num_scales, num_heads, vocab_size, e_dim, top_k=8, temperature=1.0):
        super().__init__()
        assert e_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.num_scales = num_scales
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.e_dim = e_dim
        self.head_dim = e_dim // num_heads
        self.top_k = top_k
        
        # Learnable Temperature
        # (S, H, 1, 1)
        self.temperature = nn.Parameter(torch.ones(num_scales, num_heads, 1, 1) * temperature)

        # Codebooks: (S, H, Vocab_Size, Head_Dim)
        self.embedding = nn.Parameter(torch.empty(num_scales, num_heads, vocab_size, self.head_dim))
        nn.init.xavier_uniform_(self.embedding)

    def forward(self, z):
        # z: (S, B, N, D)
        S, B, N, D = z.shape
        H = self.num_heads
        D_h = self.head_dim
        
        # Reshape for Multi-Head: (S, B, N, H, D_h)
        z_reshaped = z.view(S, B, N, H, D_h)
        
        # --- 1. Attention Scores (Dot Product) ---
        # z: (S, B, N, H, D_h)
        # emb: (S, H, V, D_h)
        # Einsum: sbnhd,shvd -> sbnhv
        logits = torch.einsum('sbnhd,shvd->sbnhv', z_reshaped, self.embedding)
        
        # --- 2. Top-K Selection (Sparse) ---
        # indices: (S, B, N, H, K)
        top_vals, indices = torch.topk(logits, k=self.top_k, dim=-1)
        
        # --- 3. Soft-Weighted Mixing ---
        # Temp: (S, H, 1, 1) -> (S, 1, 1, H, 1) to match (S, B, N, H, K)
        temp_broadcast = self.temperature.view(S, 1, 1, H, 1)
        weights = F.softmax(top_vals / temp_broadcast, dim=-1)
        
        # --- 4. Reconstruction ---
        # Gather vectors
        # Indices: (S, B, N, H, K)
        # We need to flatten to gather efficiently or use advanced indexing
        
        # Offset strategy for independent heads
        # S*H*V total vectors
        # Scale offset: s * (H*V)
        # Head offset: h * V
        s_idx = torch.arange(S, device=z.device).view(S, 1, 1, 1, 1)
        h_idx = torch.arange(H, device=z.device).view(1, 1, 1, H, 1)
        
        flat_offset = s_idx * (H * self.vocab_size) + h_idx * self.vocab_size
        flat_indices = (indices + flat_offset).view(-1, self.top_k) # (S*B*N*H, K)
        
        flat_embedding = self.embedding.view(-1, D_h) # (S*H*V, D_h)
        
        # Gather: (S*B*N*H, K, D_h)
        selected_vectors = F.embedding(flat_indices, flat_embedding)
        
        # Reshape: (S, B, N, H, K, D_h)
        selected_vectors = selected_vectors.view(S, B, N, H, self.top_k, D_h)
        
        # Weighted Sum: (S, B, N, H, D_h)
        # weights: (S, B, N, H, K) -> (..., K, 1)
        z_q_head = torch.sum(weights.unsqueeze(-1) * selected_vectors, dim=-2)
        
        # Concatenate Heads: (S, B, N, D)
        z_q = z_q_head.view(S, B, N, D)
        
        # --- 5. Losses ---
        loss_commit = F.mse_loss(z_q.detach(), z) + 0.25 * F.mse_loss(z_q, z.detach())
        loss_ortho = self.orthogonal_loss()
        
        total_loss = loss_commit + loss_ortho

        # Indices output for analysis: (S, B, N, H, K)
        return z_q, total_loss, indices

    def orthogonal_loss(self, threshold=0.1):
        """
        Orthogonal loss per head.
        """
        # (S, H, V, D_h)
        vectors_norm = F.normalize(self.embedding, p=2, dim=-1)
        
        # Gram: (S, H, V, V)
        gram = torch.matmul(vectors_norm, vectors_norm.transpose(-1, -2))
        
        # Identity
        I = torch.eye(self.vocab_size, device=gram.device).view(1, 1, self.vocab_size, self.vocab_size)
        off_diag = gram * (1 - I)
        
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
        vq_heads=None, # New param, defaults to enc_heads if None
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
        self.vocab_size = vocab_size
        self.vq_heads = vq_heads if vq_heads is not None else enc_heads
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
            num_heads=self.vq_heads,
            vocab_size=vocab_size,
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
        Name format: "S{scale}_H{head}"
        """
        codebooks = []
        # embedding: (S, H, N_E, D_h)
        for s in range(self.num_scales):
            for h in range(self.vq_heads):
                cb = self.attnvq.embedding[s, h].detach().cpu()
                name = f"S{s}_H{h}"
                codebooks.append((cb, name))
                    
        return codebooks

    def get_indices(self, x, coords):
        """
        Runs forward pass and returns indices for analysis.
        Returns: (Batch*N, Depth=1, Scales, Heads, Top-K) flat tensor structure logic.
        AttnVQ indices: (S, B, N, H, K)
        """
        # Run forward pass (ignoring gradients)
        with torch.no_grad():
            _, _, _, _, indices = self.forward(x, coords)
        
        # Indices: (S, B, N, H, K)
        # Permute to (B, N, S, H, K)
        indices = indices.permute(1, 2, 0, 3, 4) 
        
        # Flatten Batch and N -> (Batch*N, S, H, K)
        indices_flat = indices.reshape(-1, indices.shape[2], indices.shape[3], indices.shape[4])
        
        # Reshape to 5D for compatibility: (Batch*N, L=1, S, H, K)
        indices_flat = indices_flat.unsqueeze(1)
        
        return indices_flat