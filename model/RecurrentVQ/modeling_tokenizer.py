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
    Now includes an internal Top-Down Bridge (FPN) to fuse scales.
    """
    # [FIX] Added input_length parameter to remove hardcoded '200'
    def __init__(self, in_chans=1, embed_dim=200, base_filters=8, num_scales=4, input_length=200):
        super().__init__()
        self.num_scales = num_scales
        self.in_chans = in_chans
        
        # --- 1. The Bottom-Up Pathway (Your original CNNs) ---
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
            
            # Project everyone to the same 'embed_dim' so they can be summed/fused
            # Use an MLP to gradually compress the dimension (Input -> Hidden -> Embed)
            input_dim = current_filters * current_t
            hidden_dim = max(input_dim // 2, embed_dim)
            
            self.projections.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            ))


        # --- 2. The Top-Down Pathway (The New Bridge) ---
        # We need (num_scales - 1) adapters because the deepest scale has no parent.
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(), 
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_scales - 1)
        ])
        
        # Learnable Gates: Initialize to 0 so we start with pure features
        self.gates = nn.Parameter(torch.zeros(num_scales - 1)) 


    def forward(self, x):
        if x.dim() == 4:
            B, N, C, T = x.shape
        else:
            B, N, T = x.shape
            C = 1
            
        x = x.view(B * N, self.in_chans, T)
        x = self.stem(x)
        
        # --- Phase 1: Extract Raw Scales (Bottom-Up) ---
        raw_features = []
        feat = x
        for layer, proj in zip(self.layers, self.projections):
            feat = layer(feat)
            flat = feat.flatten(1) 
            emb = proj(flat).view(B, N, -1) 
            raw_features.append(emb)
        
        # raw_features = [S0, S1, S2, S3] (Fine -> Coarse)
        
        # --- Phase 2: Fuse Scales (Top-Down) ---
        # Start with the deepest/coarsest scale (S3)
        context = raw_features[-1] 
        
        # Create output list, pre-filled with the deepest scale (it doesn't change)
        refined_features = [None] * self.num_scales
        refined_features[-1] = context
        
        # Loop backwards: S2 -> S1 -> S0
        for i in range(self.num_scales - 2, -1, -1):
            target = raw_features[i]
            
            # 1. Adapt Context (S_i+1) for Scale S_i
            context_proj = self.adapters[i](context)
            
            # 2. Gated Injection
            gate = torch.sigmoid(self.gates[i])
            fused = target + (gate * context_proj)
            
            refined_features[i] = fused
            
            # 3. Update context for the next step down
            context = fused

        return refined_features

# --- 2. Transformer Encoder ---

class TransformerEncoder(nn.Module):
    """
    Standard Transformer Encoder.
    Processes features across the channel dimension.
    """
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

# --- 3. Vectorized Multi-Scale Residual VQ ---

class VectorizedMultiScaleRVQ(nn.Module):
    """
    Vectorized Recurrent VQ that processes all scales in parallel using SEPARATE codebooks per scale.
    Input: (Num_Scales, Batch, Channels, Dim)
    """
    def __init__(self, num_scales, num_recurrent_steps, n_e, e_dim, beta=0.25):
        super().__init__()
        self.num_scales = num_scales
        self.num_recurrent_steps = num_recurrent_steps
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        # Separate Codebooks: (S, N_E, D)
        self.embedding = nn.Parameter(torch.empty(num_scales, n_e, e_dim))
        self.embedding.data.normal_(mean=0.0, std=0.02)

    def forward(self, z):
        # z: (S, B, N, D)
        S, B, N, D = z.shape
        
        # Flatten Batch and Channels for vectorized processing
        # z_flat: (S, B*N, D)
        z_flat = z.view(S, B * N, D)
        
        quantized_out = 0
        residual = z_flat
        total_loss = 0
        all_indices = []

        # Precompute codebook squared norm: (S, N_E)
        # Transpose to (S, 1, N_E) for broadcasting
        codebook_sq = torch.sum(self.embedding ** 2, dim=2, keepdim=True).transpose(1, 2)

        for _ in range(self.num_recurrent_steps):
            # 1. Calculate Distances (Vectorized across S and B*N)
            # z_sq: (S, B*N, 1)
            z_sq = torch.sum(residual ** 2, dim=2, keepdim=True)
            
            # term2: (S, B*N, D) @ (S, D, N_E) -> (S, B*N, N_E)
            term2 = 2 * torch.matmul(residual, self.embedding.transpose(1, 2))
            
            # d: (S, B*N, N_E)
            d = z_sq + codebook_sq - term2
            
            # 2. Find Nearest Neighbors
            indices = torch.argmin(d, dim=2) # (S, B*N)
            
            # 3. Quantize
            # Gather embeddings: (S, B*N, D)
            ind_expanded = indices.unsqueeze(-1).expand(-1, -1, self.e_dim)
            z_q = torch.gather(self.embedding, 1, ind_expanded)
            
            # 4. Loss
            loss = torch.mean((z_q.detach() - residual) ** 2) + self.beta * torch.mean((z_q - residual.detach()) ** 2)
            
            # 5. Straight-Through Estimator
            z_q = residual + (z_q - residual).detach()
            
            residual = residual - z_q
            quantized_out = quantized_out + z_q
            total_loss += loss
            all_indices.append(indices)

        # Reshape output back to (S, B, N, D)
        quantized_out = quantized_out.view(S, B, N, D)
        
        # Stack indices: (S, B, N, Steps)
        all_indices = torch.stack(all_indices, dim=-1).view(S, B, N, -1)

        return quantized_out, total_loss, all_indices

# --- 4. Main Tokenizer Model ---

class RecurrentVQTokenizer(nn.Module):
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
        num_recurrent_steps=8, 
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
        
        # Precompute Frequency Mask (Optimization)
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
        
        # 2. Transformer Encoder (Shared)
        self.transformer_encoder = TransformerEncoder(embed_dim, enc_depth, enc_heads, mlp_ratio=enc_mlp_ratio)
        
        # 3. Vectorized Multi-Scale RVQ (Parallelized)
        self.rvq = VectorizedMultiScaleRVQ(
            num_scales=num_scales,
            num_recurrent_steps=num_recurrent_steps,
            n_e=vocab_size,
            e_dim=embed_dim
        )
        
        # 4. Decoder
        self.transformer_decoder = TransformerEncoder(embed_dim, dec_depth, dec_heads, mlp_ratio=dec_mlp_ratio) 
        
        self.head_amp = nn.Linear(embed_dim, self.fft_dim)
        self.head_sin = nn.Linear(embed_dim, self.fft_dim)
        self.head_cos = nn.Linear(embed_dim, self.fft_dim)

    def forward(self, x, coords):
        if x.dim() == 4:
            B, N, C, T = x.shape
        else:
            B, N, T = x.shape
            C = 1
        S = self.num_scales
        
        # 1. Extract Multi-Scale Features
        ms_features = self.temporal_encoder(x) 
        spatial_emb = self.spatial_mlp(coords) 
        
        # 2. Shared Transformer Pass (Batch Optimized)
        # Combine scales and add spatial embeddings
        h_all = torch.stack(ms_features, dim=0) + spatial_emb.unsqueeze(0) # (S, B, N, D)
        h_all = h_all.view(S * B, N, -1)
        
        h_encoded = self.transformer_encoder(h_all)
        h_scales = h_encoded.view(S, B, N, -1)
        
        # 3. Vectorized VQ Pass (Process all scales at once)
        all_z_q, total_vq_loss, all_indices = self.rvq(h_scales)
            
        # 4. Latent Fusion
        z_fused = torch.sum(all_z_q, dim=0)
        
        # 5. Decoder
        dec_h = self.transformer_decoder(z_fused)
        
        pred_amp = self.head_amp(dec_h)
        pred_sin = self.head_sin(dec_h)
        pred_cos = self.head_cos(dec_h)
        
        return pred_amp, pred_sin, pred_cos, total_vq_loss, all_indices

    def get_loss(self, x, pred_amp, pred_sin, pred_cos, x_fft=None):
        # x: (B, N, 200)
        # Compute Ground Truth FFT with target resolution
        # n_fft determines the frequency spacing
        if x_fft is None:
            x_fft = torch.fft.rfft(x, n=self.n_fft, dim=-1) # (B, N, n_fft//2 + 1)
        
        # Filter GT using precomputed mask
        target_fft = x_fft[..., self.freq_mask]
        
        # Check shapes (Debugging safety)
        if target_fft.shape[-1] != pred_amp.shape[-1]:
             # If mismatch (e.g. due to rounding), simple interpolation or cropping
             # For now, let's assume calc is correct, or slice to match
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
        
        # 2. Phase Loss (Amplitude-Weighted Cosine Similarity)
        # Target vectors
        target_sin = torch.sin(gt_phase)
        target_cos = torch.cos(gt_phase)
        
        # Phase Loss using MSE on Sin/Cos components
        loss_phase = F.mse_loss(pred_sin, target_sin) + F.mse_loss(pred_cos, target_cos)
        
        # 3. Temporal Loss
        # Differentiable Reconstruction using the same logic as inference
        x_recon = self.reconstruct(pred_amp, pred_sin, pred_cos, n_samples=x.shape[-1])
        
        loss_temp = F.l1_loss(x_recon, x)
        
        # Total Loss
        total_loss = loss_amp + loss_phase + loss_temp
        
        return total_loss, loss_amp, loss_phase, loss_temp

    def reconstruct(self, pred_amp, pred_sin, pred_cos, n_samples=200, x=None):
        """
        Reconstructs the time-domain signal from predicted coefficients.
        """
        # 1. Recover Amplitude
        amp = torch.exp(pred_amp) - 1
        amp = torch.clamp(amp, min=0) # Ensure non-negative
        
        # 2. Normalize Phase
        pred_norm = torch.sqrt(pred_cos**2 + pred_sin**2 + 1e-8)
        norm_cos = pred_cos / pred_norm
        norm_sin = pred_sin / pred_norm

        # 3. Form Complex Coefficients (Predicted Part)
        real = amp * norm_cos
        imag = amp * norm_sin
        z_pred = torch.complex(real, imag)
        
        # 4. Embed into Full Spectrum
        # We need to construct the full spectrum (size n_fft//2 + 1)
        # and fill the indices corresponding to [min_freq, max_freq]
        full_fft_dim = self.n_fft // 2 + 1
        full_z = torch.zeros((z_pred.shape[0], z_pred.shape[1], full_fft_dim), dtype=z_pred.dtype, device=z_pred.device)
        
        # Use precomputed indices
        indices = self.freq_indices
        count = min(len(indices), z_pred.shape[-1])
        full_z[..., indices[:count]] = z_pred[..., :count]
        
        # 5. Inverse Real FFT
        # Reconstruct to n_fft length first (padded length)
        x_recon_padded = torch.fft.irfft(full_z, n=self.n_fft, dim=-1)
        
        # 6. Crop to original length
        x_recon = x_recon_padded[..., :n_samples]
        
        return x_recon

    # --- Analysis Helpers ---

    def get_codebooks(self):
        """
        Returns (tensor, name) tuples.
        RecurrentVQ shares codebooks across steps (L), but we return entries for each L
        to analyze usage per step.
        """
        codebooks = []
        # self.rvq.embedding: (S, N_E, D)
        emb = self.rvq.embedding
        steps = self.rvq.num_recurrent_steps
        
        for s in range(emb.shape[0]):
            cb = emb[s].detach().cpu()
            for l in range(steps):
                name = f"S{s}_L{l}_H0"
                codebooks.append((cb, name))
                
        return codebooks

    def get_indices(self, x, coords):
        """
        Returns usage indices.
        Target: (Batch*N, L(Steps), S, H=1, K=1)
        """
        with torch.no_grad():
            # forward returns (..., all_indices)
            # all_indices: (S, B, N, Steps)
            _, _, _, _, all_indices = self.forward(x, coords)
        
        # Permute: (S, B, N, Steps) -> (B, N, Steps, S)
        indices = all_indices.permute(1, 2, 3, 0)
        
        # Flatten Batch*N -> (T, L, S)
        indices_flat = indices.flatten(0, 1)
        
        # Expand for H=1, K=1 -> (T, L, S, 1, 1)
        return indices_flat.unsqueeze(-1).unsqueeze(-1)