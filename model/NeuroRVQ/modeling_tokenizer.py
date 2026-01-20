import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Multi-Scale Temporal Encoder ---

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(4, out_channels) 
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(4, out_channels)
        
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

class TemporalEncoder(nn.Module):
    """
    Dynamically extracts features from EEG patches across multiple scales.
    """
    def __init__(self, in_chans=1, embed_dim=200, base_filters=8, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_chans, base_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(4, base_filters),
            nn.GELU()
        )
        
        self.layers = nn.ModuleList()
        self.projections = nn.ModuleList()
        
        current_filters = base_filters
        current_t = 200 
        
        for i in range(num_scales):
            stride = 1 if i == 0 else 2
            out_filters = current_filters if i == 0 else current_filters * 2
            self.layers.append(ResBlock1D(current_filters, out_filters, stride=stride))
            current_filters = out_filters
            current_t = current_t // stride
            self.projections.append(nn.Linear(current_filters * current_t, embed_dim))

    def forward(self, x):
        B, N, T = x.shape
        x = x.view(B * N, 1, T)
        x = self.stem(x)
        
        outputs = []
        feat = x
        for layer, proj in zip(self.layers, self.projections):
            feat = layer(feat)
            flat = feat.flatten(1) 
            emb = proj(flat).view(B, N, -1) 
            outputs.append(emb)
        return outputs 

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

# --- 3. Residual VQ (RVQ) ---

class VectorQuantizer(nn.Module):
    """
    Efficient VQ layer.
    """
    def __init__(self, n_e, e_dim, beta=0.25):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, z):
        # z: (Batch, Channels, D)
        z_flattened = z.view(-1, self.e_dim)
        
        # Distances
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        indices = torch.argmin(d, dim=1)
        z_q = self.embedding(indices).view(z.shape)

        # Loss
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # Preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, loss, indices.view(z.shape[0], z.shape[1], 1)

class ResidualVQ(nn.Module):
    """
    Residual Vector Quantizer.
    """
    def __init__(self, num_quantizers, n_e, e_dim, beta=0.25):
        super().__init__()
        self.layers = nn.ModuleList([
            VectorQuantizer(n_e, e_dim, beta) for _ in range(num_quantizers)
        ])

    def forward(self, z):
        quantized_out = 0
        residual = z
        total_loss = 0
        all_indices = []

        for layer in self.layers:
            quantized, loss, indices = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            total_loss += loss
            all_indices.append(indices)

        return quantized_out, total_loss, torch.cat(all_indices, dim=-1)

# --- 4. Main Tokenizer Model ---

class NeuroRVQTokenizer(nn.Module):
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
        n_codebooks=8, 
        vocab_size=512,
        freq_resolution=1.0,
        min_freq=0.0,
        max_freq=100.0,
        fs=200.0
    ):
        super().__init__()
        
        self.num_scales = num_scales
        self.embed_dim = embed_dim
        self.freq_resolution = freq_resolution
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.fs = fs
        
        self.fft_dim = int((max_freq - min_freq) / freq_resolution)
        if (max_freq - min_freq) % freq_resolution == 0:
             self.fft_dim += 1 
        
        self.n_fft = int(self.fs / freq_resolution)
        
        # 1. Temporal Encoder
        self.temporal_encoder = TemporalEncoder(in_chans, embed_dim)
        
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
        
        # 3. Multi-Branch RVQ
        self.rvqs = nn.ModuleList([
            ResidualVQ(n_codebooks, vocab_size, embed_dim) for _ in range(num_scales)
        ])
        
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
        
        # 2. Shared Transformer Pass (Batch Optimized)
        # Combine scales and add spatial embeddings
        h_all = torch.stack(ms_features, dim=0) + spatial_emb.unsqueeze(0) # (S, B, N, D)
        h_all = h_all.view(S * B, N, -1)
        
        h_encoded = self.transformer_encoder(h_all)
        h_scales = h_encoded.view(S, B, N, -1)
        
        # 3. Encode each scale with dedicated RVQ
        all_z_q = []
        total_vq_loss = 0
        for i in range(S):
            z_q, loss, _ = self.rvqs[i](h_scales[i])
            all_z_q.append(z_q)
            total_vq_loss += loss
            
        # 4. Latent Fusion
        z_fused = torch.sum(torch.stack(all_z_q, dim=0), dim=0)
        
        # 5. Decoder
        dec_h = self.transformer_decoder(z_fused)
        
        pred_amp = self.head_amp(dec_h)
        pred_sin = self.head_sin(dec_h)
        pred_cos = self.head_cos(dec_h)
        
        return pred_amp, pred_sin, pred_cos, total_vq_loss

    def get_loss(self, x, pred_amp, pred_sin, pred_cos):
        # x: (B, N, 200)
        # Compute Ground Truth FFT with target resolution
        # n_fft determines the frequency spacing
        x_fft = torch.fft.rfft(x, n=self.n_fft, dim=-1) # (B, N, n_fft//2 + 1)
        
        # Get Frequencies
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0/self.fs).to(x.device)
        
        # Select Indices in range [min_freq, max_freq]
        # We use a tolerance because of float comparison
        mask = (freqs >= self.min_freq - 1e-5) & (freqs <= self.max_freq + 1e-5)
        
        # Filter GT
        target_fft = x_fft[..., mask]
        
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

    def reconstruct(self, pred_amp, pred_sin, pred_cos, n_samples=200):
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
        
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0/self.fs).to(z_pred.device)
        mask = (freqs >= self.min_freq - 1e-5) & (freqs <= self.max_freq + 1e-5)
        
        # Handle potential shape mismatch by slicing mask or prediction
        indices = torch.where(mask)[0]
        count = min(len(indices), z_pred.shape[-1])
        full_z[..., indices[:count]] = z_pred[..., :count]
        
        # 5. Inverse Real FFT
        # Reconstruct to n_fft length first (padded length)
        x_recon_padded = torch.fft.irfft(full_z, n=self.n_fft, dim=-1)
        
        # 6. Crop to original length
        x_recon = x_recon_padded[..., :n_samples]
        
        return x_recon

