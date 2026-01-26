import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. Multi-Scale Temporal Encoder (Same as RecurrentVQ) ---

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

class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, in_chans=1, embed_dim=200, base_filters=8, num_scales=4, input_length=200):
        super().__init__()
        self.num_scales = num_scales
        self.stem = nn.Sequential(
            nn.Conv1d(in_chans, base_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(4, base_filters),
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
            input_dim = current_filters * current_t
            self.projections.append(nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            ))
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(), 
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_scales - 1)
        ])
        self.gates = nn.Parameter(torch.zeros(num_scales - 1)) 

    def forward(self, x):
        B, N, T = x.shape
        x = x.view(B * N, 1, T)
        x = self.stem(x)
        raw_features = []
        feat = x
        for layer, proj in zip(self.layers, self.projections):
            feat = layer(feat)
            flat = feat.flatten(1) 
            emb = proj(flat).view(B, N, -1) 
            raw_features.append(emb)
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

# --- 3. Finite Scalar Quantization (FSQ) ---

class FiniteScalarQuantizer(nn.Module):
    """
    Finite Scalar Quantizer (FSQ)
    As described in "Finite Scalar Quantization: VQ-VAE Made Simple"
    """
    def __init__(self, levels, embed_dim, dim):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.int32))
        self.register_buffer("basis", torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32))
        self.dim = dim
        self.embed_dim = embed_dim
        
        hidden_dim = embed_dim * 2
        
        # MLP Projection In (3 layers, wide)
        self.project_in = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
        # MLP Projection Out
        self.project_out = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Ranges for rounding: [- (L-1)/2, (L-1)/2]
        self.register_buffer("half_width", (self.levels - 1) / 2)

    def forward(self, z):
        # 1. Project and Bound
        # Project and apply tanh to bound to (-1, 1)
        z = self.project_in(z)
        z = torch.tanh(z)
        # Scale to [-half_width, half_width]
        z_bounded = z * self.half_width
        
        # 2. Quantize
        # Round to nearest level with Straight-Through Estimator
        z_q = z_bounded + (torch.round(z_bounded) - z_bounded).detach()
        
        # 3. Get Indices (for lookup/analysis)
        # Shift to [0, L-1] then multiply by basis
        indices = (z_q + self.half_width).to(torch.int32)
        idx = torch.sum(indices * self.basis, dim=-1, keepdim=True)
        
        # 4. Project back to embed_dim
        z_out = self.project_out(z_q)
        
        return z_out, idx

class RecurrentFSQ(nn.Module):
    """
    Recurrent FSQ.
    Applies FSQ iteratively to the residual.
    Uses SEPARATE FSQ modules for each scale.
    """
    def __init__(self, num_scales, num_recurrent_steps, levels, embed_dim):
        super().__init__()
        self.num_scales = num_scales
        self.num_recurrent_steps = num_recurrent_steps
        self.dim = len(levels)
        
        # Independent FSQ per scale
        self.fsqs = nn.ModuleList([
            FiniteScalarQuantizer(levels, embed_dim, self.dim)
            for _ in range(num_scales)
        ])

    def forward(self, z):
        # z: (S, B, N, D)
        S, B, N, D = z.shape
        
        # We need to process each scale independently because they have different FSQ modules
        # but we can still vectorize over (B, N)
        
        scale_outputs = []
        scale_indices = []
        
        for i in range(S):
            # Extract current scale: (B, N, D) -> Flatten to (B*N, D)
            z_scale = z[i].view(B * N, D)
            fsq_module = self.fsqs[i]
            
            quantized_out = 0
            residual = z_scale
            indices_per_step = []
            
            # Recurrent Loop for this scale
            for _ in range(self.num_recurrent_steps):
                z_q, idx = fsq_module(residual)
                residual = residual - z_q
                quantized_out = quantized_out + z_q
                indices_per_step.append(idx)
            
            # Store results for this scale
            scale_outputs.append(quantized_out) # (B*N, D)
            scale_indices.append(torch.stack(indices_per_step, dim=-1)) # (B*N, Steps)

        # Stack back to (S, B, N, D)
        # scale_outputs list of (B*N, D) -> stack (S, B*N, D) -> view (S, B, N, D)
        quantized_out = torch.stack(scale_outputs, dim=0).view(S, B, N, D)
        
        # indices list of (B*N, Steps) -> stack (S, B*N, Steps) -> view (S, B, N, Steps)
        all_indices = torch.stack(scale_indices, dim=0).view(S, B, N, -1)
        
        return quantized_out, torch.tensor(0.0, device=z.device), all_indices

# --- 4. Main Tokenizer Model ---

class RecurrentFSQTokenizer(nn.Module):
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
        fsq_levels=[8, 5, 5, 5], # Example levels, total codebook size = 8*5*5*5 = 1000
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
        
        # 3. Vectorized Multi-Scale Recurrent FSQ
        self.fsq = RecurrentFSQ(
            num_scales=num_scales,
            num_recurrent_steps=num_recurrent_steps,
            levels=fsq_levels,
            embed_dim=embed_dim
        )
        
        # 4. Decoder
        self.transformer_decoder = TransformerEncoder(embed_dim, dec_depth, dec_heads, mlp_ratio=dec_mlp_ratio) 
        
        self.head_amp = nn.Linear(embed_dim, self.fft_dim)
        self.head_sin = nn.Linear(embed_dim, self.fft_dim)
        self.head_cos = nn.Linear(embed_dim, self.fft_dim)

    def forward(self, x, coords):
        B, N, T = x.shape
        S = self.num_scales
        ms_features = self.temporal_encoder(x) 
        spatial_emb = self.spatial_mlp(coords) 
        h_all = torch.stack(ms_features, dim=0) + spatial_emb.unsqueeze(0) 
        h_all = h_all.view(S * B, N, -1)
        h_encoded = self.transformer_encoder(h_all)
        h_scales = h_encoded.view(S, B, N, -1)
        
        # 3. Vectorized FSQ Pass
        all_z_q, _, _ = self.fsq(h_scales)
            
        # 4. Latent Fusion
        z_fused = torch.sum(all_z_q, dim=0)
        
        # 5. Decoder
        dec_h = self.transformer_decoder(z_fused)
        pred_amp = self.head_amp(dec_h)
        pred_sin = self.head_sin(dec_h)
        pred_cos = self.head_cos(dec_h)
        
        return pred_amp, pred_sin, pred_cos, torch.tensor(0.0, device=x.device)

    def get_loss(self, x, pred_amp, pred_sin, pred_cos, x_fft=None):
        if x_fft is None:
            x_fft = torch.fft.rfft(x, n=self.n_fft, dim=-1)
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0/self.fs).to(x.device)
        mask = (freqs >= self.min_freq - 1e-5) & (freqs <= self.max_freq + 1e-5)
        target_fft = x_fft[..., mask]
        if target_fft.shape[-1] != pred_amp.shape[-1]:
             min_len = min(target_fft.shape[-1], pred_amp.shape[-1])
             target_fft = target_fft[..., :min_len]
             pred_amp = pred_amp[..., :min_len]
             pred_sin = pred_sin[..., :min_len]
             pred_cos = pred_cos[..., :min_len]
        gt_amp = torch.abs(target_fft)
        gt_phase = torch.angle(target_fft)
        target_log_amp = torch.log1p(gt_amp)
        loss_amp = F.mse_loss(pred_amp, target_log_amp)
        target_sin = torch.sin(gt_phase)
        target_cos = torch.cos(gt_phase)
        loss_phase = F.mse_loss(pred_sin, target_sin) + F.mse_loss(pred_cos, target_cos)
        x_recon = self.reconstruct(pred_amp, pred_sin, pred_cos, n_samples=x.shape[-1])
        loss_temp = F.l1_loss(x_recon, x)
        total_loss = loss_amp + loss_phase + loss_temp
        return total_loss, loss_amp, loss_phase, loss_temp

    def reconstruct(self, pred_amp, pred_sin, pred_cos, n_samples=200):
        amp = torch.exp(pred_amp) - 1
        amp = torch.clamp(amp, min=0)
        pred_norm = torch.sqrt(pred_cos**2 + pred_sin**2 + 1e-8)
        norm_cos = pred_cos / pred_norm
        norm_sin = pred_sin / pred_norm
        real = amp * norm_cos
        imag = amp * norm_sin
        z_pred = torch.complex(real, imag)
        full_fft_dim = self.n_fft // 2 + 1
        full_z = torch.zeros((z_pred.shape[0], z_pred.shape[1], full_fft_dim), dtype=z_pred.dtype, device=z_pred.device)
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0/self.fs).to(z_pred.device)
        mask = (freqs >= self.min_freq - 1e-5) & (freqs <= self.max_freq + 1e-5)
        indices = torch.where(mask)[0]
        count = min(len(indices), z_pred.shape[-1])
        full_z[..., indices[:count]] = z_pred[..., :count]
        x_recon_padded = torch.fft.irfft(full_z, n=self.n_fft, dim=-1)
        x_recon = x_recon_padded[..., :n_samples]
        return x_recon
