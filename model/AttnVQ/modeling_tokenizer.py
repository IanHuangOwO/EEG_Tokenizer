import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations

# --- 1. Efficient Building Blocks (2D) ---

class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, 
            kernel_size=(1, kernel_size), 
            stride=(1, stride), 
            padding=(0, padding * dilation), 
            dilation=(1, dilation), 
            padding_mode='replicate',
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# --- 2. SpatialTemporal Encoder ---

class SpatialTemporalEncoder(nn.Module):
    def __init__(self, in_chans=1, base_filters=16, embed_dim=256, max_temporal_patches=10):
        super().__init__()
        # Stem: RF 3, Stride 1 [P1]
        self.stem = ConvBlock2D(in_chans, base_filters, 3, 1, 1)
        
        # Stage 1: Stride 2, Dilation 1
        self.stage1 = ConvBlock2D(base_filters, base_filters * 2, 3, 2, 1)
        
        # Stage 2: Stride 2, Dilation 2 [P2]
        self.stage2 = ConvBlock2D(base_filters * 2, base_filters * 4, 3, 2, 1, dilation=2)
        
        # Stage 3: Stride 2, Dilation 4 [P3]
        self.stage3 = ConvBlock2D(base_filters * 4, base_filters * 8, 3, 2, 1, dilation=4)
        
        # Multiscale Projections to embed_dim
        self.projections = nn.ModuleList([
            nn.LazyLinear(embed_dim), # For P1
            nn.LazyLinear(embed_dim), # For P2
            nn.LazyLinear(embed_dim)  # For P3
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(3)])
        
        # Spatial MLP
        self.spatial_mlp = nn.Sequential(
            nn.Linear(3, embed_dim), 
            nn.GELU(), 
            nn.Linear(embed_dim, embed_dim)
        )
        nn.init.zeros_(self.spatial_mlp[-1].weight)
        nn.init.zeros_(self.spatial_mlp[-1].bias)
        
        # Temporal Embedding
        self.temporal_emb = nn.Embedding(max_temporal_patches, embed_dim)
        nn.init.normal_(self.temporal_emb.weight, std=0.02)

    def forward(self, x, coords, time_idx=None):
        # x: (B, C, T)
        # coords: (B, C, 3)
        x = x.unsqueeze(1) # (B, 1, C, T)
        p1 = self.stem(x) 
        s1 = self.stage1(p1)
        p2 = self.stage2(s1)
        p3 = self.stage3(p2)
        
        # Spatial Embedding
        spatial_emb = self.spatial_mlp(coords) # (B, C, embed_dim)
        
        # Temporal Embedding
        if time_idx is not None:
            t_emb = self.temporal_emb(time_idx).unsqueeze(1) # (B, 1, embed_dim)
        else:
            t_emb = 0
        
        raw_feats = [p1, p2, p3]
        projected_feats = []
        for feat, proj, norm in zip(raw_feats, self.projections, self.norms):
            # feat: (B, Filters, C, T_scale)
            # Permute and flatten filters and temporal dimension per channel
            z = proj(feat.permute(0, 2, 1, 3).flatten(2)) # (B, C, embed_dim)
            # Combine with spatial and temporal embeddings
            z = norm(z + spatial_emb + t_emb)
            projected_feats.append(z)
            
        return projected_feats

# --- 3. Transformer Components ---

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=0., activation='gelu', batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x): return self.norm(self.block(x))

class ScaleEncoder(nn.Module):
    def __init__(self, embed_dim, depth, heads, mlp_ratio=4.):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(embed_dim, heads, mlp_ratio) for _ in range(depth)])
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x

class ScaleDecoder(nn.Module):
    def __init__(self, embed_dim, depth, heads, fft_dim, mlp_ratio=4.):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(embed_dim, heads, mlp_ratio) for _ in range(depth)])
        self.head_amp = nn.Linear(embed_dim, fft_dim)
        self.head_sin = nn.Linear(embed_dim, fft_dim)
        self.head_cos = nn.Linear(embed_dim, fft_dim)
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return self.head_amp(x), self.head_sin(x), self.head_cos(x)

# --- 4. Quantization (Low-Rank Subspace Expert) ---

class AttnVQ(nn.Module):
    def __init__(self, in_scales, num_heads, vq_head_vocab_size, e_dim, decay=0.99, eps=1e-5):
        super().__init__()
        # Rank of the subspace (r)
        self.r = vq_head_vocab_size
        self.in_scales, self.num_heads, self.vq_head_vocab_size, self.e_dim = in_scales, num_heads, vq_head_vocab_size, e_dim
        
        # Low-Rank Projections
        # A (Filter): Projects D -> r
        self.A = nn.Parameter(torch.empty(in_scales, num_heads, e_dim, self.r))
        # B (Synthesizer): Projects r -> D
        self.B = nn.Parameter(torch.empty(in_scales, num_heads, self.r, e_dim))
        
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        
        # Learnable Head Weights for the Additive Bottleneck (Scale-Specific)
        # Initialized to zeros so Softmax starts as a uniform distribution (1/num_heads)
        self.head_weights = nn.Parameter(torch.zeros(in_scales, 1, 1, num_heads, 1))

        # Fixed Orthogonal Codebook (Identity Matrix)
        eye = torch.eye(self.r).view(1, 1, self.r, self.r).expand(in_scales, num_heads, -1, -1)
        self.register_buffer('embedding', eye)

    @torch.no_grad()
    def get_current_metrics(self):
        """
        Calculates health metrics for the projections and gating.
        Returns: Dictionary of metrics.
        """
        metrics = {}
        
        # Track Normalized Head Weights (Actual contribution)
        gate_weights = F.softmax(self.head_weights, dim=3).flatten()
        metrics['head_weight_mean'] = gate_weights.mean().item()
        metrics['head_weight_max'] = gate_weights.max().item()
        metrics['head_weight_min'] = gate_weights.min().item()
        
        # Calculate Orthogonality of A
        A_norm = F.normalize(self.A, p=2, dim=2) # (S, H, D, r)
        A_corr = torch.einsum('shdx,skdy->shkxy', A_norm, A_norm)
        mask = torch.eye(self.num_heads, device=self.A.device).bool().view(1, self.num_heads, self.num_heads, 1, 1)
        off_diag_A = A_corr[~mask.expand(self.in_scales, -1, -1, self.r, self.r)]
        metrics['A_overlap'] = (off_diag_A ** 2).mean().item()
        
        # Calculate Orthogonality of B
        B_norm = F.normalize(self.B, p=2, dim=3) # (S, H, r, D)
        B_corr = torch.einsum('shxd,skyd->shkxy', B_norm, B_norm)
        off_diag_B = B_corr[~mask.expand(self.in_scales, -1, -1, self.r, self.r)]
        metrics['B_overlap'] = (off_diag_B ** 2).mean().item()

        return metrics

    def forward(self, z):
        # z: (S, B, C, D) -> S: scales, B: batch, C: channels, D: dim
        S, B_sz, C, D = z.shape
        H, r = self.num_heads, self.r
        
        # Broadcast input for multi-head processing
        z_reshaped = z.unsqueeze(3).expand(-1, -1, -1, H, -1) # (S, B, C, H, D)
        
        # 1. Project to Subspace (Filter)
        # q = z * A
        q = torch.einsum('sbchd,shdr->sbchr', z_reshaped, self.A)
        q_norm = F.normalize(q, p=2, dim=-1)
        
        # 2. Match with Fixed Codebook
        # Since embedding is an Identity matrix, this is essentially a pass-through
        # but we keep the explicit calculation for clarity and potential fixed bases.
        logits = torch.einsum('sbchr,shvr->sbchv', q_norm, self.embedding)
        
        # 3. Simple Soft-Attention (No Top-K, No Temperature)
        weights = F.softmax(logits, dim=-1) # (S, B, C, H, r)
        
        # indices for compatibility (Argmax)
        indices = logits.argmax(dim=-1, keepdim=True) # (S, B, C, H, 1)
        
        # 4. Reconstruct in Subspace and Synthesize
        v_q = torch.einsum('sbchv,shvr->sbchr', weights, self.embedding)
        # Project back to full dimension
        z_q_soft_heads = torch.einsum('sbchr,shre->sbche', v_q, self.B) # (S, B, C, H, D)
        
        # 5. Weighted Additive Bottleneck (Normalized Gating)
        gate_weights = F.softmax(self.head_weights, dim=3) # (S, 1, 1, H, 1)
        z_q_soft_heads_weighted = z_q_soft_heads * gate_weights
        z_q_soft = z_q_soft_heads_weighted.sum(dim=3)
        
        # Straight-Through Estimator (STE)
        z_q = z + (z_q_soft - z).detach()

        # 6. Losses
        # Commitment Loss
        loss_commit = 0.25 * F.mse_loss(z_q_soft.detach(), z) + 0.25 * F.mse_loss(z_q_soft, z.detach())
        
        # Subspace Orthogonality Loss (Diversification)
        loss_subspace_overlap = 0
        if H > 1:
            # Filter Overlap
            A_norm = F.normalize(self.A, p=2, dim=2)
            A_corr = torch.einsum('shdx,skdy->shkxy', A_norm, A_norm)
            
            # Synthesizer Overlap
            B_norm = F.normalize(self.B, p=2, dim=3)
            B_corr = torch.einsum('shxd,skyd->shkxy', B_norm, B_norm)
            
            mask = torch.eye(H, device=z.device).bool().view(1, H, H, 1, 1)
            off_diag_A = A_corr[~mask.expand(S, -1, -1, r, r)]
            off_diag_B = B_corr[~mask.expand(S, -1, -1, r, r)]
            
            loss_subspace_overlap = (torch.mean(off_diag_A ** 2) + torch.mean(off_diag_B ** 2)) * 10.0

        # Combined Loss
        loss = loss_commit + loss_subspace_overlap
        
        return z_q, loss, indices, weights

# --- 5. Main Tokenizer Model (AttnVQTokenizer) ---

class AttnVQTokenizer(nn.Module):
    def __init__(
        self,
        in_chans=1, embed_dim=256, enc_depth=4, enc_heads=8, enc_mlp_ratio=4.,
        dec_depth=2, dec_heads=8, dec_mlp_ratio=4., in_scales=3,
        vq_head_num=8, vq_head_vocab_size=64,
        freq_resolution=1.0, min_freq=0.0, max_freq=100.0, fs=200.0, input_length=200,
        max_temporal_patches=10
    ):
        super().__init__()
        self.in_scales, self.embed_dim, self.vq_head_num, self.fs, self.input_length = in_scales, embed_dim, vq_head_num, fs, input_length
        self.fft_dim = int(round((max_freq - min_freq) / freq_resolution)) + 1 
        self.n_fft = int(self.fs / freq_resolution)
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0/self.fs)
        mask = (freqs >= min_freq - 1e-5) & (freqs <= max_freq + 1e-5)
        self.register_buffer('freq_mask', mask)
        self.register_buffer('freq_indices', torch.where(mask)[0])
        
        # SpatialTemporalEncoder handles projections and spatial embeddings
        self.spatial_temporal_encoder = SpatialTemporalEncoder(in_chans, base_filters=16, embed_dim=embed_dim, max_temporal_patches=max_temporal_patches)
        
        self.scale_encoders = nn.ModuleList([ScaleEncoder(embed_dim, depth=enc_depth, heads=enc_heads, mlp_ratio=enc_mlp_ratio) for _ in range(in_scales)])
        
        self.attnvq = AttnVQ(in_scales, self.vq_head_num, vq_head_vocab_size, embed_dim)
        self.scale_decoders = nn.ModuleList([ScaleDecoder(embed_dim, depth=dec_depth, heads=dec_heads, fft_dim=self.fft_dim, mlp_ratio=dec_mlp_ratio) for _ in range(in_scales)])

    def forward(self, x, coords, time_idx=None):
        # Projected features with spatial embeddings: List of (B, C, embed_dim) for each scale
        h_scales_projected = self.spatial_temporal_encoder(x, coords, time_idx) 
        
        h_scales = []
        for i in range(self.in_scales):
            z = h_scales_projected[i]
            enc = self.scale_encoders[i]
            
            # Apply transformer encoder
            z = enc(z)
            h_scales.append(z)
            
        all_z_q, vq_loss, top_k_indices, weights = self.attnvq(torch.stack(h_scales, dim=0))
        
        pred_amp, pred_sin, pred_cos = 0, 0, 0
        for i, decoder in enumerate(self.scale_decoders):
            a, s, c = decoder(all_z_q[i])
            pred_amp += a; pred_sin += s; pred_cos += c
            
        return pred_amp, pred_sin, pred_cos, vq_loss, top_k_indices, weights

    def get_loss(self, x, pred_amp, pred_sin, pred_cos, x_fft=None):
        if x_fft is None: x_fft = torch.fft.rfft(x, n=self.n_fft, dim=-1)
        target_fft = x_fft[..., self.freq_mask]
        if target_fft.shape[-1] != pred_amp.shape[-1]:
             min_len = min(target_fft.shape[-1], pred_amp.shape[-1])
             target_fft, pred_amp, pred_sin, pred_cos = target_fft[..., :min_len], pred_amp[..., :min_len], pred_sin[..., :min_len], pred_cos[..., :min_len]
        gt_amp, gt_phase = torch.abs(target_fft), torch.angle(target_fft)
        loss_amp = F.mse_loss(pred_amp, torch.log1p(gt_amp))
        loss_phase = F.mse_loss(pred_sin, torch.sin(gt_phase)) + F.mse_loss(pred_cos, torch.cos(gt_phase))
        x_recon = self.reconstruct(pred_amp, pred_sin, pred_cos, n_samples=x.shape[-1])
        return loss_amp + loss_phase + F.l1_loss(x_recon, x), loss_amp, loss_phase, F.l1_loss(x_recon, x), F.mse_loss(x_recon, x)

    def reconstruct(self, pred_amp, pred_sin, pred_cos, n_samples=200):
        amp = torch.clamp(torch.exp(pred_amp) - 1, min=0) 
        norm = torch.sqrt(pred_cos**2 + pred_sin**2 + 1e-8)
        z_pred = torch.complex(amp * (pred_cos / norm), amp * (pred_sin / norm))
        full_z = torch.zeros((z_pred.shape[0], z_pred.shape[1], self.n_fft // 2 + 1), dtype=z_pred.dtype, device=z_pred.device)
        count = min(len(self.freq_indices), z_pred.shape[-1])
        full_z[..., self.freq_indices[:count]] = z_pred[..., :count]
        return torch.fft.irfft(full_z, n=self.n_fft, dim=-1)[..., :n_samples]

    def get_codebooks(self):
        return [(self.attnvq.embedding[s, h].detach().cpu(), f"S{s}_H{h}") for s in range(self.in_scales) for h in range(self.vq_head_num)]

    def get_indices(self, x, coords, time_idx=None):
        with torch.no_grad(): _, _, _, _, indices, weights = self.forward(x, coords, time_idx)
        indices_flat = indices.permute(1, 2, 0, 3, 4).reshape(-1, self.in_scales, self.vq_head_num, indices.shape[4]).unsqueeze(1)
        weights_flat = weights.permute(1, 2, 0, 3, 4).reshape(-1, self.in_scales, self.vq_head_num, weights.shape[4]).unsqueeze(1)
        return indices_flat, weights_flat
