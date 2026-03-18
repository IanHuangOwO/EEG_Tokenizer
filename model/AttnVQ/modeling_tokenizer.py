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
        
        # Low-Rank Projections (Optimized Flattened Format)
        # A (Filter): Projects D -> H * r
        self.A = nn.Parameter(torch.empty(in_scales, e_dim, num_heads * self.r))
        # B (Synthesizer): Projects H * r -> D
        self.B = nn.Parameter(torch.empty(in_scales, num_heads * self.r, e_dim))
        
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        
        # Learnable Head Weights for the Additive Bottleneck (Scale-Specific)
        # Initialized to zeros so Softmax starts as a uniform distribution (1/num_heads)
        self.head_weights = nn.Parameter(torch.zeros(in_scales, 1, 1, num_heads, 1))

        # Fixed Orthogonal Codebook (Identity Matrix)
        eye = torch.eye(self.r).view(1, 1, self.r, self.r).expand(in_scales, num_heads, -1, -1)
        self.register_buffer('embedding', eye)
        
        # Learnable Temperature/Scale for the Softmax (One per head)
        # Initialized to 1.0 to start with broad/soft selection
        self.logit_scale = nn.Parameter(torch.ones(in_scales, 1, 1, num_heads, 1) * 1.0)

        # Buffers for monitoring codebook health
        # avg_probs: Tracking the global distribution of code usage across the dataset
        self.register_buffer('avg_probs', torch.ones(in_scales, num_heads, self.r) / self.r)
        # max_prob_ema: Tracking how "sharp" the softmax picks are
        self.register_buffer('max_prob_ema', torch.tensor(1.0 / self.r))

    @torch.no_grad()
    def get_current_metrics(self):
        """
        Calculates health metrics for the projections and gating.
        Returns: Dictionary of metrics.
        """
        metrics = {}
        S, H, r, D = self.in_scales, self.num_heads, self.r, self.e_dim
        
        # 1. Gating Metrics
        gate_weights = F.softmax(self.head_weights, dim=3).squeeze() # (S, H) or (H,)
        if gate_weights.dim() == 1:
            gate_weights = gate_weights.unsqueeze(0)
            
        metrics['head_weight_mean'] = gate_weights.mean().item()
        metrics['head_weight_max'] = gate_weights.max().item()
        metrics['head_weight_min'] = gate_weights.min().item()
            
        for s in range(S):
            for h in range(H):
                metrics[f'head_weight_s{s}_h{h}'] = gate_weights[s, h].item()
        
        # 2. Codebook Diversity (Perplexity) and Sharpness
        # p: (S, H, r)
        p = self.avg_probs
        entropy = -torch.sum(p * torch.log(p + 1e-10), dim=-1) # (S, H)
        perplexity = torch.exp(entropy) # (S, H)
        metrics['codebook_perplexity'] = perplexity.mean().item()
        metrics['codebook_sharpness'] = self.max_prob_ema.item()
        metrics['logit_scale_mean'] = self.logit_scale.mean().item()
        metrics['logit_scale_max'] = self.logit_scale.max().item()
        metrics['logit_scale_min'] = self.logit_scale.min().item()
        
        # 3. Subspace Orthogonality and Rank Health (Large SVD)
        A_reshaped = self.A.view(S, D, H, r).transpose(1, 2) # (S, H, D, r)
        B_reshaped = self.B.view(S, H, r, D) # (S, H, r, D)

        # Calculate all singular values: (S, H, r)
        A_sv = torch.linalg.svdvals(A_reshaped) 
        B_sv = torch.linalg.svdvals(B_reshaped)
        
        # Global stats across all heads and scales
        metrics['A_sv_min'] = A_sv.min().item()
        metrics['A_sv_max'] = A_sv.max().item()
        metrics['A_sv_mean'] = A_sv.mean().item()
        metrics['A_condition'] = (A_sv.max() / (A_sv.min() + 1e-8)).item()
        
        metrics['B_sv_min'] = B_sv.min().item()
        metrics['B_sv_max'] = B_sv.max().item()
        metrics['B_sv_mean'] = B_sv.mean().item()
        metrics['B_condition'] = (B_sv.max() / (B_sv.min() + 1e-8)).item()

        A_norm_vals = F.normalize(A_reshaped, p=2, dim=2)
        A_corr = torch.einsum('shdx,skdy->shkxy', A_norm_vals, A_norm_vals)
        mask = torch.eye(H, device=self.A.device).bool().view(1, H, H, 1, 1)
        off_diag_A = A_corr[~mask.expand(S, -1, -1, r, r)]
        metrics['A_overlap'] = (off_diag_A ** 2).mean().item()
        
        B_norm_vals = F.normalize(B_reshaped, p=2, dim=3)
        B_corr = torch.einsum('shxd,skyd->shkxy', B_norm_vals, B_norm_vals)
        off_diag_B = B_corr[~mask.expand(S, -1, -1, r, r)]
        metrics['B_overlap'] = (off_diag_B ** 2).mean().item()

        return metrics

    def forward(self, z):
        # z: (S, B, C, D) -> S: scales, B: batch, C: channels, D: dim
        S, B_sz, C, D = z.shape
        H, r = self.num_heads, self.r
        
        # 1. Project to Subspace (Filter) using Grouped Linear (BMM)
        z_flat = z.view(S, B_sz * C, D) # (S, B_sz * C, D)
        q_flat = torch.bmm(z_flat, self.A) # (S, B_sz * C, H * r)
        
        # Reshape to separate heads and rank
        q = q_flat.view(S, B_sz, C, H, r)
        
        # Stabilize with L2-norm but allow sharpness via learnable logit_scale
        q_norm = F.normalize(q, p=2, dim=-1)
        
        # Clamp logit scale to prevent extreme sharpness from destabilizing reconstruction
        s_clamped = self.logit_scale.clamp(1.0, 5.0)
        
        # 2. Match with Fixed Codebook
        # We scale the normalized logits to control the Softmax distribution "width"
        logits = torch.einsum('sbchr,shvr->sbchv', q_norm * s_clamped, self.embedding)
        
        # 3. Simple Soft-Attention
        weights = F.softmax(logits, dim=-1) # (S, B, C, H, r)
        indices = logits.argmax(dim=-1, keepdim=True) # (S, B, C, H, 1)
        
        # --- Update Health Buffers ---
        if self.training:
            with torch.no_grad():
                # Global Usage EMA
                batch_avg = weights.mean(dim=(1, 2)) # (S, H, r)
                self.avg_probs.mul_(0.99).add_(batch_avg, alpha=0.01)
                
                # Sharpness EMA (mean of max probability)
                batch_max = weights.max(dim=-1)[0].mean()
                self.max_prob_ema.mul_(0.99).add_(batch_max, alpha=0.01)
        
        # 4. Reconstruct and Gate in Subspace
        v_q = torch.einsum('sbchv,shvr->sbchr', weights, self.embedding)
        gate_weights = F.softmax(self.head_weights, dim=3) # (S, 1, 1, H, 1)
        v_q_gated = v_q * gate_weights # (S, B, C, H, r)
        
        # 5. Synthesize using Grouped Linear (BMM)
        v_q_gated_flat = v_q_gated.reshape(S, B_sz * C, H * r)
        z_q_soft_flat = torch.bmm(v_q_gated_flat, self.B) # (S, B_sz * C, D)
        z_q_soft = z_q_soft_flat.view(S, B_sz, C, D)
        
        # Straight-Through Estimator (STE)
        z_q = z + (z_q_soft - z).detach()

        # 6. Losses
        # Commitment Loss
        loss_commit = 0.25 * F.mse_loss(z_q_soft.detach(), z) + 0.25 * F.mse_loss(z_q_soft, z.detach())
        
        # Subspace Diversification & Rank Health
        def get_subspace_loss(M, is_A=True):
            # M: (S, D, Hr) for A, (S, Hr, D) for B
            Hr = H * r
            if is_A:
                # Self-correlation: (S, Hr, Hr)
                # No normalization here: forces columns to be unit-length AND orthogonal
                corr = torch.bmm(M.transpose(1, 2), M)
            else:
                # Self-correlation: (S, Hr, Hr)
                corr = torch.bmm(M, M.transpose(1, 2))
            
            # Create Block-Diagonal Mask for Intra-Head (Rank)
            mask_intra = torch.block_diag(*[torch.ones(r, r, device=z.device) for _ in range(H)])
            mask_intra = mask_intra.unsqueeze(0).expand(S, -1, -1)
            
            # 1. Rank Loss (Intra-Head): Force each head's r columns to be orthonormal (I_r)
            I_hr = torch.eye(Hr, device=z.device).unsqueeze(0).expand(S, -1, -1)
            loss_rank = F.mse_loss(corr * mask_intra, I_hr) 
            
            # 2. Diversity Loss (Inter-Head): Force different heads to have minimal overlap
            # We use a softer penalty for inter-head overlap since Hr > D
            loss_div = torch.mean((corr * (1 - mask_intra)) ** 2)
            
            return loss_rank * 20.0 + loss_div * 5.0 # Lowered diversity weight

        loss_ortho = get_subspace_loss(self.A, is_A=True) + get_subspace_loss(self.B, is_A=False)

        # Combined Loss
        loss = loss_commit + loss_ortho
        
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
