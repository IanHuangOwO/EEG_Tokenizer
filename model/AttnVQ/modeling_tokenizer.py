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
    def __init__(self, in_chans=1, base_filters=16, embed_dim=256):
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

    def forward(self, x, coords):
        # x: (B, C, T)
        # coords: (B, C, 3)
        x = x.unsqueeze(1) # (B, 1, C, T)
        p1 = self.stem(x) 
        s1 = self.stage1(p1)
        p2 = self.stage2(s1)
        p3 = self.stage3(p2)
        
        # Spatial Embedding
        spatial_emb = self.spatial_mlp(coords) # (B, C, embed_dim)
        
        raw_feats = [p1, p2, p3]
        projected_feats = []
        for feat, proj, norm in zip(raw_feats, self.projections, self.norms):
            # feat: (B, Filters, C, T_scale)
            # Permute and flatten filters and temporal dimension per channel
            z = proj(feat.permute(0, 2, 1, 3).flatten(2)) # (B, C, embed_dim)
            # Combine with spatial embedding
            z = norm(z + spatial_emb)
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

# --- 4. Quantization (EMA-based Soft Attention) ---

class AttnVQ(nn.Module):
    def __init__(self, in_scales, num_heads, vq_head_vocab_size, e_dim, vq_head_top_k=8, temperature=1.0, decay=0.99, eps=1e-5):
        super().__init__()
        # Ensure Top-K forces selection (must be less than vocab size to avoid blurry averages)
        self.vq_head_top_k = min(vq_head_top_k, max(1, vq_head_vocab_size // 2))
        self.in_scales, self.num_heads, self.vq_head_vocab_size, self.e_dim = in_scales, num_heads, vq_head_vocab_size, e_dim
        self.decay = decay
        self.eps = eps
        self.head_dim = e_dim // num_heads
        
        # Temperature is now a buffer, we will update it from the training loop
        self.register_buffer('temperature', torch.ones(in_scales, num_heads, 1, 1) * temperature)
        
        # Query and Output Projections (Wq and Wo)
        self.Wq = nn.Parameter(torch.empty(in_scales, num_heads, self.head_dim, self.head_dim))
        self.Wo = nn.Parameter(torch.empty(in_scales, num_heads, self.head_dim, self.head_dim))
        
        # Identity Initialization for Wq and Wo: Ensures the signal passes through at the start
        with torch.no_grad():
            eye = torch.eye(self.head_dim).view(1, 1, self.head_dim, self.head_dim)
            self.Wq.copy_(eye.expand(in_scales, num_heads, -1, -1))
            self.Wo.copy_(eye.expand(in_scales, num_heads, -1, -1))

        # Apply orthogonal parameterization.
        # This forces Wq and Wo to remain orthogonal matrices during optimization.
        parametrizations.orthogonal(self, 'Wq')
        parametrizations.orthogonal(self, 'Wo')

        # Codebook is updated via EMA, so we disable gradients for its values
        self.embedding = nn.Parameter(torch.empty(in_scales, num_heads, vq_head_vocab_size, self.head_dim), requires_grad=False)
        nn.init.xavier_uniform_(self.embedding)

        # EMA Buffers for running averages
        self.register_buffer('ema_cluster_size', torch.zeros(in_scales, num_heads, vq_head_vocab_size))
        self.register_buffer('ema_dw', torch.empty(in_scales, num_heads, vq_head_vocab_size, self.head_dim))
        self.ema_dw.data.copy_(self.embedding.data)

    def set_temperature(self, value):
        self.temperature.fill_(value)

    @torch.no_grad()
    def get_current_metrics(self):
        """
        Calculates health metrics for the codebooks and projections.
        Returns: Dictionary of metrics.
        """
        S, H, V, D_h = self.embedding.shape
        metrics = {}
        
        # 1. Effective Rank Calculation helper
        def calc_erank(tensor):
            # tensor: (..., D, D) or (..., V, D)
            _, s, _ = torch.svd(tensor)
            p = s / (s.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -torch.sum(p * torch.log(p + 1e-8), dim=-1)
            return torch.exp(entropy)

        # 2. Average Pairwise Similarity helper
        def calc_avg_sim(tensor):
            # tensor: (S, H, ...) -> flat: (S, H, D_flat)
            if H < 2: return torch.zeros(S, device=tensor.device)
            flat = F.normalize(tensor.reshape(S, H, -1), p=2, dim=-1)
            # Pairwise cosine sim: (S, H, H)
            sim_mtx = torch.einsum('shd,skd->shk', flat, flat)
            # Mask diagonal and average
            mask = torch.eye(H, device=tensor.device).bool().view(1, H, H)
            return sim_mtx[~mask.expand(S, -1, -1)].view(S, -1).mean(dim=-1)

        # --- Execute Calculations ---
        
        # Codebook Metrics
        # We calculate rank on normalized embeddings because that's what the model uses in forward()
        embedding_norm = F.normalize(self.embedding, p=2, dim=-1)
        cb_erank = calc_erank(embedding_norm) # (S, H)
        metrics['cb_erank'] = cb_erank.mean().item()
        metrics['cb_sim'] = calc_avg_sim(self.embedding).mean().item()
        
        # Projection Metrics
        wq_erank = calc_erank(self.Wq) # (S, H)
        wo_erank = calc_erank(self.Wo) # (S, H)
        metrics['wq_erank'] = wq_erank.mean().item()
        metrics['wo_erank'] = wo_erank.mean().item()
        
        metrics['wq_sim'] = calc_avg_sim(self.Wq).mean().item()
        metrics['wo_sim'] = calc_avg_sim(self.Wo).mean().item()
        
        # Orthogonality Error
        q_eye = torch.eye(D_h, device=self.Wq.device).view(1, 1, D_h, D_h)
        metrics['wq_ortho'] = F.mse_loss(torch.einsum('shde,shfe->shdf', self.Wq, self.Wq), q_eye.expand(S, H, -1, -1)).item()
        metrics['wo_ortho'] = F.mse_loss(torch.einsum('shde,shfe->shdf', self.Wo, self.Wo), q_eye.expand(S, H, -1, -1)).item()
        
        # 3. Perplexity (Measure of codebook utilization uniformity)
        usage = self.ema_cluster_size + self.eps
        probs = usage / usage.sum(dim=-1, keepdim=True)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1) # (S, H)
        metrics['perplexity'] = torch.exp(entropy).mean().item()
            
        return metrics

    def forward(self, z):
        # z: (S, B, C, D) -> S: scales, B: batch, C: channels, D: dim
        S, B, C, D = z.shape
        H, V, D_h = self.num_heads, self.vq_head_vocab_size, self.head_dim
        
        # Reshape input for multi-head attention
        z_reshaped = z.view(S, B, C, H, D_h)
        
        # 1. Query Projection and Normalization
        # Q = z * Wq
        q = torch.einsum('sbchd,shde->sbche', z_reshaped, self.Wq)
        q_norm = F.normalize(q, p=2, dim=-1)
        
        # Codebook normalization
        embedding_norm = F.normalize(self.embedding, p=2, dim=-1)
        
        # Calculate Logits (Cosine Similarity between Q and Codebook)
        logits = torch.einsum('sbche,shve->sbchv', q_norm, embedding_norm)
        
        # 2. Soft-Attention over Top-K
        top_vals, indices = torch.topk(logits, k=self.vq_head_top_k, dim=-1)
        weights = F.softmax(top_vals / self.temperature.view(S, 1, 1, H, 1), dim=-1)
        
        # weights_full: (S, B, C, H, V) used for reconstruction and EMA updates
        weights_full = torch.zeros_like(logits).scatter_(-1, indices, weights)
        
        # 3. Retrieve and Project Output
        # Reconstruct z_q: Weighted sum of codebook vectors followed by Wo
        # V_q = weighted sum(embedding_norm)
        v_q = torch.einsum('sbchv,shvd->sbchd', weights_full, embedding_norm)
        
        # z_q_soft = V_q * Wo
        z_q_soft = torch.einsum('sbchd,shde->sbche', v_q, self.Wo).reshape(S, B, C, D)
        
        # Straight-Through Estimator (STE):
        # In the forward pass, we use the quantized representation z_q_soft.
        # In the backward pass, gradients flow directly to the encoder z.
        # This is critical for stable reconstruction.
        z_q = z + (z_q_soft - z).detach()

        # 4. EMA Updates (Training only)
        if self.training:
            # usage: total soft-weight assigned to each vector in this batch
            usage = weights_full.detach().sum(dim=(1, 2)) # (S, H, V)
            self.ema_cluster_size.mul_(self.decay).add_(usage, alpha=1 - self.decay)
            
            # dw: weighted sum of Query features for each vector
            dw = torch.einsum('sbchv,sbche->shve', weights_full.detach(), q_norm.detach())
            self.ema_dw.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            # Laplace Smoothing to update codebook
            n = self.ema_cluster_size.sum(dim=-1, keepdim=True)
            smoothed_cluster_size = (self.ema_cluster_size + self.eps) / (n + V * self.eps) * n
            
            # Update and Normalize: Keep the codebook on the unit sphere
            # This ensures Norm is always 1.0 and stabilizes similarity calculations.
            updated_embedding = self.ema_dw / smoothed_cluster_size.unsqueeze(-1)
            self.embedding.data.copy_(F.normalize(updated_embedding, p=2, dim=-1))

        # 5. Commitment, Alignment, and Projection Diversity Loss
        # Term 1: Pulls the Encoder (z) toward the quantized representation
        # Term 2: Pulls the Projections (Wq, Wo) toward the Encoder's output
        loss_commit = 0.25 * F.mse_loss(z_q_soft.detach(), z) + 0.25 * F.mse_loss(z_q_soft, z.detach())
        
        # Term 3: Inter-head Diversity Loss (Weight 1.0 + Sum)
        # Intra-head Orthogonality is now guaranteed by structural parameterization.
        loss_div = 0
        h_eye = torch.eye(H, device=z.device).view(1, H, H)
        
        for w in [self.Wq, self.Wo]:
            # Inter-head Diversity: Each head must be unique
            # Flatten weight matrix per head and calculate pairwise similarity
            w_flat = F.normalize(w.view(S, H, -1), p=2, dim=-1)
            h_sim = torch.einsum('shd,skd->shk', w_flat, w_flat)
            loss_div += 1.0 * F.mse_loss(h_sim, h_eye.expand(S, -1, -1), reduction='sum') / (S * H * H)

        # Combined Loss
        loss = loss_commit + loss_div
        
        return z_q, loss, indices, weights

# --- 5. Main Tokenizer Model (AttnVQTokenizer) ---

class AttnVQTokenizer(nn.Module):
    def __init__(
        self,
        in_chans=1, embed_dim=256, enc_depth=4, enc_heads=8, enc_mlp_ratio=4.,
        dec_depth=2, dec_heads=8, dec_mlp_ratio=4., in_scales=3,
        vq_head_top_k=8, vq_head_num=8, vq_head_vocab_size=64,
        freq_resolution=1.0, min_freq=0.0, max_freq=100.0, fs=200.0, input_length=200,
        temperature=1.0
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
        self.spatial_temporal_encoder = SpatialTemporalEncoder(in_chans, base_filters=16, embed_dim=embed_dim)
        
        self.scale_encoders = nn.ModuleList([ScaleEncoder(embed_dim, depth=enc_depth, heads=enc_heads, mlp_ratio=enc_mlp_ratio) for _ in range(in_scales)])
        
        self.attnvq = AttnVQ(in_scales, self.vq_head_num, vq_head_vocab_size, embed_dim, vq_head_top_k, temperature=temperature)
        self.scale_decoders = nn.ModuleList([ScaleDecoder(embed_dim, depth=dec_depth, heads=dec_heads, fft_dim=self.fft_dim, mlp_ratio=dec_mlp_ratio) for _ in range(in_scales)])

    def set_temperature(self, value):
        self.attnvq.set_temperature(value)

    def forward(self, x, coords):
        # Projected features with spatial embeddings: List of (B, C, embed_dim) for each scale
        h_scales_projected = self.spatial_temporal_encoder(x, coords) 
        
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

    def get_indices(self, x, coords):
        with torch.no_grad(): _, _, _, _, indices, weights = self.forward(x, coords)
        indices_flat = indices.permute(1, 2, 0, 3, 4).reshape(-1, self.in_scales, self.vq_head_num, indices.shape[4]).unsqueeze(1)
        weights_flat = weights.permute(1, 2, 0, 3, 4).reshape(-1, self.in_scales, self.vq_head_num, weights.shape[4]).unsqueeze(1)
        return indices_flat, weights_flat
