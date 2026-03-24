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
    def __init__(self, in_chans=1, in_scales=3, base_filters=16, embed_dim=256):
        super().__init__()
        self.in_scales = in_scales
        self.conv_stages = nn.ModuleList()
        curr_filters = base_filters
        
        # Stage 0 (Stem)
        self.conv_stages.append(ConvBlock2D(in_chans, curr_filters, 3, 1, 1))
        
        # Downsampling Stages
        for i in range(1, in_scales):
            out_filters = curr_filters * 2
            dilation = 2 ** (i - 1) if i > 1 else 1
            self.conv_stages.append(
                ConvBlock2D(curr_filters, out_filters, 3, 2, 1, dilation=dilation)
            )
            curr_filters = out_filters
            
        self.projections = nn.ModuleList([nn.LazyLinear(embed_dim) for _ in range(in_scales)])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(in_scales)])
        
        self.spatial_mlp = nn.Sequential(
            nn.Linear(3, embed_dim), 
            nn.GELU(), 
            nn.Linear(embed_dim, embed_dim)
        )
        nn.init.zeros_(self.spatial_mlp[-1].weight)
        nn.init.zeros_(self.spatial_mlp[-1].bias)
        
        self.register_buffer('inv_freq', 1.0 / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim)))
        self.fusion_weights = nn.Parameter(torch.ones(in_scales))

    @torch.no_grad()
    def get_fusion_metrics(self):
        weights = F.softmax(self.fusion_weights, dim=0)
        return {f'fusion_weight_s{i}': w.item() for i, w in enumerate(weights)}

    def get_sinusoidal_emb(self, t):
        if t.dim() == 1: t = t.unsqueeze(-1)
        sin_inp = torch.einsum("bi,j->bij", t.float(), self.inv_freq)
        return torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)

    def forward(self, x, coords, time_idx=None):
        if x.dim() == 4:
            B, C, P, T = x.shape
            x = x.permute(0, 2, 1, 3).reshape(B * P, C, T)
            if coords.dim() == 3:
                coords = coords.unsqueeze(1).expand(-1, P, -1, -1).reshape(B * P, C, 3)
            if time_idx is not None:
                time_idx = time_idx.repeat(B) if time_idx.dim() == 1 else time_idx.view(-1)

        x = x.unsqueeze(1) 
        raw_feats = []
        curr_feat = x
        for stage in self.conv_stages:
            curr_feat = stage(curr_feat)
            raw_feats.append(curr_feat)
            
        spatial_emb = self.spatial_mlp(coords) 
        t_emb = self.get_sinusoidal_emb(time_idx) if time_idx is not None else 0
        
        weights = F.softmax(self.fusion_weights, dim=0)
        projected_feats = 0
        for i, (feat, proj, norm) in enumerate(zip(raw_feats, self.projections, self.norms)):
            z = proj(feat.permute(0, 2, 1, 3).reshape(feat.shape[0], feat.shape[2], -1)) 
            z = norm(z + spatial_emb + t_emb)
            projected_feats = projected_feats + z * weights[i]
            
        return projected_feats

# --- 3. Transformer Components ---

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x): return self.norm(self.block(x))

class Encoder(nn.Module):
    def __init__(self, embed_dim, depth, heads, mlp_ratio=4.):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(embed_dim, heads, mlp_ratio) for _ in range(depth)])
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embed_dim, depth, heads, fft_dim, mlp_ratio=4.):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(embed_dim, heads, mlp_ratio) for _ in range(depth)])
        self.head_amp = nn.Linear(embed_dim, fft_dim)
        self.head_sin = nn.Linear(embed_dim, fft_dim)
        self.head_cos = nn.Linear(embed_dim, fft_dim)
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return self.head_amp(x), self.head_sin(x), self.head_cos(x)

# --- 4. Quantization (Refined Low-Rank Subspace Expert) ---

class AttnVQ(nn.Module):
    def __init__(self, num_heads, vq_head_vocab_size, e_dim, num_discrete=5, decay=0.99):
        super().__init__()
        self.r = vq_head_vocab_size
        self.num_heads, self.e_dim = num_heads, e_dim
        self.num_discrete = num_discrete
        
        # A (Analysis/Encoder): Projects D -> Hr
        self.A = nn.Parameter(torch.empty(e_dim, num_heads * self.r))
        # B (Synthesis/Decoder): Projects Hr -> D
        self.B = nn.Parameter(torch.empty(num_heads * self.r, e_dim))
        
        nn.init.orthogonal_(self.A, gain=1.0)
        nn.init.orthogonal_(self.B, gain=1.0)
        
        self.register_buffer('avg_probs', torch.ones(num_heads, num_discrete) / num_discrete)
        self.register_buffer('max_prob_ema', torch.tensor(1.0 / num_discrete))

    def get_joint_subspace_loss(self):
        """
        Refined loss using a Joint Gram Matrix (B * A).
        Enforces:
        1. Symmetry (Intra-head diagonal blocks approx Identity)
        2. Independence (Inter-head off-diagonal blocks approx Zero)
        """
        # G is (Hr, Hr)
        G = torch.matmul(self.B, self.A) 
        Hr = self.num_heads * self.r
        
        # Identity matrix for the subspace dimensions
        I_hr = torch.eye(Hr, device=self.A.device)
        
        # Mask for the r x r blocks along the diagonal (Intra-head)
        mask_intra = torch.block_diag(*[torch.ones(self.r, self.r, device=self.A.device) for _ in range(self.num_heads)])
        
        # Loss 1: Symmetry. Ensure B is the 'honest' decoder for A within each head.
        loss_sym = F.mse_loss(G * mask_intra, I_hr * mask_intra)
        
        # Loss 2: Diversity. Ensure different heads don't share features.
        loss_div = torch.mean((G * (1.0 - mask_intra)) ** 2)
        
        return loss_sym + loss_div

    @torch.no_grad()
    def get_current_metrics(self):
        metrics = {}
        H, r, D = self.num_heads, self.r, self.e_dim
        N = self.num_discrete
        
        # Gram Matrix Metrics
        metrics['subspace_loss'] = self.get_joint_subspace_loss().item()
        G = torch.matmul(self.B, self.A)
        mask_intra = torch.block_diag(*[torch.ones(r, r, device=G.device) for _ in range(H)])
        metrics['subspace_symmetry_err'] = F.mse_loss(G * mask_intra, torch.eye(H*r, device=G.device) * mask_intra).item()
        metrics['subspace_cross_head_corr'] = torch.mean((G * (1-mask_intra))**2).item()
        
        # Perplexity & Sharpness
        p = self.avg_probs
        entropy = -torch.sum(p * torch.log(p + 1e-10), dim=-1)
        metrics['codebook_perplexity'] = torch.exp(entropy).mean().item() * (r / 2.0)
        metrics['codebook_sharpness'] = self.max_prob_ema.item()
        
        # Matrix Health (Singular Values & Condition Number)
        s_a = torch.linalg.svdvals(self.A)
        s_b = torch.linalg.svdvals(self.B)
        metrics['A_sing_val_avg'] = s_a.mean().item()
        metrics['B_sing_val_avg'] = s_b.mean().item()
        metrics['A_cond'] = (s_a.max() / (s_a.min() + 1e-8)).item()
        metrics['B_cond'] = (s_b.max() / (s_b.min() + 1e-8)).item()
            
        return metrics

    def forward(self, z):
        B_sz, C, D = z.shape
        H, r = self.num_heads, self.r
        N = self.num_discrete
        half_range = (N - 1) / 2.0
        
        # --- ENCODE ---
        z_flat = z.view(B_sz * C, D)
        q = torch.matmul(z_flat, self.A).view(B_sz, C, H, r)
        
        q_soft = (N - 1) * torch.sigmoid(q * 5) - half_range
        
        if self.training:
            q_scaled = q_soft + (torch.rand_like(q_soft) - 0.5) * 0.4
        else:
            q_scaled = q_soft
            
        q_quant = torch.round(q_scaled)
        v_q = q_soft + (q_quant - q_soft).detach()
        indices = (q_quant + half_range).long().unsqueeze(-1)
        
        if self.training:
            with torch.no_grad():
                q_one_hot = F.one_hot(torch.clamp(indices.squeeze(-1), 0, N - 1), num_classes=N).float()
                self.avg_probs.mul_(0.99).add_(q_one_hot.mean(dim=(0, 1, 3)), alpha=0.01)
                self.max_prob_ema.mul_(0.99).add_(1.0 - 2.0 * torch.abs(q_soft - torch.round(q_soft)).mean(), alpha=0.01)
                
        # --- DECODE ---
        v_q_flat = v_q.reshape(B_sz * C, H * r)
        z_q_soft = torch.matmul(v_q_flat, self.B).view(B_sz, C, D)
        
        # # --- LOSSES ---
        z_q = z + (z_q_soft - z).detach()
        # loss_sharp = F.mse_loss(q_soft, q_quant.detach())

        # Integrated Subspace Loss
        loss_subspace = self.get_joint_subspace_loss()

        return z_q, loss_subspace, indices, q_scaled


# --- 5. Main Tokenizer Model (AttnVQTokenizer) ---

class AttnVQTokenizer(nn.Module):
    def __init__(
        self,
        in_chans=1, in_scales=3, embed_dim=256, enc_depth=4, enc_heads=8, enc_mlp_ratio=4.,
        dec_depth=2, dec_heads=8, dec_mlp_ratio=4.,
        vq_head_num=8, vq_head_vocab_size=64, vq_num_discrete=5,
        freq_resolution=1.0, min_freq=0.0, max_freq=100.0, fs=200.0, input_length=200,
    ):
        super().__init__()
        self.embed_dim, self.vq_head_num, self.fs, self.input_length = embed_dim, vq_head_num, fs, input_length
        self.fft_dim = int(round((max_freq - min_freq) / freq_resolution)) + 1 
        self.n_fft = int(self.fs / freq_resolution)
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0/self.fs)
        mask = (freqs >= min_freq - 1e-5) & (freqs <= max_freq + 1e-5)
        self.register_buffer('freq_mask', mask)
        self.register_buffer('freq_indices', torch.where(mask)[0])
        
        self.spatial_temporal_encoder = SpatialTemporalEncoder(in_chans, in_scales, base_filters=16, embed_dim=embed_dim)
        self.encoder = Encoder(embed_dim, depth=enc_depth, heads=enc_heads, mlp_ratio=enc_mlp_ratio)
        self.attnvq = AttnVQ(self.vq_head_num, vq_head_vocab_size, embed_dim, num_discrete=vq_num_discrete)
        self.decoder = Decoder(embed_dim, depth=dec_depth, heads=dec_heads, fft_dim=self.fft_dim, mlp_ratio=dec_mlp_ratio)

    @torch.no_grad()
    def get_current_metrics(self):
        metrics = self.attnvq.get_current_metrics()
        metrics.update(self.spatial_temporal_encoder.get_fusion_metrics())
        return metrics

    def forward(self, x, coords, time_idx=None):
        is_trial = (x.dim() == 4)
        z_projected = self.spatial_temporal_encoder(x, coords, time_idx) 
        z_enc = self.encoder(z_projected)
        z_q, sub_loss, top_k_indices, weights = self.attnvq(z_enc)
        pred_amp, pred_sin, pred_cos = self.decoder(z_q)
        
        if is_trial:
            B, C, P = x.shape[0], x.shape[1], x.shape[2]
            top_k_indices = top_k_indices.view(B, P, C, self.vq_head_num, 1)
            weights = weights.view(B, P, C, self.vq_head_num, -1)
            pred_amp = pred_amp.view(B, P, C, -1)
            pred_sin = pred_sin.view(B, P, C, -1)
            pred_cos = pred_cos.view(B, P, C, -1)
            
        return pred_amp, pred_sin, pred_cos, sub_loss, top_k_indices, weights

    def get_loss(self, x, pred_amp, pred_sin, pred_cos, x_fft=None):
        if x_fft is None: x_fft = torch.fft.rfft(x, n=self.n_fft, dim=-1)
        target_fft = x_fft[..., self.freq_mask]
        min_len = min(target_fft.shape[-1], pred_amp.shape[-1])
        target_fft, pred_amp, pred_sin, pred_cos = target_fft[..., :min_len], pred_amp[..., :min_len], pred_sin[..., :min_len], pred_cos[..., :min_len]
        
        gt_amp, gt_phase = torch.abs(target_fft), torch.angle(target_fft)
        loss_amp = F.mse_loss(pred_amp, torch.log1p(gt_amp))
        loss_phase = F.mse_loss(pred_sin, torch.sin(gt_phase)) + F.mse_loss(pred_cos, torch.cos(gt_phase))
        x_recon = self.reconstruct(pred_amp, pred_sin, pred_cos, n_samples=x.shape[-1])
        loss_recon = F.mse_loss(x_recon, x)
        
        return loss_amp + loss_phase + loss_recon, loss_amp, loss_phase, loss_recon, loss_recon

    def reconstruct(self, pred_amp, pred_sin, pred_cos, n_samples=200):
        amp = torch.clamp(torch.exp(pred_amp) - 1, min=0) 
        norm = torch.sqrt(pred_cos**2 + pred_sin**2 + 1e-8)
        z_pred = torch.complex(amp * (pred_cos / norm), amp * (pred_sin / norm))
        full_z = torch.zeros((*z_pred.shape[:-1], self.n_fft // 2 + 1), dtype=z_pred.dtype, device=z_pred.device)
        count = min(len(self.freq_indices), z_pred.shape[-1])
        full_z[..., self.freq_indices[:count]] = z_pred[..., :count]
        return torch.fft.irfft(full_z, n=self.n_fft, dim=-1)[..., :n_samples]

    def get_indices(self, x, coords, time_idx=None):
        with torch.no_grad(): 
            _, _, _, _, indices, weights = self.forward(x, coords, time_idx)
        return indices.reshape(-1, self.vq_head_num, indices.shape[-1]).unsqueeze(1), weights.reshape(-1, self.vq_head_num, weights.shape[-1]).unsqueeze(1)