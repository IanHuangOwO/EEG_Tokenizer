import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. Base Utilities & Embeddings
# ==========================================

def get_sinusoidal_pos(seq_len, dim, device):
    """ Generates continuous sinusoidal positional embeddings dynamically. """
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    sin_inp = torch.einsum("i,j->ij", t, inv_freq)
    pos_emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return pos_emb.unsqueeze(0) #[1, SeqLen, Dim]


class SpatialTemporalEmbeddings(nn.Module):
    def __init__(self, patch_len, dim, max_patches=5000):
        super().__init__()
        self.proj = nn.Linear(patch_len, dim)
        self.norm = nn.LayerNorm(dim)
        
        # 🚀 FIX: Precompute once!
        pos_emb = get_sinusoidal_pos(max_patches, dim, torch.device('cpu'))
        self.register_buffer('pos_emb', pos_emb)

    def forward(self, x):
        B, C, N, L = x.shape
        x_flat = x.reshape(B * C, N, L)
        z = self.proj(x_flat) 
        
        # 🚀 FIX: Just slice the precomputed buffer
        z = z + self.pos_emb[:, :N, :] 
        
        z = self.norm(z)
        return z.reshape(B, C, N, -1)


# ==========================================
# 2. Efficient Hierarchical Temporal Encoder
# ==========================================

class ConvolutionalAdditiveAttention(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        # 🚀 OPTIMIZATION: Standard Conv instead of Depthwise (groups=1)
        # Much faster hardware utilization on modern GPUs
        self.qkv_conv = nn.Conv1d(dim, dim * 3, kernel_size=kernel_size, padding=kernel_size // 2)
        self.attn_weight = nn.Linear(dim, 1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        x_t = x.transpose(1, 2).contiguous()              
        qkv = self.qkv_conv(x_t)             
        qkv = qkv.transpose(1, 2).contiguous()            
        
        # Split Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)
        
        attn = F.softmax(self.attn_weight(q), dim=1)
        global_context = torch.sum(attn * k, dim=1, keepdim=True)
        return self.proj(q * global_context * v)

class ConvFFN(nn.Module):
    def __init__(self, dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        # 🚀 OPTIMIZATION: Standard Conv instead of Depthwise
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = x.transpose(1, 2).contiguous()
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()
        
        x = self.act(x)
        return self.fc2(x)

class TSABlock(nn.Module):
    """ The fully upgraded Two-Stage Attention Block """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., apply_cross_dim=False):
        super().__init__()
        self.apply_cross_dim = apply_cross_dim
        self.num_heads = num_heads
        
        # --- STAGE 1: Temporal (Conv-Attention) ---
        self.norm_time = nn.LayerNorm(dim)
        self.conv_attn_time = ConvolutionalAdditiveAttention(dim, kernel_size=3)
        
        # --- STAGE 2: Spatial (Standard Attention) ---
        if self.apply_cross_dim:
            self.norm_space = nn.LayerNorm(dim)
            self.qkv_space = nn.Linear(dim, dim * 3, bias=False)
            self.proj_space = nn.Linear(dim, dim)
            
            nn.init.zeros_(self.proj_space.weight)
            nn.init.zeros_(self.proj_space.bias)
            
        # --- STAGE 3: Convolutional FFN ---
        self.norm_ffn = nn.LayerNorm(dim)
        self.conv_ffn = ConvFFN(dim, hidden_dim=int(dim * mlp_ratio), kernel_size=3)

    def forward(self, x):
        B, C, N, D = x.shape
        
        # --- STAGE 1: TEMPORAL ---
        x_time = x.reshape(B * C, N, D)
        x_time_norm = self.norm_time(x_time)
        
        # Convolutional QKV extracts local features BEFORE global attention
        x_attn = self.conv_attn_time(x_time_norm)
        x = (x_time + x_attn).reshape(B, C, N, D)
        
        # --- STAGE 2: SPATIAL ---
        if self.apply_cross_dim:
            x_space = x.permute(0, 2, 1, 3).reshape(B * N, C, D)
            x_space_norm = self.norm_space(x_space)
            
            head_dim = D // self.num_heads
            qkv = self.qkv_space(x_space_norm).reshape(B * N, C, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn_out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B * N, C, D)
            x_space_out = self.proj_space(attn_out)
            
            x = x + x_space_out.reshape(B, N, C, D).permute(0, 2, 1, 3)
            
        # --- STAGE 3: CONVOLUTIONAL FFN ---
        x_flat = x.reshape(B * C, N, D) 
        x_ffn_norm = self.norm_ffn(x_flat)
        x_ffn_out = self.conv_ffn(x_ffn_norm)
        x = x + x_ffn_out.reshape(B, C, N, D)
        
        return x
    
class TemporalPatchMerging(nn.Module):
    """ Safe downsampling that skips if N is too short. """
    def __init__(self, dim, downscale_factor=2):
        super().__init__()
        self.factor = downscale_factor
        self.downsample = nn.Conv1d(dim, dim, kernel_size=self.factor, stride=self.factor)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, N, D = x.shape
        if N < self.factor: return x # Safety Check
            
        x_flat = x.reshape(B * C, N, D).transpose(1, 2)
        x_down = self.downsample(x_flat).transpose(1, 2)
        
        return self.norm(x_down).reshape(B, C, -1, D)

class HierarchicalTSAEncoder(nn.Module):
    def __init__(self, dim, depth=12, num_heads=8, mlp_ratio=4., merge_factors=[1, 2, 2], apply_cross_dim=False):
        super().__init__()
        self.stages = nn.ModuleList()
        self.num_stages = len(merge_factors)
        self.stage_weights = nn.Parameter(torch.ones(self.num_stages))
        
        depth_per_stage = max(1, depth // self.num_stages)
        for i, m in enumerate(merge_factors):
            stage_blocks = nn.ModuleList()
            for _ in range(depth_per_stage):
                stage_blocks.append(
                    TSABlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, apply_cross_dim=apply_cross_dim)
                )
            self.stages.append(nn.ModuleDict({
                'merge': TemporalPatchMerging(dim, m) if m > 1 else nn.Identity(),
                'blocks': stage_blocks
            }))

    def forward(self, x):
        B, C = x.shape[0], x.shape[1]
        stage_outputs = []
        
        for stage in self.stages:
            x = stage['merge'](x)
            for block in stage['blocks']:
                x = block(x)
            stage_outputs.append(x)
            
        final_out = stage_outputs[-1]
        N_coarse = final_out.shape[2]
        weights = F.softmax(self.stage_weights, dim=0)
        
        # 🚀 FIX: Ultra-Fast Multi-Scale Weighted Sum (with safety fallback)
        fused_x = 0
        for w, feat in zip(weights, stage_outputs):
            N_current = feat.shape[2]
            
            if N_current != N_coarse:
                if N_current % N_coarse == 0:
                    factor = N_current // N_coarse
                    # Fast path: Reshape and mean
                    feat = feat.reshape(B, C, N_coarse, factor, -1).mean(dim=3)
                else:
                    # Safety Fallback: Adaptive pool for odd dimensions
                    feat = F.adaptive_avg_pool1d(feat.transpose(1, 2), N_coarse).transpose(1, 2)
                
            fused_x = fused_x + w * feat
            
        return fused_x, stage_outputs

    @torch.no_grad()
    def get_fusion_metrics(self):
        weights = F.softmax(self.stage_weights, dim=0)
        return {f'tsa_fusion_stage_{i+1}': w.item() for i, w in enumerate(weights)}


# ==========================================
# 3. Tokenizer Components
# ==========================================

class AttnVQ(nn.Module):
    def __init__(self, num_heads, vq_head_vocab_size, e_dim, num_discrete=5):
        super().__init__()
        self.r, self.num_heads, self.e_dim, self.num_discrete = vq_head_vocab_size, num_heads, e_dim, num_discrete
        self.norm = nn.LayerNorm(e_dim) # Prevents sigmoid saturation / dead codebook
        self.A = nn.Parameter(torch.empty(e_dim, num_heads * self.r))
        nn.init.orthogonal_(self.A, gain=1.0)
        
        self.register_buffer('avg_probs', torch.zeros(num_heads, vq_head_vocab_size, num_discrete))
        self.register_buffer('max_prob_ema', torch.tensor(0.0))
        self.ema_decay = 0.99

    def get_joint_subspace_loss(self):
        A_per_head = self.A.view(self.e_dim, self.num_heads, self.r)
        A_mean = A_per_head.mean(dim=-1) 
        A_mean = F.normalize(A_mean, p=2, dim=0) # Prevents loss explosion
        
        G_head_mean = torch.matmul(A_mean.t(), A_mean) 
        identity_h = torch.eye(self.num_heads, device=G_head_mean.device)
        return torch.mean((G_head_mean * (1.0 - identity_h)) ** 2)
    
    def forward(self, z):
        B_sz, N_c, D = z.shape
        z = self.norm(z) 
        
        H, r, N_d = self.num_heads, self.r, self.num_discrete
        half_range = (N_d - 1) / 2.0
        
        q = torch.matmul(z.reshape(B_sz * N_c, D), self.A).reshape(B_sz, N_c, H, r)
        q_soft = (N_d - 1) * torch.sigmoid(q) - half_range
        
        q_scaled = q_soft + (torch.rand_like(q_soft) - 0.5) * 0.4 if self.training else q_soft
        q_quant = torch.round(q_scaled)
        v_q = q_soft + (q_quant - q_soft).detach()
        indices = (q_quant + half_range).long()
        
        if self.training:
            with torch.no_grad():
                # 🚀 FIX: Memory leak fix using scatter_add_ instead of one_hot
                B_total = B_sz * N_c
                flat_idx = indices.view(B_total, H * r).t().clamp(0, N_d - 1)
                
                batch_probs = torch.zeros(H * r, N_d, device=indices.device)
                batch_probs.scatter_add_(1, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
                batch_probs = (batch_probs / B_total).view(H, r, N_d)
                
                self.avg_probs.mul_(self.ema_decay).add_(batch_probs, alpha=1 - self.ema_decay)
                max_p = batch_probs.max(dim=-1)[0].mean()
                self.max_prob_ema.mul_(self.ema_decay).add_(max_p, alpha=1 - self.ema_decay)

        v_q_per_head = v_q.reshape(B_sz, N_c, H, r)
        A_per_head = self.A.view(D, H, r)
        z_q_heads = torch.einsum('bnhr,dhr->bnhd', v_q_per_head, A_per_head)
        sub_loss = self.get_joint_subspace_loss()
        
        return z_q_heads, sub_loss, indices, q_scaled

    @torch.no_grad()
    def get_current_metrics(self):
        metrics = {}
        H, r, D = self.num_heads, self.r, self.e_dim
        
        A_flat = self.A
        Gram = torch.matmul(A_flat.t(), A_flat)
        identity = torch.eye(H * r, device=Gram.device)
        metrics['subspace_ortho'] = torch.mean((Gram - identity).pow(2)).item()
        
        A_per_head = self.A.view(D, H, r)
        A_mean = A_per_head.mean(dim=-1)
        G_head_mean = torch.matmul(A_mean.t(), A_mean)
        identity_h = torch.eye(H, device=G_head_mean.device)
        metrics['head_cross_corr'] = torch.mean((G_head_mean * (1.0 - identity_h)).abs()).item()

        p = self.avg_probs
        entropy = -torch.sum(p * torch.log(p + 1e-10), dim=-1)
        metrics['codebook_perplexity'] = torch.exp(entropy).mean().item()
        metrics['codebook_sharpness'] = self.max_prob_ema.item()
        
        active_mask = (p.max(dim=-1)[0] < 0.99).float()
        metrics['active_rank_ratio'] = active_mask.mean().item()
        
        s_a = torch.linalg.svdvals(self.A)
        metrics['A_sing_val_avg'] = s_a.mean().item()
        metrics['A_cond'] = (s_a.max() / (s_a.min() + 1e-8)).item()
            
        return metrics

class FastAdditiveDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, fft_dim, dropout=0.0, use_norm=True, use_gelu=False):
        super().__init__()
        self.num_heads, self.fft_dim = num_heads, fft_dim
        
        # 🚀 ADDED: Optional Norm, GELU, and Dropout for better stability/capacity
        self.norm = nn.LayerNorm(embed_dim) if use_norm else nn.Identity()
        self.W = nn.Parameter(torch.empty(num_heads, embed_dim, 2 * fft_dim))
        self.bias = nn.Parameter(torch.zeros(2 * fft_dim))
        self.act = nn.GELU() if use_gelu else nn.Identity()
        self.drop = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        with torch.no_grad():
            self.W.data.mul_(1.0 / math.sqrt(num_heads))

    def forward(self, z_q_heads):
        """ z_q_heads: [B, C, N, H, D] """
        # Apply Norm per head
        z = self.norm(z_q_heads)
        
        B, C, N, H, D = z.shape
        z_flat = z.reshape(B, C, N, H * D)
        W_flat = self.W.view(H * D, 2 * self.fft_dim)
        
        # Additive Projection
        out = torch.matmul(z_flat, W_flat) / math.sqrt(self.num_heads) + self.bias
        
        # Apply Act & Dropout
        out = self.act(out)
        out = self.drop(out)
        
        return out.chunk(2, dim=-1)


# ==========================================
# 4. MAIN TOKENIZER MODEL
# ==========================================

class AttnVQTokenizer(nn.Module):
    def __init__(
        self,
        in_chans=1,
        embed_dim=256, 
        enc_depth=12,
        enc_heads=8,
        enc_mlp_ratio=4.0,
        in_scales=200, 
        merge_factors=[1, 2, 2],
        freq_resolution=1.0, min_freq=0.0, max_freq=100.0, fs=200.0,
        decoder_heads_config=None 
    ):
        super().__init__()
        self.patch_len = in_scales 
        self.n_fft = int(fs / freq_resolution)
        self.fs = fs
        
        # 1. Embed & Temporal Hierarchy
        self.embed = SpatialTemporalEmbeddings(self.patch_len, embed_dim)
        self.encoder = HierarchicalTSAEncoder(
            embed_dim, depth=enc_depth, num_heads=enc_heads, mlp_ratio=enc_mlp_ratio,
            merge_factors=merge_factors, apply_cross_dim=False
        )
        
        # 2. Multi-Head VQ & Decoder setup
        if decoder_heads_config is None:
            # Default fallback: one head on the last stage
            decoder_heads_config = [{
                "stage_idx": len(merge_factors) - 1, 
                "freq_range": [min_freq, max_freq],
                "vq_head_num": 8,
                "vq_head_vocab_size": 64,
                "vq_num_discrete": 5
            }]
        
        self.decoder_heads_config = decoder_heads_config
        self.vq_heads = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        full_freqs = torch.fft.rfftfreq(self.n_fft, d=1.0/fs)
        
        for i, h_cfg in enumerate(decoder_heads_config):
            vq = AttnVQ(
                num_heads=h_cfg.get("vq_head_num", 8),
                vq_head_vocab_size=h_cfg.get("vq_head_vocab_size", 64),
                e_dim=embed_dim,
                num_discrete=h_cfg.get("vq_num_discrete", 5)
            )
            self.vq_heads.append(vq)
            
            f_min, f_max = h_cfg["freq_range"]
            mask = (full_freqs >= f_min - 1e-5) & (full_freqs <= f_max + 1e-5)
            self.register_buffer(f'freq_mask_{i}', mask)
            
            fft_dim = int(mask.sum().item())
            dec = FastAdditiveDecoder(
                embed_dim, 
                num_heads=h_cfg.get("vq_head_num", 8), 
                fft_dim=fft_dim,
                dropout=h_cfg.get("dropout", 0.0),
                use_norm=h_cfg.get("use_norm", True),
                use_gelu=h_cfg.get("use_gelu", False)
            )
            self.decoders.append(dec)

    def forward(self, x, coords=None, time_idx=None):
        """ x:[B, C, N, L] """
        B, C, N, L = x.shape
        
        # 1. Embed ->[B, C, N, D]
        z = self.embed(x).reshape(B, C, N, -1)
        
        # 2. Temporal Encode -> Get all stages
        _, stage_outputs = self.encoder(z)
        
        all_pred_real = []
        all_pred_imag = []
        all_indices = []
        all_weights = []
        total_sub_loss = 0
        
        for i, h_cfg in enumerate(self.decoder_heads_config):
            stage_idx = h_cfg["stage_idx"]
            z_stage = stage_outputs[stage_idx]
            B_s, C_s, N_s, D_s = z_stage.shape
            
            # 1. VQ Quantize (at coarse resolution N_s)
            z_q_heads, sub_loss, indices, weights = self.vq_heads[i](z_stage.reshape(B_s * C_s, N_s, D_s))
            total_sub_loss = total_sub_loss + sub_loss
            
            # 2. Decode frequency band (at coarse resolution N_s)
            H = h_cfg.get("vq_head_num", 8)
            z_q_heads = z_q_heads.reshape(B, C, N_s, H, -1)
            p_real_s, p_imag_s = self.decoders[i](z_q_heads) # [B, C, N_s, F_head]
            
            # 3. Upsample frequencies to original N resolution
            up_mode = h_cfg.get("upsample_mode", "nearest")
            align_corners = None if up_mode == "nearest" else False
            
            p_real = F.interpolate(p_real_s.reshape(B*C, N_s, -1).transpose(1, 2), 
                                   size=N, mode=up_mode, align_corners=align_corners).transpose(1, 2)
            p_imag = F.interpolate(p_imag_s.reshape(B*C, N_s, -1).transpose(1, 2), 
                                   size=N, mode=up_mode, align_corners=align_corners).transpose(1, 2)
            
            all_pred_real.append(p_real.reshape(B, C, N, -1))
            all_pred_imag.append(p_imag.reshape(B, C, N, -1))
            all_indices.append(indices) 
            all_weights.append(weights)
            
        return all_pred_real, all_pred_imag, total_sub_loss, all_indices, all_weights

    @torch.no_grad()
    def get_current_metrics(self):
        # Average VQ metrics across all heads
        all_m = [vq.get_current_metrics() for vq in self.vq_heads]
        if not all_m: return {}
        avg_m = {k: sum(m[k] for m in all_m) / len(all_m) for k in all_m[0].keys()}
        return avg_m

    def get_loss(self, x, p_reals, p_imags, l_sub, x_fft=None):
        """
        x: [B, C, N, L] original patches
        p_reals/p_imags: lists of [B, C, N, F_head] predictions
        """
        B, C, N, L = x.shape
        x_target = x.reshape(B, C, -1) # [B, C, T]
        
        # 1. Trial-Level Target FFT
        # We compute this once per batch.
        target_fft_full = torch.fft.rfft(x_target, n=self.n_fft, dim=-1, norm='ortho')
        
        # 2. Fully Reconstruct Signal in Time Domain
        # This aggregates all heads and performs one irfft.
        x_recon = self.reconstruct(p_reals, p_imags, n_samples=x_target.shape[-1])
        
        # 3. Trial-Level Reconstructed FFT
        # We compute FFT on the reconstructed signal to compare in frequency domain.
        recon_fft_full = torch.fft.rfft(x_recon, n=self.n_fft, dim=-1, norm='ortho')
        
        l_rec_spectral = 0
        l_real_total = 0
        l_imag_total = 0
        
        # 4. Compare Spectral Components per Head range
        # This ensures each head is only penalized for its assigned expert range,
        # but on the global trial-level spectrum.
        for i in range(len(self.decoder_heads_config)):
            mask = getattr(self, f'freq_mask_{i}')
            
            # Target components
            t_real = target_fft_full.real[..., mask]
            t_imag = target_fft_full.imag[..., mask]
            
            # Predicted components (from the global reconstruction)
            r_real = recon_fft_full.real[..., mask]
            r_imag = recon_fft_full.imag[..., mask]
            
            lr = F.mse_loss(r_real, t_real)
            li = F.mse_loss(r_imag, t_imag)
            
            l_real_total = l_real_total + lr
            l_imag_total = l_imag_total + li
            l_rec_spectral = l_rec_spectral + (lr + li)
        
        # 5. Time-domain MSE (already reconstructed)
        l_mse = F.mse_loss(x_recon, x_target)
        
        l_total = l_rec_spectral + l_sub + l_mse
        return l_total, l_real_total, l_imag_total, l_rec_spectral, l_sub, l_mse

    def reconstruct(self, p_reals, p_imags, n_samples=None):
        B, C, N, _ = p_reals[0].shape
        # Initialize the complex buffer
        full_fft = torch.zeros((B, C, N, self.n_fft // 2 + 1), 
                               device=p_reals[0].device, dtype=torch.complex64)

        # 🚀 MEMORY OPTIMIZATION: Assign directly to views to avoid temporary complex tensors
        for i in range(len(self.decoder_heads_config)):
            mask = getattr(self, f'freq_mask_{i}')
            full_fft.real[..., mask] = full_fft.real[..., mask] + p_reals[i]
            full_fft.imag[..., mask] = full_fft.imag[..., mask] + p_imags[i]

        # 1. Inverse FFT at patch level -> [B, C, N, self.n_fft]
        x_recon_patches = torch.fft.irfft(full_fft, n=self.n_fft, dim=-1, norm='ortho')

        # 2. Slice each patch to its actual temporal width L
        # If n_samples is provided, it's total samples. L = n_samples // N
        if n_samples is not None:
             L = n_samples // N
             # Slice the time dimension of each patch
             x_recon_patches = x_recon_patches[..., :L]

        # 3. Stitch back to [B, C, Total_Time]
        return x_recon_patches.reshape(B, C, -1)