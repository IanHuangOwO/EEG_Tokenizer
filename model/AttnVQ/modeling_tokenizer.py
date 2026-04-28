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
        self.qkv_conv = nn.Conv1d(dim, dim * 3, kernel_size=kernel_size, padding=kernel_size // 2)
        self.attn_weight = nn.Linear(dim, 1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """ x: [B*C, N, D] """
        # One transpose for the conv
        x_t = x.transpose(1, 2)              
        qkv = self.qkv_conv(x_t).transpose(1, 2) # [B*C, N, 3*D]
        
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Additive Attention Logic
        attn = F.softmax(self.attn_weight(q), dim=1) # [B*C, N, 1]
        global_context = torch.sum(attn * k, dim=1, keepdim=True) # [B*C, 1, D]
        
        return self.proj(q * global_context * v)

class ConvFFN(nn.Module):
    def __init__(self, dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        """ x: [B*C, N, D] """
        x = self.fc1(x)
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        x = self.act(x)
        return self.fc2(x)

class TSABlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., apply_cross_dim=False):
        super().__init__()
        self.apply_cross_dim = apply_cross_dim
        self.num_heads = num_heads
        
        self.norm_time = nn.LayerNorm(dim)
        self.conv_attn_time = ConvolutionalAdditiveAttention(dim, kernel_size=3)
        
        if self.apply_cross_dim:
            self.norm_space = nn.LayerNorm(dim)
            self.qkv_space = nn.Linear(dim, dim * 3, bias=False)
            self.proj_space = nn.Linear(dim, dim)
            
        self.norm_ffn = nn.LayerNorm(dim)
        self.conv_ffn = ConvFFN(dim, hidden_dim=int(dim * mlp_ratio), kernel_size=3)

    def forward(self, x):
        B, C, N, D = x.shape
        x_flat = x.view(B * C, N, D)
        
        # --- STAGE 1: TEMPORAL ---
        x_flat = x_flat + self.conv_attn_time(self.norm_time(x_flat))
        
        # --- STAGE 2: SPATIAL ---
        if self.apply_cross_dim:
            x = x_flat.view(B, C, N, D).permute(0, 2, 1, 3).reshape(B * N, C, D)
            x = x + F.scaled_dot_product_attention(
                *self.qkv_space(self.norm_space(x)).chunk(3, dim=-1)
            ) # This assumes heads=1 for simplicity, refine if multi-head space needed
            x_flat = x.view(B, N, C, D).permute(0, 2, 1, 3).reshape(B * C, N, D)
            
        # --- STAGE 3: CONVOLUTIONAL FFN ---
        x_flat = x_flat + self.conv_ffn(self.norm_ffn(x_flat))
        
        return x_flat.view(B, C, N, D)
    
class HierarchicalTSAEncoder(nn.Module):
    def __init__(self, dim, depth=12, num_heads=8, mlp_ratio=4., apply_cross_dim=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            TSABlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, apply_cross_dim=apply_cross_dim)
            for _ in range(depth)
        ])

    def forward(self, x):
        block_outputs = []
        for block in self.blocks:
            x = block(x)
            block_outputs.append(x)

        return x, block_outputs


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
        
        self.register_buffer('identity_h', torch.eye(num_heads))
        self.register_buffer('avg_probs', torch.zeros(num_heads, vq_head_vocab_size, num_discrete))
        self.register_buffer('max_prob_ema', torch.tensor(0.0))
        self.ema_decay = 0.99

    def get_joint_subspace_loss(self):
        A_per_head = self.A.view(self.e_dim, self.num_heads, self.r)
        A_mean = A_per_head.mean(dim=-1) 
        A_mean = F.normalize(A_mean, p=2, dim=0) # Prevents loss explosion
        
        G_head_mean = torch.matmul(A_mean.t(), A_mean) 
        return torch.mean((G_head_mean * (1.0 - self.identity_h)) ** 2)
    
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
                B_total = B_sz * N_c
                flat_idx = indices.view(B_total, H * r).t().clamp(0, N_d - 1)
                
                batch_probs = torch.zeros(H * r, N_d, device=indices.device)
                batch_probs.scatter_add_(1, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
                batch_probs = (batch_probs / B_total).view(H, r, N_d)
                
                self.avg_probs.mul_(self.ema_decay).add_(batch_probs, alpha=1 - self.ema_decay)
                max_p = batch_probs.max(dim=-1)[0].mean()
                self.max_prob_ema.mul_(self.ema_decay).add_(max_p, alpha=1 - self.ema_decay)

        v_q_per_head = v_q.reshape(B_sz, N_c, H, r)
        
        # 🚀 OPTIMIZATION: Return raw v_q instead of projecting to high-dim z_q
        # z_q_heads = torch.einsum('bnhr,dhr->bnhd', v_q_per_head, A_per_head)
            
        sub_loss = self.get_joint_subspace_loss()
        return v_q_per_head, sub_loss, indices, q_scaled

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
    def __init__(self, embed_dim, num_heads, fft_dim, dropout=0.0):
        super().__init__()
        self.num_heads, self.fft_dim = num_heads, fft_dim
        
        # 🚀 COLLAPSED PHYSICS: If norm and gelu are off, we can skip the hidden dim
        self.W = nn.Parameter(torch.empty(num_heads, embed_dim, 2 * fft_dim))
        self.bias = nn.Parameter(torch.zeros(2 * fft_dim))
        self.drop = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        with torch.no_grad():
            self.W.data.mul_(1.0 / math.sqrt(num_heads))

    def forward(self, v_q, A):
        """ 
        v_q: [B*C, N, H, r]
        A: [D, Hr] parameter from VQ head
        """
        _, _, H, r = v_q.shape
        D = A.shape[0]
        
        # 🚀 ULTRA-FAST COLLAPSED PROJECTION
        # Mathematically: (v_q @ A) @ W = v_q @ (A @ W)
        # 1. Collapse A [D, H, r] and W [H, D, F] into M [H, r, F]
        A_per_head = A.view(D, H, r)
        M = torch.einsum('dhr,hdf->hrf', A_per_head, self.W) # [H, r, 2*F]
        
        # 2. Directly project VQ weights to FFT bins
        out = torch.einsum('bnhr,hrf->bnf', v_q, M) # [B*C, N, 2*F]
        out = out / math.sqrt(self.num_heads) + self.bias
        out = self.drop(out)

        # Sum over Patches (N) -> [B, C, F]
        out_trial = out.sum(dim=1)
        return out_trial.chunk(2, dim=-1)


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
        n_fft_trial=None,
        fs=200.0,
        decoder_heads_config=None
    ):
        super().__init__()
        self.patch_len = in_scales
        self.n_fft = n_fft_trial if n_fft_trial is not None else 800
        self.fs = fs

        # 1. Embed & Temporal Hierarchy
        self.embed = SpatialTemporalEmbeddings(self.patch_len, embed_dim)
        self.encoder = HierarchicalTSAEncoder(
            embed_dim, depth=enc_depth, num_heads=enc_heads, mlp_ratio=enc_mlp_ratio,
            apply_cross_dim=False
        )

        # 2. Multi-Head VQ & Decoder setup
        if decoder_heads_config is None:
            decoder_heads_config = [{
                "stage_idx": enc_depth - 1,
                "freq_range": [0.0, fs / 2.0],
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
                dropout=h_cfg.get("dropout", 0.0)
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
            
            # 1. VQ Quantize (Returns v_q instead of projecting to D)
            v_q, sub_loss, indices, weights = self.vq_heads[i](z_stage.reshape(B_s * C_s, N_s, D_s))
            total_sub_loss = total_sub_loss + sub_loss

            # 2. Decode using Collapsed Projection
            p_real_s, p_imag_s = self.decoders[i](v_q, self.vq_heads[i].A)
            
            # Reshape result to [B, C, F]
            p_real_s = p_real_s.reshape(B, C, -1)
            p_imag_s = p_imag_s.reshape(B, C, -1)

            all_pred_real.append(p_real_s)
            all_pred_imag.append(p_imag_s)
            all_indices.append(indices) 
            all_weights.append(weights)
            
        return all_pred_real, all_pred_imag, total_sub_loss, all_indices, all_weights

    @torch.no_grad()
    def get_current_metrics(self):
        # 1. Get metrics for each individual head
        head_metrics = [vq.get_current_metrics() for vq in self.vq_heads]
        if not head_metrics: return {}
        
        # 2. Compile comprehensive dictionary
        full_metrics = {}
        
        # Average metrics (Backward compatibility for plots)
        for k in head_metrics[0].keys():
            full_metrics[k] = sum(m[k] for m in head_metrics) / len(head_metrics)
            
        # Individual head metrics for deep analysis
        for i, m in enumerate(head_metrics):
            for k, v in m.items():
                full_metrics[f"{k}_head_{i}"] = v
                
        return full_metrics

    def get_loss(self, x, p_reals, p_imags, l_sub, x_fft=None):
        """
        Trial-Level Spectral + Temporal supervision with consistent scaling.
        """
        B, C, N, L = x.shape
        x_target = x.reshape(B, C, -1)  # [B, C, T]
        T_actual = x_target.shape[-1]

        # Accept precomputed trial FFT [B, C, F_trial] or compute it here
        if x_fft is not None and x_fft.numel() > 0 and x_fft.dim() == 3:
            x_fft = x_fft.to(x.device)
        else:
            x_fft = torch.fft.rfft(x_target, n=self.n_fft, dim=-1, norm='ortho')

        l_expert_real = 0.0
        l_expert_imag = 0.0
        num_heads = len(self.decoder_heads_config)

        for i in range(num_heads):
            mask = getattr(self, f'freq_mask_{i}').to(x.device)  # [F_trial] bool
            weight = self.decoder_heads_config[i].get("loss_weight", 1.0)

            # 1. Spectral L1 on trial FFT bins
            t_real = x_fft.real[..., mask]
            t_imag = x_fft.imag[..., mask]
            l_expert_real += weight * F.mse_loss(p_reals[i].float(), t_real)
            l_expert_imag += weight * F.mse_loss(p_imags[i].float(), t_imag)

        # 3. Global time-domain reconstruction MSE
        x_recon = self.reconstruct(p_reals, p_imags, n_samples=T_actual)
        T_recon = x_recon.shape[-1]
        l_mse_global = F.mse_loss(x_recon.float(), x_target[..., :T_recon].float())

        # Average expert losses so total magnitude doesn't scale with head count
        l_total = (l_expert_real + l_expert_imag) / num_heads + l_mse_global + l_sub

        return l_total, l_sub, l_expert_real/num_heads, l_expert_imag/num_heads, l_mse_global


    def reconstruct(self, p_reals, p_imags, n_samples=None):
        """
        Reconstruct trial waveform from predicted band FFTs with consistent scaling.
        """
        B, C = p_reals[0].shape[:2]
        
        # 🚀 FIX: Always use self.n_fft for irfft to maintain scaling consistency with rfft(n=self.n_fft)
        # Using norm='ortho' with a different n would change the signal magnitude.
        full_fft = torch.zeros((B, C, self.n_fft // 2 + 1),
                               device=p_reals[0].device, dtype=torch.complex64)
        
        # Use additive approach with count for averaging overlaps (more robust than overwriting)
        count = torch.zeros((self.n_fft // 2 + 1,), device=p_reals[0].device)
        for i in range(len(self.decoder_heads_config)):
            mask = getattr(self, f'freq_mask_{i}')
            full_fft.real[..., mask] += p_reals[i].float()
            full_fft.imag[..., mask] += p_imags[i].float()
            count[mask] += 1.0
        
        full_fft = full_fft / count.clamp(min=1.0)

        recon = torch.fft.irfft(full_fft, n=self.n_fft, dim=-1, norm='ortho')
        
        # Truncate to desired length
        if n_samples is not None:
            recon = recon[..., :n_samples]
        return recon