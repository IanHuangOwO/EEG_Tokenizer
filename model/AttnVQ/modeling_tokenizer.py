import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. Base Utilities & Embeddings
# ==========================================

def get_sinusoidal_pos(seq_len, dim, device):
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    sin_inp = torch.einsum("i,j->ij", t, inv_freq)
    pos_emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return pos_emb.unsqueeze(0)  # [1, SeqLen, Dim]


class SpatialTemporalEmbeddings(nn.Module):
    def __init__(self, patch_len, dim, max_patches=5000):
        super().__init__()
        self.proj = nn.Linear(patch_len, dim)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer('pos_emb', get_sinusoidal_pos(max_patches, dim, torch.device('cpu')))
        self.coord_proj = nn.Sequential(
            nn.Linear(3, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
        )

    def forward(self, x, coords=None, time_idx=None):
        B, C, N, L = x.shape
        z = self.proj(x.reshape(B * C, N, L))  # [B*C, N, D]

        # Temporal: use absolute patch positions if provided, else fall back to 0..N-1
        if time_idx is not None:
            t = time_idx.clamp(0, self.pos_emb.shape[1] - 1)  # [B, N]
            temp_emb = self.pos_emb[0][t]                      # [B, N, D]
            z = z + temp_emb.unsqueeze(1).expand(B, C, N, -1).reshape(B * C, N, -1)
        else:
            z = z + self.pos_emb[:, :N, :]

        # Spatial: project 3D coords and broadcast across all patches
        if coords is not None:
            s = self.coord_proj(coords.reshape(B * C, 3)).unsqueeze(1)  # [B*C, 1, D]
            z = z + s

        return self.norm(z).reshape(B, C, N, -1)


# ==========================================
# 2. TSA Encoder
# ==========================================

class ConvolutionalAdditiveAttention(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.qkv_conv = nn.Conv1d(dim, dim * 3, kernel_size=kernel_size, padding=kernel_size // 2)
        self.attn_weight = nn.Linear(dim, 1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """ x: [B*C, N, D] """
        qkv = self.qkv_conv(x.transpose(1, 2)).transpose(1, 2)  # [B*C, N, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)
        attn = F.softmax(self.attn_weight(q), dim=1)             # [B*C, N, 1]
        global_context = torch.sum(attn * k, dim=1, keepdim=True)
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
        x = self.fc1(x).transpose(1, 2)
        return self.fc2(self.act(self.conv(x).transpose(1, 2)))


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

        x_flat = x_flat + self.conv_attn_time(self.norm_time(x_flat))

        if self.apply_cross_dim:
            x = x_flat.view(B, C, N, D).permute(0, 2, 1, 3).reshape(B * N, C, D)
            x = x + F.scaled_dot_product_attention(
                *self.qkv_space(self.norm_space(x)).chunk(3, dim=-1)
            )
            x_flat = x.view(B, N, C, D).permute(0, 2, 1, 3).reshape(B * C, N, D)

        x_flat = x_flat + self.conv_ffn(self.norm_ffn(x_flat))
        return x_flat.view(B, C, N, D)


class TSAEncoder(nn.Module):
    def __init__(self, dim, depth=12, num_heads=8, mlp_ratio=4., apply_cross_dim=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            TSABlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, apply_cross_dim=apply_cross_dim)
            for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ==========================================
# 3. VQ Decoder Head
# ==========================================

class AttnVQ(nn.Module):
    def __init__(self, num_heads, vq_head_vocab_size, e_dim, num_discrete=5, sigmoid_gain=5.0):
        super().__init__()
        self.r, self.num_heads, self.e_dim, self.num_discrete = vq_head_vocab_size, num_heads, e_dim, num_discrete
        self.sigmoid_gain = sigmoid_gain
        self.norm = nn.LayerNorm(e_dim)
        self.A = nn.Parameter(torch.empty(e_dim, num_heads * self.r))
        nn.init.orthogonal_(self.A, gain=1.0)

        self.register_buffer('identity_h', torch.eye(num_heads))
        self.register_buffer('avg_probs', torch.zeros(num_heads, vq_head_vocab_size, num_discrete))
        self.register_buffer('max_prob_ema', torch.tensor(0.0))
        self.ema_decay = 0.99

    def forward(self, z):
        B_sz, N_c, D = z.shape
        z = self.norm(z)

        H, r, N_d = self.num_heads, self.r, self.num_discrete
        half_range = (N_d - 1) / 2.0

        q = torch.matmul(z.reshape(B_sz * N_c, D), self.A).reshape(B_sz, N_c, H, r)
        q_soft = (N_d - 1) * torch.sigmoid(self.sigmoid_gain * q) - half_range
        q_quant = torch.round(q_soft)
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

        return v_q.reshape(B_sz, N_c, H, r), self.get_joint_subspace_loss(), indices, q_soft

    def get_joint_subspace_loss(self):
        A_per_head = self.A.view(self.e_dim, self.num_heads, self.r)
        A_mean = F.normalize(A_per_head.mean(dim=-1), p=2, dim=0)
        G_head_mean = torch.matmul(A_mean.t(), A_mean)
        return torch.mean((G_head_mean * (1.0 - self.identity_h)) ** 2)
    
    @torch.no_grad()
    def get_current_metrics(self):
        metrics = {}
        H, r, D = self.num_heads, self.r, self.e_dim

        Gram = torch.matmul(self.A.t(), self.A)
        metrics['codebook_total_orthogonality'] = torch.mean((Gram - torch.eye(H * r, device=Gram.device)).pow(2)).item()

        A_per_head = self.A.view(D, H, r)
        A_mean = A_per_head.mean(dim=-1)
        G_head_mean = torch.matmul(A_mean.t(), A_mean)
        metrics['codebook_head_orthogonality'] = torch.mean((G_head_mean * (1.0 - torch.eye(H, device=G_head_mean.device))).abs()).item()

        p = self.avg_probs
        entropy = -torch.sum(p * torch.log(p + 1e-10), dim=-1)
        metrics['codebook_perplexity'] = torch.exp(entropy).mean().item()
        metrics['codebook_sharpness'] = self.max_prob_ema.item()
        metrics['codebook_active_rank'] = (p.max(dim=-1)[0] < 0.99).float().mean().item()

        s_a = torch.linalg.svdvals(self.A)
        metrics['codebook_avg_singular_value'] = s_a.mean().item()
        metrics['codebook_condition_number'] = (s_a.max() / (s_a.min() + 1e-8)).item()

        return metrics


class FastAdditiveDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, fft_dim, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.W = nn.Parameter(torch.empty(num_heads, embed_dim, 2 * fft_dim))
        self.bias = nn.Parameter(torch.zeros(2 * fft_dim))
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        with torch.no_grad():
            self.W.data.mul_(1.0 / math.sqrt(num_heads))

    def forward(self, v_q, A):
        """
        v_q: [B*C, N, H, r]
        A:   [D, H*r] parameter from AttnVQ
        """
        _, _, H, r = v_q.shape
        D = A.shape[0]
        M = torch.einsum('dhr,hdf->hrf', A.view(D, H, r), self.W)  # [H, r, 2*F]
        out = torch.einsum('bnhr,hrf->bnf', v_q, M) / math.sqrt(H) + self.bias
        return self.drop(out).sum(dim=1).chunk(2, dim=-1)  # two tensors of [B*C, F]


# ==========================================
# 4. Main Tokenizer Model
# ==========================================

class AttnVQTokenizer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        enc_depth=12,
        enc_heads=8,
        enc_mlp_ratio=4.0,
        patch_len=200,
        n_fft_trial=None,
        fs=200.0,
        decoder_heads_config=None,
        use_spatial_embedding=False
    ):
        super().__init__()
        self.patch_len = patch_len
        self.n_fft = n_fft_trial if n_fft_trial is not None else 800
        self.fs = fs

        self.use_spatial_embedding = use_spatial_embedding
        self.embed = SpatialTemporalEmbeddings(self.patch_len, embed_dim)
        self.encoder = TSAEncoder(
            embed_dim, depth=enc_depth, num_heads=enc_heads,
            mlp_ratio=enc_mlp_ratio, apply_cross_dim=False
        )

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
        full_freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / fs)

        for i, h_cfg in enumerate(decoder_heads_config):
            f_min, f_max = h_cfg["freq_range"]
            mask = (full_freqs >= f_min - 1e-5) & (full_freqs <= f_max + 1e-5)
            self.register_buffer(f'freq_mask_{i}', mask)
            fft_dim = int(mask.sum().item())
            self.vq_heads.append(AttnVQ(
                num_heads=h_cfg.get("vq_head_num", 8),
                vq_head_vocab_size=h_cfg.get("vq_head_vocab_size", 64),
                e_dim=embed_dim,
                num_discrete=h_cfg.get("vq_num_discrete", 5),
            ))
            self.decoders.append(FastAdditiveDecoder(
                embed_dim=embed_dim,
                num_heads=h_cfg.get("vq_head_num", 8),
                fft_dim=fft_dim,
                dropout=h_cfg.get("dropout", 0.0),
            ))

    def forward(self, x, coords=None, time_idx=None):
        """ x: [B, C, N, L] """
        B, C, N, L = x.shape

        z = self.embed(x, coords=coords if self.use_spatial_embedding else None, time_idx=time_idx)

        needed_stages = {h_cfg["stage_idx"] for h_cfg in self.decoder_heads_config}
        stage_outputs = {}
        for i, block in enumerate(self.encoder.blocks):
            z = block(z)
            if i in needed_stages:
                stage_outputs[i] = z

        all_pred_real, all_pred_imag, all_indices, all_weights = [], [], [], []
        total_sub_loss = 0

        for i, h_cfg in enumerate(self.decoder_heads_config):
            z_stage = stage_outputs[h_cfg["stage_idx"]]
            B_s, C_s, N_s, D_s = z_stage.shape
            z_flat = z_stage.reshape(B_s * C_s, N_s, D_s)

            v_q, sub_loss, indices, weights = self.vq_heads[i](z_flat)
            p_real, p_imag = self.decoders[i](v_q, self.vq_heads[i].A)
            total_sub_loss = total_sub_loss + sub_loss

            all_pred_real.append(p_real.reshape(B, C, -1))
            all_pred_imag.append(p_imag.reshape(B, C, -1))
            all_indices.append(indices)
            all_weights.append(weights)

        return all_pred_real, all_pred_imag, total_sub_loss, all_indices, all_weights

    @torch.no_grad()
    def get_current_metrics(self):
        head_metrics = [vq.get_current_metrics() for vq in self.vq_heads]
        if not head_metrics:
            return {}

        full_metrics = {}
        for k in head_metrics[0]:
            full_metrics[k] = sum(m[k] for m in head_metrics) / len(head_metrics)
        for i, m in enumerate(head_metrics):
            for k, v in m.items():
                full_metrics[f"{k}_head_{i}"] = v

        return full_metrics

    def get_loss(self, x, p_reals, p_imags, l_sub, x_fft=None):
        B, C, N, L = x.shape
        x_target = x.reshape(B, C, -1)
        T_actual = x_target.shape[-1]

        if x_fft is not None and x_fft.numel() > 0 and x_fft.dim() == 3:
            x_fft = x_fft.to(x.device)
        else:
            x_fft = torch.fft.rfft(x_target, n=self.n_fft, dim=-1, norm='ortho')

        l_expert_real = 0.0
        l_expert_imag = 0.0
        num_heads = len(self.decoder_heads_config)

        for i in range(num_heads):
            mask = getattr(self, f'freq_mask_{i}')
            weight = self.decoder_heads_config[i].get("loss_weight", 1.0)
            l_expert_real += weight * F.mse_loss(p_reals[i].float(), x_fft.real[..., mask])
            l_expert_imag += weight * F.mse_loss(p_imags[i].float(), x_fft.imag[..., mask])

        x_recon = self.reconstruct(p_reals, p_imags, n_samples=T_actual)
        l_mse_global = F.mse_loss(x_recon.float(), x_target[..., :x_recon.shape[-1]].float())

        l_total = (l_expert_real + l_expert_imag) / num_heads + l_mse_global + l_sub
        return l_total, l_sub, l_expert_real / num_heads, l_expert_imag / num_heads, l_mse_global

    def reconstruct(self, p_reals, p_imags, n_samples=None):
        B, C = p_reals[0].shape[:2]
        full_fft = torch.zeros((B, C, self.n_fft // 2 + 1), device=p_reals[0].device, dtype=torch.complex64)
        count = torch.zeros(self.n_fft // 2 + 1, device=p_reals[0].device)

        for i in range(len(self.decoder_heads_config)):
            mask = getattr(self, f'freq_mask_{i}')
            full_fft.real[..., mask] += p_reals[i].float()
            full_fft.imag[..., mask] += p_imags[i].float()
            count[mask] += 1.0

        recon = torch.fft.irfft(full_fft / count.clamp(min=1.0), n=self.n_fft, dim=-1, norm='ortho')
        return recon[..., :n_samples] if n_samples is not None else recon
