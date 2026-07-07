import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize


class _PerHeadOrthogonal(nn.Module):
    def __init__(self, num_heads, vq_head_vocab_size):
        super().__init__()
        self.H = num_heads
        self.r = vq_head_vocab_size

    def forward(self, A):
        D = A.shape[0]
        Q, _ = torch.linalg.qr(A.view(D, self.H, self.r).permute(1, 0, 2))  # [H, D, r]
        return Q.permute(1, 0, 2).reshape(D, self.H * self.r)


# ==========================================
# Embeddings
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
        self.spatial_active = False
        _coord_out = nn.Linear(dim // 4, dim)
        nn.init.zeros_(_coord_out.weight)
        nn.init.zeros_(_coord_out.bias)
        self.coord_proj = nn.Sequential(
            nn.Linear(3, dim // 4),
            nn.GELU(),
            _coord_out,
        )

    def enable_spatial(self):
        self.spatial_active = True

    def forward(self, x, coords=None, time_idx=None):
        B, C, N, L = x.shape
        z = self.proj(x.reshape(B * C, N, L))  # [B*C, N, D]

        if time_idx is not None:
            t = time_idx.clamp(0, self.pos_emb.shape[1] - 1)
            temp_emb = self.pos_emb[0][t]       # [B, N, D]
            z = z + temp_emb.unsqueeze(1).expand(B, C, N, -1).reshape(B * C, N, -1)
        else:
            z = z + self.pos_emb[:, :N, :]

        if coords is not None and self.spatial_active:
            s = self.coord_proj(coords.reshape(B * C, 3)).unsqueeze(1)  # [B*C, 1, D]
            z = z + s

        return self.norm(z).reshape(B, C, N, -1)


# ==========================================
# TSA Encoder
# ==========================================

class ConvolutionalAdditiveAttention(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.qkv_conv = nn.Conv1d(dim, dim * 3, kernel_size=kernel_size, padding=kernel_size // 2)
        self.attn_weight = nn.Linear(dim, 1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """ x: [B*C, N, D] """
        qkv = self.qkv_conv(x.transpose(1, 2)).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        attn = F.softmax(self.attn_weight(q), dim=1)
        global_context = torch.sum(attn * k, dim=1, keepdim=True)
        return self.proj(q * global_context * v)


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        """ x: [B*C, N, D] """
        return self.fc2(self.act(self.fc1(x)))


class TSABlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.norm_time = nn.LayerNorm(dim)
        self.conv_attn_time = ConvolutionalAdditiveAttention(dim, kernel_size=3)

        self.norm_space = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FFN(dim, hidden_dim=int(dim * mlp_ratio))

        self.spatial_active = False
        nn.init.zeros_(self.spatial_attn.out_proj.weight)
        nn.init.zeros_(self.spatial_attn.out_proj.bias)

    def enable_spatial(self):
        self.spatial_active = True

    def forward(self, x):
        B, C, N, D = x.shape
        x_flat = x.view(B * C, N, D)

        x_flat = x_flat + self.conv_attn_time(self.norm_time(x_flat))

        x_space = x_flat.view(B, C, N, D).permute(0, 2, 1, 3).reshape(B * N, C, D)
        if self.spatial_active:
            x_norm = self.norm_space(x_space)
            attn_out, _ = self.spatial_attn(x_norm, x_norm, x_norm)
            x_space = x_space + attn_out
        x_flat = x_space.view(B, N, C, D).permute(0, 2, 1, 3).reshape(B * C, N, D)

        x_flat = x_flat + self.ffn(self.norm_ffn(x_flat))
        return x_flat.view(B, C, N, D)


class TSAEncoder(nn.Module):
    def __init__(self, dim, depth=12, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.blocks = nn.ModuleList([
            TSABlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

    def enable_spatial(self):
        for block in self.blocks:
            block.enable_spatial()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ==========================================
# VQ & Decoder
# ==========================================

class MeFSQ(nn.Module):
    def __init__(self, num_heads, vq_head_vocab_size, e_dim, num_discrete=5, sigmoid_gain=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.r = vq_head_vocab_size
        self.e_dim = e_dim
        self.num_discrete = num_discrete
        self.sigmoid_gain = sigmoid_gain

        self.norm = nn.LayerNorm(e_dim)
        self.A = nn.Parameter(torch.empty(e_dim, num_heads * vq_head_vocab_size))
        nn.init.orthogonal_(self.A, gain=1.0)
        parametrize.register_parametrization(self, 'A', _PerHeadOrthogonal(num_heads, vq_head_vocab_size))

        self.register_buffer('avg_probs',        torch.zeros(num_heads, vq_head_vocab_size, num_discrete))
        self.register_buffer('max_prob_ema',     torch.tensor(0.0))
        self.register_buffer('ema_ste_gap',      torch.tensor(0.0))
        self.register_buffer('ema_head_ppl_std', torch.tensor(0.0))
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

        if not self.training:
            with torch.no_grad():
                B_total = B_sz * N_c
                flat_idx = indices.view(B_total, H * r).t().clamp(0, N_d - 1)
                batch_probs = torch.zeros(H * r, N_d, device=indices.device)
                batch_probs.scatter_add_(1, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
                batch_probs = (batch_probs / B_total).view(H, r, N_d)
                self.avg_probs.mul_(self.ema_decay).add_(batch_probs, alpha=1 - self.ema_decay)
                max_p = batch_probs.max(dim=-1)[0].mean()
                self.max_prob_ema.mul_(self.ema_decay).add_(max_p, alpha=1 - self.ema_decay)

                ste_gap = (q_quant - q_soft).detach().abs().mean()
                self.ema_ste_gap.mul_(self.ema_decay).add_(ste_gap, alpha=1 - self.ema_decay)

                h_entropy = -(batch_probs * torch.log(batch_probs + 1e-10)).sum(dim=-1)
                h_ppl = torch.exp(h_entropy).mean(dim=-1)
                self.ema_head_ppl_std.mul_(self.ema_decay).add_(h_ppl.std(), alpha=1 - self.ema_decay)

        return v_q.reshape(B_sz, N_c, H, r), indices, q_soft
