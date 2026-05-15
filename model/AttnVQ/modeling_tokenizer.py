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

        if time_idx is not None:
            t = time_idx.clamp(0, self.pos_emb.shape[1] - 1)  # [B, N]
            temp_emb = self.pos_emb[0][t]                      # [B, N, D]
            z = z + temp_emb.unsqueeze(1).expand(B, C, N, -1).reshape(B * C, N, -1)
        else:
            z = z + self.pos_emb[:, :N, :]

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
# 3. VQ Decoder Head & Manifold Pooling
# ==========================================

class AttnVQ(nn.Module):
    def __init__(self, num_heads, vq_head_vocab_size, e_dim, num_discrete=5, sigmoid_gain=1.0):
        super().__init__()
        self.r, self.num_heads, self.e_dim, self.num_discrete = vq_head_vocab_size, num_heads, e_dim, num_discrete
        self.sigmoid_gain = sigmoid_gain
        self.norm = nn.LayerNorm(e_dim)
        self.A = nn.Parameter(torch.empty(e_dim, num_heads * self.r))
        nn.init.orthogonal_(self.A, gain=1.0)

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

        return v_q.reshape(B_sz, N_c, H, r), indices, q_soft

    @torch.no_grad()
    def get_current_metrics(self):
        metrics = {}
        H, r, D = self.num_heads, self.r, self.e_dim

        p = self.avg_probs
        entropy = -torch.sum(p * torch.log(p + 1e-10), dim=-1)
        metrics['codebook_perplexity'] = torch.exp(entropy).mean().item()
        metrics['codebook_sharpness'] = self.max_prob_ema.item()
        metrics['codebook_active_rank'] = (p.max(dim=-1)[0] < 0.99).float().mean().item()

        s_a = torch.linalg.svdvals(self.A)
        metrics['codebook_avg_singular_value'] = s_a.mean().item()
        metrics['codebook_condition_number'] = (s_a.max() / (s_a.min() + 1e-8)).item()

        return metrics


class LaplacianManifoldPooling(nn.Module):
    """
    Pools multi-head VQ representations using a Low-Pass Graph Filter (Lazy Random Walk).
    Edges between heads are computed with an RBF kernel on decoder weight similarity:
      W[h,m] = exp(-(1 - cosine_sim(W_h, W_m)) / sigma^2)
    where W_h is the flattened decoder weight for head h.
    Using decoder.W makes edges data-independent (stable cluster membership across inputs)
    and lets gradient flow back to the decoder through G.

    G = self_weight * I + other_weight * W_norm
      self_weight  : how much each head relies on its own representation (→1 = fully independent)
      other_weight : how much similar heads amplify each other       (→1 = pure random walk)
      sigma        : RBF bandwidth; large → dense graph, small → sparse/clustered
    """
    def __init__(self, num_heads, gamma=2.0, ema_decay=0.99,
                 self_weight=0.5, other_weight=0.5, init_sigma=1.0):
        super().__init__()
        self.num_heads    = num_heads
        self.gamma        = gamma
        self.ema_decay    = ema_decay
        self.self_weight  = self_weight
        self.other_weight = other_weight
        self.log_sigma    = nn.Parameter(torch.tensor(math.log(init_sigma)))

        self.register_buffer('ema_spectral_gap',    torch.tensor(0.0))
        self.register_buffer('ema_filter_eff_rank', torch.tensor(float(num_heads) / 2.0))
        self.register_buffer('ema_filter_contrast', torch.tensor(1.0))
        self.register_buffer('ema_graph_edge_std',  torch.tensor(0.0))
        self.register_buffer('ema_vq_head_sim',     torch.tensor(0.0))
        self.register_buffer('ema_n_clusters',      torch.tensor(1.0))

    def _update_ema(self, name, value):
        buf = getattr(self, f'ema_{name}')
        buf.mul_(self.ema_decay).add_(value, alpha=1.0 - self.ema_decay)

    def forward(self, v_q, B, C, decoder_W):
        # v_q: [B*C, N, H, r],  decoder_W: [H, D, 2F]
        B_C, N, H, r = v_q.shape

        # 1. Build Head Affinity Graph via RBF on decoder weight similarity
        #    Data-independent: same graph for every sample → stable cluster membership
        #    For unit vectors: ||w_h - w_m||^2 = 2(1 - cosine_sim)
        #    RBF(w_h, w_m) = exp(-(1 - sim) / sigma^2)
        w_flat = F.normalize(decoder_W.reshape(H, -1), dim=-1)         # [H, D*2F]
        sim    = torch.einsum('hd,md->hm', w_flat, w_flat)             # [H, H]
        sim    = sim.unsqueeze(0).expand(B, -1, -1)                    # [B, H, H]
        sigma  = self.log_sigma.exp()
        W      = torch.exp(-(1.0 - sim) / (sigma ** 2 + 1e-8))        # [B, H, H], symmetric
        # Zero diagonal: self-connection already handled by self_weight * I in G
        W      = W * (1.0 - torch.eye(H, device=v_q.device).unsqueeze(0))

        # 3. Normalized Graph Laplacian
        D_deg = W.sum(dim=-1)
        D_inv_sqrt = 1.0 / (D_deg.sqrt() + 1e-8)
        W_norm = W * D_inv_sqrt.unsqueeze(-1) * D_inv_sqrt.unsqueeze(-2)
        I = torch.eye(H, device=v_q.device).unsqueeze(0)  # [1, H, H]

        # 4. Graph Diffusion Filter (used inside the decoder, not applied to v_q here)
        G = self.self_weight * I + self.other_weight * W_norm  # [B, H, H]

        # 5. Track v_q diversity across heads (every step — collapse monitor)
        with torch.no_grad():
            v_mean = v_q.reshape(B, C, N, H, r).mean(dim=(1, 2))          # [B, H, r]
            v_norm = F.normalize(v_mean, dim=-1)
            v_sim  = torch.einsum('bhr,bmr->bhm', v_norm, v_norm)         # [B, H, H]
            off    = 1.0 - torch.eye(H, device=v_q.device)
            self._update_ema('vq_head_sim', (v_sim * off).sum() / (off.sum() * B))

        # 6. Head importance — training uses cheap diagonal, eval uses full Laplacian + stats
        with torch.no_grad():
            head_importance = G.diagonal(dim1=-2, dim2=-1).clone()  # [B, H] fast fallback
            if not self.training:
                # Full Laplacian Survival Score + health metrics (validation only)
                try:
                    L = I - W_norm
                    lambda_vals, U = torch.linalg.eigh(L + 1e-3 * I)
                    spectral_weights = torch.exp(-self.gamma * lambda_vals)
                    spectral_weights = spectral_weights / (spectral_weights.sum(dim=-1, keepdim=True) + 1e-8)
                    head_importance = torch.einsum('bhk,bk,bhk->bh', U, spectral_weights, U)
                except RuntimeError:
                    pass
                gap        = (lambda_vals[:, 1] - lambda_vals[:, 0]).mean()
                sw         = spectral_weights.clamp(min=1e-10)
                eff_rank   = torch.exp(-torch.sum(sw * sw.log(), dim=-1)).mean()
                contrast   = (spectral_weights.max(dim=-1)[0] / (spectral_weights.min(dim=-1)[0] + 1e-8)).mean()
                off_mask   = (1.0 - torch.eye(H, device=W.device)).unsqueeze(0)
                edge_std   = (W * off_mask).std(dim=(-1, -2)).mean()
                # Number of clusters = index of largest consecutive eigenvalue gap + 1
                eig_gaps   = lambda_vals[:, 1:] - lambda_vals[:, :-1]   # [B, H-1]
                n_clusters = (eig_gaps.argmax(dim=-1) + 1).float().mean()
                self._update_ema('spectral_gap',    gap)
                self._update_ema('filter_eff_rank', eff_rank)
                self._update_ema('filter_contrast', contrast)
                self._update_ema('graph_edge_std',  edge_std)
                self._update_ema('n_clusters',      n_clusters)

        return v_q, head_importance, sim.detach(), G

    @torch.no_grad()
    def get_current_metrics(self):
        return {
            'pool_spectral_gap':    self.ema_spectral_gap.item(),
            'pool_filter_eff_rank': self.ema_filter_eff_rank.item(),
            'pool_filter_contrast': self.ema_filter_contrast.item(),
            'pool_graph_edge_std':  self.ema_graph_edge_std.item(),
            'pool_rbf_sigma':       self.log_sigma.exp().item(),
            'pool_vq_head_sim':     self.ema_vq_head_sim.item(),
            'pool_n_clusters':      self.ema_n_clusters.item(),
        }


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

    def forward(self, v_q, A, G=None, B=None, C=None):
        """
        v_q: [B*C, N, H, r]
        A:   [D, H*r] parameter from AttnVQ
        G:   [B, H, H] diffusion filter — applied as column sums to h_out.
             Σ_h Σ_m G[h,m]*h_out[m] = Σ_m col_sum[m]*h_out[m], so we fuse
             into v_q scaling before the einsum (same result, no H×H intermediate).
        """
        B_C, N, H, r = v_q.shape
        D = A.shape[0]
        M = torch.einsum('dhr,hdf->hrf', A.view(D, H, r), self.W)  # [H, r, 2*F]

        if G is not None:
            g_w = G.sum(dim=1)                                        # [B, H] col sums
            g_w = g_w.unsqueeze(1).expand(B, C, H).reshape(B_C, H)   # [B*C, H]
            v_q = v_q * g_w.unsqueeze(1).unsqueeze(-1)                # [B*C, N, H, r]

        out = torch.einsum('bnhr,hrf->bnf', v_q, M) / math.sqrt(H)

        out = out + self.bias
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
            decoder_heads_config =[{
                "stage_idx": enc_depth - 1,
                "freq_range": [0.0, fs / 2.0],
                "vq_head_num": 128,          # Scaled up for your 128-head usecase!
                "vq_head_vocab_size": 64,
                "vq_num_discrete": 5,
                "pooling_gamma": 2.0
            }]

        self.decoder_heads_config = decoder_heads_config
        
        self.vq_heads = nn.ModuleList()
        self.poolers = nn.ModuleList()      
        self.decoders = nn.ModuleList()
        
        full_freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / fs)

        for i, h_cfg in enumerate(decoder_heads_config):
            f_min, f_max = h_cfg["freq_range"]
            mask = (full_freqs >= f_min - 1e-5) & (full_freqs <= f_max + 1e-5)
            self.register_buffer(f'freq_mask_{i}', mask)
            fft_dim = int(mask.sum().item())
            
            num_h = h_cfg.get("vq_head_num", 8)
            r_dim = h_cfg.get("vq_head_vocab_size", 64)
            
            self.vq_heads.append(AttnVQ(
                num_heads=num_h,
                vq_head_vocab_size=r_dim,
                e_dim=embed_dim,
                num_discrete=h_cfg.get("vq_num_discrete", 5),
            ))
            
            # Initialize the Spectral Pooler
            self.poolers.append(LaplacianManifoldPooling(
                num_heads=num_h,
                gamma=h_cfg.get("pooling_gamma", 2.0),
                self_weight=h_cfg.get("pooling_self_weight",  0.5),
                other_weight=h_cfg.get("pooling_other_weight", 0.5),
                init_sigma=h_cfg.get("pooling_sigma", 1.0),
            ))
            
            self.decoders.append(FastAdditiveDecoder(
                embed_dim=embed_dim,
                num_heads=num_h,
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

        all_pred_real, all_pred_imag = [], []
        all_indices, all_weights = [], []
        all_head_importances = []
        all_sim, all_G = [], []

        for i, h_cfg in enumerate(self.decoder_heads_config):
            z_stage = stage_outputs[h_cfg["stage_idx"]]
            B_s, C_s, N_s, D_s = z_stage.shape
            z_flat = z_stage.reshape(B_s * C_s, N_s, D_s)

            v_q, indices, weights = self.vq_heads[i](z_flat)

            _, head_importance, sim, G = self.poolers[i](v_q, B, C, decoder_W=self.decoders[i].W)

            p_real, p_imag = self.decoders[i](v_q, self.vq_heads[i].A, G=G, B=B, C=C)

            all_pred_real.append(p_real.reshape(B, C, -1))
            all_pred_imag.append(p_imag.reshape(B, C, -1))
            all_indices.append(indices)
            all_weights.append(weights)
            all_head_importances.append(head_importance)    # [B, H]
            all_sim.append(sim)                             # [B, H, H]
            all_G.append(G)                                 # [B, H, H]

        return all_pred_real, all_pred_imag, all_indices, all_weights, all_head_importances, all_sim, all_G

    @torch.no_grad()
    def get_current_metrics(self):
        # Merge VQ metrics
        head_metrics =[vq.get_current_metrics() for vq in self.vq_heads]
        full_metrics = {}
        if head_metrics:
            for k in head_metrics[0]:
                full_metrics[k] = sum(m[k] for m in head_metrics) / len(head_metrics)
            for i, m in enumerate(head_metrics):
                for k, v in m.items():
                    full_metrics[f"{k}_head_{i}"] = v
                    
        # Merge Pooler EMA metrics
        pooler_metrics = [pooler.get_current_metrics() for pooler in self.poolers]
        if pooler_metrics:
            for k in pooler_metrics[0]:
                full_metrics[k] = sum(m[k] for m in pooler_metrics) / len(pooler_metrics)
            for i, m in enumerate(pooler_metrics):
                for k, v in m.items():
                    full_metrics[f"{k}_head_{i}"] = v

        return full_metrics

    def get_loss(self, x, p_reals, p_imags, x_fft=None):
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

        l_total = (l_expert_real + l_expert_imag) / num_heads + l_mse_global
        return l_total, l_expert_real / num_heads, l_expert_imag / num_heads, l_mse_global

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