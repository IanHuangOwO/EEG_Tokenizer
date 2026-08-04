import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        """ x: [B*C, N, D] """
        return self.fc2(self.drop(self.act(self.fc1(x))))


class TSABlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., dropout=0.0):
        super().__init__()
        self.norm_time = nn.LayerNorm(dim)
        self.temporal_attn = ConvolutionalAdditiveAttention(dim, kernel_size=3)
        self.drop_t = nn.Dropout(dropout)

        self.norm_space = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.drop_s = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FFN(dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

        # norm_time/norm_space/norm_ffn are pre-norm (normalize the input to each
        # sub-layer) — nothing caps the residual stream x itself after three unbounded
        # adds, so later blocks operating on an already-inflated x can blow up further
        # (observed: block_norm_10/11 growing ~13x over 17 Pretrain epochs while a frozen
        # SAE dictionary downstream can't adapt to the drifting scale, see dead_feature_rate
        # climb). This final norm caps x's own magnitude every block.
        self.norm_out = nn.LayerNorm(dim)

        # LayerScale (Touvron et al., CaiT) on all three residual branches: a learnable
        # per-channel multiplier, init small, applied to each branch's output before the
        # residual add. Unlike the zero-init below (a one-time starting condition on two of
        # the three branches only, nothing stopping unbounded growth afterward), this stays
        # active for the whole run and throttles each branch's net contribution to x
        # throughout training — the actual mechanism behind the compounding-depth blowup
        # (block_norm growing block-to-block, epoch-to-epoch, eventually NaN) is unbounded
        # per-branch growth stacked across 12 blocks x many epochs, and this is what caps
        # that growth at its source instead of only re-normalizing x after the fact.
        layerscale_init = 1e-4
        self.scale_t   = nn.Parameter(torch.full((dim,), layerscale_init))
        self.scale_s   = nn.Parameter(torch.full((dim,), layerscale_init))
        self.scale_ffn = nn.Parameter(torch.full((dim,), layerscale_init))

        # Both cross-patch (temporal, global context pooled over all N) and cross-channel
        # (spatial) mixing default off — MeSAE's tokenizer stage trains the SAE on
        # patch-local features only, so the frozen dictionary can't leak already-seen
        # context into masked-stage reconstruction targets (see
        # docs/adr/0003-mesae-two-stage-masked-training.md). Both out_proj-equivalents are
        # zero-inited so enabling later starts as a no-op and grows in under gradient,
        # instead of shocking a checkpoint that never saw either term active.
        self.temporal_active = False
        self.spatial_active = False
        nn.init.zeros_(self.temporal_attn.proj.weight)
        nn.init.zeros_(self.temporal_attn.proj.bias)
        nn.init.zeros_(self.spatial_attn.out_proj.weight)
        nn.init.zeros_(self.spatial_attn.out_proj.bias)

    def enable_temporal(self):
        self.temporal_active = True

    def enable_spatial(self):
        self.spatial_active = True

    def forward(self, x):
        B, C, N, D = x.shape
        x_flat = x.view(B * C, N, D)

        if self.temporal_active:
            x_norm_t = self.norm_time(x_flat)
            attn_out_t = self.temporal_attn(x_norm_t)
            x_flat = x_flat + self.drop_t(self.scale_t * attn_out_t)

        x_space = x_flat.view(B, C, N, D).permute(0, 2, 1, 3).reshape(B * N, C, D)
        if self.spatial_active:
            x_norm = self.norm_space(x_space)
            attn_out, _ = self.spatial_attn(x_norm, x_norm, x_norm)
            x_space = x_space + self.drop_s(self.scale_s * attn_out)
        x_flat = x_space.view(B, N, C, D).permute(0, 2, 1, 3).reshape(B * C, N, D)

        x_flat = x_flat + self.scale_ffn * self.ffn(self.norm_ffn(x_flat))
        x_flat = self.norm_out(x_flat)
        return x_flat.view(B, C, N, D)


class TSAEncoder(nn.Module):
    def __init__(self, dim, depth=12, num_heads=8, mlp_ratio=4., dropout=0.0,
                 pool_after_blocks=(), upsample_residual_add=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            TSABlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        # UNet-style temporal down/up: avg-pool N in half after each listed block, then
        # (once, after the last block) nearest-repeat upsample + skip-add back through the
        # same points in reverse, restoring the original N. Parameter-free (mean + repeat),
        # so downstream (VQ/decoder/loss) never sees a shape change.
        self.pool_after_blocks = set(pool_after_blocks)
        self.upsample_residual_add = upsample_residual_add
        # per-skip learned gate on the residual add, sigmoid init ~0.95 (near plain add);
        # ordered ascending by block index to match `skips` build order in forward()
        self.skip_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(3.0)) for _ in sorted(self.pool_after_blocks)
        ]) if upsample_residual_add else None

    def enable_spatial(self):
        for block in self.blocks:
            block.enable_spatial()

    def enable_temporal(self):
        for block in self.blocks:
            block.enable_temporal()

    @staticmethod
    def _pool(x):
        B, C, N, D = x.shape
        if N % 2 == 1:
            x = torch.cat([x, x[:, :, -1:, :]], dim=2)  # repeat last token to make N even
        return x.reshape(B, C, x.shape[2] // 2, 2, D).mean(dim=3)

    def forward(self, x):
        skips = []  # unpadded pre-pool tensors, one per pool point, in block order
        # Per-block contribution norm — how much each block actually changes its input,
        # not just the skip-gate residual-add strength (which conflates "shallow skip
        # re-injected on top" with "deep processing did nothing"; this measures the
        # deep processing directly). Eval-only (no_grad, .item() sync) — same convention
        # as the other diagnostics in this codebase (fingerprint stats, codebook health).
        record_norms = not self.training
        if record_norms:
            self.last_block_norms = []
        for i, block in enumerate(self.blocks):
            x_in = x
            x = block(x)
            if record_norms:
                with torch.no_grad():
                    self.last_block_norms.append((x - x_in).norm(dim=-1).mean().item())
            if i in self.pool_after_blocks:
                skips.append(x)
                x = self._pool(x)

        gates = reversed(self.skip_gates) if self.upsample_residual_add else [None] * len(skips)
        for skip, gate in zip(reversed(skips), gates):
            N_pre = skip.shape[2]
            x = x.repeat_interleave(2, dim=2)  # upsample
            x = x[:, :, :N_pre, :]              # trim off any pool-time padding
            if self.upsample_residual_add:
                x = x + torch.sigmoid(gate) * skip
        return x


# ==========================================
# Decoder & channel pooling
# ==========================================

class MultiHeadDecoder(nn.Module):
    """
    Per-head decoder: each VQ head gets its own independent 2-layer MLP mapping its
    embed_dim vector to a patch_len reconstruction. Heads share no weights — replaces a
    single shared nn.Linear so that "sum after decode" is a genuinely different (and
    correct) operation from "sum before decode": each head is now a real nonlinear
    function of its own input, not just a linear slice of one shared matrix.

    Vectorized across heads via batched matmul (einsum), not a Python loop over H modules.

    Nonlinearity (activation) sits on the HIDDEN layer only. The final projection to
    patch_len is a plain linear map — it's compared directly against raw EEG amplitude via
    MSE, so squashing it with activation/norm would clip or zero real signal ranges before
    the loss ever sees them.
    """
    def __init__(self, num_heads, embed_dim, patch_len, hidden=None, activation=nn.GELU):
        super().__init__()
        hidden = hidden or embed_dim

        self.w1 = nn.Parameter(torch.empty(num_heads, embed_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(num_heads, hidden))
        self.act  = activation()
        self.w2 = nn.Parameter(torch.empty(num_heads, hidden, patch_len))
        self.b2 = nn.Parameter(torch.zeros(num_heads, patch_len))

        for h in range(num_heads):
            nn.init.kaiming_uniform_(self.w1[h], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w2[h], a=math.sqrt(5))

    def forward(self, z_per_head):
        """z_per_head: [..., H, embed_dim] -> recon_per_head [..., H, patch_len]"""
        h = torch.einsum('...hd,hdk->...hk', z_per_head, self.w1) + self.b1   # [..., H, hidden]
        h = self.act(h)
        return torch.einsum('...hk,hkp->...hp', h, self.w2) + self.b2         # [..., H, patch_len]


class ExpertChannelPool(nn.Module):
    """
    Per-Expert content-based channel-attention pooling — replaces the concatenated Token
    (see CONTEXT.md: Token, retired / Expert View). Each Expert gets its own learnable
    query attending over the C per-channel embeddings (keys), so different Experts can
    weight channels differently for the same patch, instead of every Expert reading an
    identical C*D concatenation.

    Content-based (keys are the actual per-channel vectors, not a fixed per-channel-index
    weight) so channel count/order can vary across datasets without corrupting the pooled
    result — a static per-position weight would just reintroduce the same fixed-slot
    problem concatenation had. Invalid/zero-padded channels (`valid_mask`) are masked to
    -inf before softmax so they get exactly zero attention weight.

    `temperature` (default 1.0, no-op) divides the scaled scores before softmax — raising
    it flattens attention across channels, countering a filter collapsing onto a very
    narrow local channel cluster when broader coverage is wanted. Separate from `scale`:
    `scale` is fixed variance-stabilization (standard attention practice, scales with
    `hidden`), `temperature` is a free knob purely for controlling peakiness/entropy,
    tunable independent of `hidden`.
    """
    def __init__(self, dim, n_experts, hidden=32, temperature=1.0):
        super().__init__()
        self.n_experts = n_experts
        self.scale = hidden ** -0.5
        self.temperature = temperature
        self.key_w = nn.Parameter(torch.empty(n_experts, dim, hidden))
        self.key_b = nn.Parameter(torch.zeros(n_experts, hidden))
        self.query = nn.Parameter(torch.empty(n_experts, hidden))
        for h in range(n_experts):
            nn.init.kaiming_uniform_(self.key_w[h], a=math.sqrt(5))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, z, valid_mask=None):
        """
        z: [M, C, D] per-channel embeddings for M=B*N patches
        valid_mask: [M, C] bool, True = real (not zero-padded) channel (optional)
        -> pooled [M, n_experts, D], attn [M, n_experts, C]
        """
        key = torch.einsum('mcd,hdk->mhck', z, self.key_w) + self.key_b[:, None, :]  # [M, H, C, hidden]
        scores = torch.einsum('mhck,hk->mhc', key, self.query) * self.scale / self.temperature  # [M, H, C]
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask[:, None, :], float('-inf'))
        attn = torch.softmax(scores, dim=-1)  # [M, H, C]
        pooled = torch.einsum('mhc,mcd->mhd', attn, z)  # [M, H, D]
        return pooled, attn


class FilterRouter(nn.Module):
    """
    Top-k softmax router over the routed Filter subset — logic ported unchanged from
    model/MeFSQ/MeFSQ_modules.py's Router (duplicated, not cross-imported, same convention
    PerChannelHeadAttn already uses in this file). See
    docs/adr/0007-routed-filter-gating-for-mesae.md: MeSAE's dense
    `recon_per_filter.sum(dim=1)` gives Filters no competitive pressure to specialize, and
    a single shared TopKSAE dictionary reused across all Filters means two Filters whose
    pooled views land in similar regions draw on the same dictionary atoms. This router
    gates *which routed Filters' decoded output survives into the sum*, so specialization
    has to earn its way into a fixed per-patch budget instead of being always-on for free.

    Each routed Filter is scored against its own pooled view (see ExpertChannelPool)
    rather than a shared input, so routing stays interpretable. Top-k scores are
    softmax-normalized (weights sum to 1, regardless of k) and scattered into a dense
    [M, n_routed_filters] gate mask (nonzero only at the selected Filters). Also returns
    the Switch-Transformer-style load-balance loss (dense softmax prob, has grad, times
    hard selection frequency, detached — 1.0 at perfectly uniform routing).
    """
    def __init__(self, dim, n_routed_filters, n_top_k):
        super().__init__()
        self.n_routed_filters = n_routed_filters
        self.n_top_k = min(n_top_k, n_routed_filters)
        self.scale = dim ** -0.5
        self.weight = nn.Parameter(torch.empty(n_routed_filters, dim))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, filter_views):
        """filter_views: [M, n_routed_filters, D] -> gate_mask [M, n_routed_filters], lb_loss scalar"""
        gate_logits = (filter_views * self.weight).sum(dim=-1) * self.scale  # [M, R]
        topk_val, topk_idx = gate_logits.topk(self.n_top_k, dim=-1)
        topk_weight = torch.softmax(topk_val, dim=-1).to(gate_logits.dtype)
        gate_mask = torch.zeros_like(gate_logits).scatter_(-1, topk_idx, topk_weight)

        f = (gate_mask.detach() > 0).float().mean(dim=0)     # [R] hard selection frequency
        p = torch.softmax(gate_logits, dim=-1).mean(dim=0)   # [R] dense prob, has grad
        lb_loss = self.n_routed_filters * ((f / (f.sum() + 1e-8)) * (p / (p.sum() + 1e-8))).sum()
        return gate_mask, lb_loss


class PerChannelHeadAttn(nn.Module):
    """
    Three-stage learnable-query attention pooling — identical to
    model/MeFSQ/MeFSQ_modules.py's PerChannelHeadAttn (duplicated here rather than
    cross-imported, same convention as ExpertChannelPool/MultiHeadDecoder above: each
    model package stays self-contained). Fully backbone-agnostic — only needs
    z_per_head [B, C, N, H, d] at forward time, so it works unchanged whether H indexes
    MeFSQ Experts or MeSAE Filters.

    Stage 1 (temporal): per (channel, unit), a learnable query attends over the N patches.
    Stage 2 (unit): per channel, a learnable query attends over the H units — weights sum
    to 1 PER CHANNEL, not jointly, so channels never compete for weight.
    Stage 3 (channel): a learnable query attends over the C channels, pooling to a single
    [B, d] vector fed into cls — avoids the per-channel-concat overparameterization that
    caused val-chance memorization (see docs/agents/ / CONTEXT.md finetune val-chance bug).
    """
    def __init__(self, head_dim, num_channels, num_classes, hidden=32, dropout=0.1):
        super().__init__()
        # Un-normalized-scale input (raw decoder-output EEG amplitude) starves key/query
        # dot products and cls of usable gradient — normalize here so the pooler works
        # regardless of which encode_* the caller feeds it.
        self.input_norm = nn.LayerNorm(head_dim)
        self.key_n   = nn.Linear(head_dim, hidden)
        self.query_n = nn.Parameter(torch.zeros(hidden))
        self.key_h   = nn.Linear(head_dim, hidden)
        self.query_h = nn.Parameter(torch.zeros(hidden))
        self.key_c   = nn.Linear(head_dim, hidden)
        self.query_c = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.query_n, std=0.02)
        nn.init.normal_(self.query_h, std=0.02)
        nn.init.normal_(self.query_c, std=0.02)
        self.scale = hidden ** -0.5
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(head_dim, num_classes)

    def forward(self, z_per_head, pad_mask=None):
        """
        z_per_head: [B, C, N, H, d]
        pad_mask: [B, C, N] bool, True = valid (optional, for zero-padded channels/patches)
        Returns (logits [B, num_classes], attn_h [B, C, H], attn_n [B, C, H, N], attn_c [B, C])
        """
        z_key = self.input_norm(z_per_head)

        # ---- stage 1: attention pool over N (patches), per channel & unit ----
        logits_n = torch.einsum('bcnhk,k->bcnh', self.key_n(z_key), self.query_n) * self.scale  # [B, C, N, H]
        if pad_mask is not None:
            logits_n = logits_n.masked_fill(~pad_mask.unsqueeze(-1), float('-inf'))
        attn_n = torch.softmax(logits_n, dim=2)                                 # softmax over N, per channel & unit
        attn_n = torch.nan_to_num(attn_n)                                       # all-invalid channel -> all -inf -> nan; zero it
        z_h = torch.einsum('bcnhd,bcnh->bchd', z_per_head, attn_n)              # [B, C, H, d]
        z_h_key = torch.einsum('bcnhd,bcnh->bchd', z_key, attn_n)               # [B, C, H, d] — for stage 2 logits only

        valid_channel = pad_mask.any(dim=-1) if pad_mask is not None else None  # [B, C]

        # ---- stage 2: attention pool over H (units), per channel ----
        logits_h = torch.einsum('bchk,k->bch', self.key_h(z_h_key), self.query_h) * self.scale  # [B, C, H]
        attn_h = torch.softmax(logits_h, dim=2)                                 # softmax over H, independently per channel
        pooled_per_channel = (z_h * attn_h.unsqueeze(-1)).sum(dim=2)            # [B, C, d]
        pooled_per_channel_key = (z_h_key * attn_h.unsqueeze(-1)).sum(dim=2)    # [B, C, d] — for stage 3 logits only

        # ---- stage 3: attention pool over C (channels) ----
        logits_c = torch.einsum('bck,k->bc', self.key_c(pooled_per_channel_key), self.query_c) * self.scale  # [B, C]
        if valid_channel is not None:
            logits_c = logits_c.masked_fill(~valid_channel, float('-inf'))
        attn_c = torch.softmax(logits_c, dim=1)                                 # [B, C]
        attn_c = torch.nan_to_num(attn_c)                                       # all-invalid batch item -> nan; zero it
        pooled = (pooled_per_channel * attn_c.unsqueeze(-1)).sum(dim=1)         # [B, d]

        attn_n = attn_n.permute(0, 1, 3, 2)                                     # [B, C, H, N] for interpretability
        return self.cls(self.drop(pooled)), attn_h, attn_n, attn_c
