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


class FFNRouter(nn.Module):
    """
    Lightweight per-token router for MoEFFN's routed Experts — distinct from FilterRouter
    (dim, dot-product weight against a pooled per-Filter View): this one scores every raw
    token directly via a plain nn.Linear gate, since MoEFFN routes at (B, C, N) token
    granularity rather than over a small fixed pool of pre-pooled Views.

    Same top-k softmax gating + Switch-Transformer-style load-balance loss formula as
    FilterRouter (see docs/adr/0008-moe-ffn-for-mesae.md), applied to a much larger token
    count instead of a handful of Filters.
    """
    def __init__(self, dim, n_routed, top_k):
        super().__init__()
        self.n_routed = n_routed
        self.top_k = min(top_k, n_routed)
        self.gate = nn.Linear(dim, n_routed, bias=False)

    def forward(self, x):
        """x: [T, D] -> gate_mask [T, n_routed], lb_loss scalar"""
        gate_logits = self.gate(x)  # [T, R]
        topk_val, topk_idx = gate_logits.topk(self.top_k, dim=-1)
        topk_weight = torch.softmax(topk_val, dim=-1).to(gate_logits.dtype)
        gate_mask = torch.zeros_like(gate_logits).scatter_(-1, topk_idx, topk_weight)

        f = (gate_mask.detach() > 0).float().mean(dim=0)     # [R] hard selection frequency
        p = torch.softmax(gate_logits, dim=-1).mean(dim=0)   # [R] dense prob, has grad
        lb_loss = self.n_routed * ((f / (f.sum() + 1e-8)) * (p / (p.sum() + 1e-8))).sum()
        return gate_mask, lb_loss


class MoEFFN(nn.Module):
    """
    DeepSeekMoE-style FFN: n_routed Experts (top-k gated per token, competing for a fixed
    per-token budget) + n_shared Experts (always active on every token, summed at full
    weight — unlike ExpertChannelPool's 0.2x-weighted shared Filters, true DeepSeekMoE
    shared Experts aren't down-weighted). Replaces the single dense FFN sub-layer in
    TSABlock. See docs/adr/0008-moe-ffn-for-mesae.md.

    expert_hidden defaults to a fraction of the original dense FFN's hidden_dim so total
    *active* per-token compute (n_shared + top_k experts firing) stays roughly at parity
    with the old single dense FFN — standard DeepSeekMoE fine-grained-expert sizing.

    # ponytail: dense routed-expert compute (every routed Expert runs on every token, then
    # masked by the gate — same "compute all, mask by gate" convention FilterRouter/
    # ExpertChannelPool already use in this file), not real sparse dispatch. Fine at this
    # expert count; switch to grouped/sparse dispatch if expert count or throughput ever
    # makes this the bottleneck.
    """
    def __init__(self, dim, hidden_dim, n_routed, n_shared, top_k, expert_hidden=None, dropout=0.0):
        super().__init__()
        self.n_routed = n_routed
        self.n_shared = n_shared
        expert_hidden = expert_hidden or max(8, hidden_dim // (n_shared + top_k))

        self.routed_experts = nn.ModuleList([
            FFN(dim, expert_hidden, dropout=dropout) for _ in range(n_routed)
        ])
        self.shared_experts = nn.ModuleList([
            FFN(dim, expert_hidden, dropout=dropout) for _ in range(n_shared)
        ])
        self.router = FFNRouter(dim, n_routed, top_k)

    def forward(self, x):
        """x: [B*C, N, D] -> out [B*C, N, D], lb_loss scalar"""
        BC, N, D = x.shape
        x_flat = x.reshape(BC * N, D)

        gate_mask, lb_loss = self.router(x_flat)  # [T, R]
        routed_out = torch.stack([e(x_flat) for e in self.routed_experts], dim=1)  # [T, R, D]
        routed_sum = (routed_out * gate_mask.unsqueeze(-1)).sum(dim=1)  # [T, D]

        shared_sum = x_flat.new_zeros(x_flat.shape)
        for e in self.shared_experts:
            shared_sum = shared_sum + e(x_flat)

        out = (routed_sum + shared_sum).reshape(BC, N, D)
        return out, lb_loss


class TSABlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., dropout=0.0,
                 n_routed_ffn_experts=4, n_shared_ffn_experts=1, ffn_top_k=2, ffn_expert_hidden=None):
        super().__init__()
        self.norm_time = nn.LayerNorm(dim)
        self.temporal_attn = ConvolutionalAdditiveAttention(dim, kernel_size=3)
        self.drop_t = nn.Dropout(dropout)

        self.norm_space = nn.LayerNorm(dim)
        # dropout here is on the attention WEIGHTS themselves (nn.MultiheadAttention's own
        # `dropout` arg), on top of drop_s below which drops the branch's output — two
        # different regularization points, same shared `dropout` value.
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_s = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = MoEFFN(dim, hidden_dim=int(dim * mlp_ratio), n_routed=n_routed_ffn_experts,
                           n_shared=n_shared_ffn_experts, top_k=ffn_top_k,
                           expert_hidden=ffn_expert_hidden, dropout=dropout)

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

        ffn_out, ffn_lb_loss = self.ffn(self.norm_ffn(x_flat))
        x_flat = x_flat + self.scale_ffn * ffn_out
        x_flat = self.norm_out(x_flat)
        return x_flat.view(B, C, N, D), ffn_lb_loss


class TSAEncoder(nn.Module):
    def __init__(self, dim, depth=12, num_heads=8, mlp_ratio=4., dropout=0.0,
                 pool_after_blocks=(),
                 n_routed_ffn_experts=4, n_shared_ffn_experts=1, ffn_top_k=2, ffn_expert_hidden=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            TSABlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                     n_routed_ffn_experts=n_routed_ffn_experts, n_shared_ffn_experts=n_shared_ffn_experts,
                     ffn_top_k=ffn_top_k, ffn_expert_hidden=ffn_expert_hidden)
            for _ in range(depth)
        ])
        # UNet-style temporal down/up: triangular-kernel-filtered pool N in half after each
        # listed block, then (once, after the last block) nearest-repeat upsample + gated
        # skip-add back through the same points in reverse, restoring the original N.
        # Parameter-free pooling (fixed [1,2,1]/4 kernel + repeat), so downstream
        # (SAE/decoder/loss) never sees a shape change. The gated residual add on the way
        # back up is always on now (was an optional `upsample_residual_add` flag; validated
        # on, simplified to permanent).
        self.pool_after_blocks = set(pool_after_blocks)
        # per-skip learned gate on the residual add, sigmoid init ~0.95 (near plain add);
        # ordered ascending by block index to match `skips` build order in forward()
        self.skip_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(3.0)) for _ in sorted(self.pool_after_blocks)
        ])

    def enable_spatial(self):
        for block in self.blocks:
            block.enable_spatial()

    def enable_temporal(self):
        for block in self.blocks:
            block.enable_temporal()

    @staticmethod
    def _pool(x):
        """Downsample N by 2 through a fixed triangular ([1,2,1]/4) lowpass before
        decimating, not a bare pair-mean (box filter): a 2-tap box filter's frequency
        response has stopband sidelobes near its cutoff, so patch-to-patch content above
        the new Nyquist rate isn't fully attenuated before every-other-sample is dropped —
        it folds back in as aliasing, indistinguishable from genuine low-frequency content
        to every block deeper than this pool point. The 3-tap triangular kernel attenuates
        harder near cutoff, same as the standard Burt-Adelson pyramid REDUCE filter. Output
        sample i is centered on original sample 2i (taps 2i-1, 2i, 2i+1); the left edge
        (i=0, needing sample -1) is handled by replicating x[0], no pad needed on the right
        since the last output only ever reads up to index N-1."""
        B, C, N, D = x.shape
        if N % 2 == 1:
            x = torch.cat([x, x[:, :, -1:, :]], dim=2)  # repeat last token to make N even
        N = x.shape[2]
        xp = torch.cat([x[:, :, :1, :], x], dim=2)  # replicate-pad one sample on the left
        left   = xp[:, :, 0:N:2, :]
        center = xp[:, :, 1:N + 1:2, :]
        right  = xp[:, :, 2:N + 2:2, :]
        return (left + 2 * center + right) / 4.0

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
        ffn_lb_loss = x.new_zeros(())
        for i, block in enumerate(self.blocks):
            x_in = x
            x, blk_ffn_lb = block(x)
            ffn_lb_loss = ffn_lb_loss + blk_ffn_lb
            if record_norms:
                with torch.no_grad():
                    self.last_block_norms.append((x - x_in).norm(dim=-1).mean().item())
            if i in self.pool_after_blocks:
                skips.append(x)
                x = self._pool(x)

        for skip, gate in zip(reversed(skips), reversed(self.skip_gates)):
            N_pre = skip.shape[2]
            x = x.repeat_interleave(2, dim=2)  # upsample
            x = x[:, :, :N_pre, :]              # trim off any pool-time padding
            x = x + torch.sigmoid(gate) * skip
        return x, ffn_lb_loss


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
    Per-Expert STATIC spatial pooling — each Expert learns one fixed unmixing vector over
    the C canonical channels (channel index is a stable physical identity across datasets,
    see `canonical_channels` in config/config.json — unified once, other datasets zero-pad
    onto it), applied identically to every patch and trial.

    Replaces the earlier content-based attention pool (query/key over the actual
    per-channel vectors), which let an Expert's channel weighting drift patch to patch —
    fine for local reconstruction, but the opposite of an ICA/CSP-style source
    decomposition, which wants one stable spatial pattern per component rather than a
    moving one.

    `spatial_logit` is a plain [n_experts, n_channels] parameter table, softmaxed per
    sample over valid channels only, so per-dataset zero-padding still renormalizes
    correctly.

    `temperature` (default 1.0, no-op) divides the logits before softmax — raising it
    flattens an Expert's spatial pattern across channels, countering collapse onto a
    single dominant channel when broader spatial coverage is wanted.
    """
    def __init__(self, n_experts, n_channels, temperature=1.0):
        super().__init__()
        self.n_experts = n_experts
        self.n_channels = n_channels
        self.temperature = temperature
        self.spatial_logit = nn.Parameter(torch.zeros(n_experts, n_channels))
        nn.init.normal_(self.spatial_logit, std=0.02)

    def forward(self, z, valid_mask=None):
        """
        z: [M, C, D] per-channel embeddings for M=B*N patches
        valid_mask: [M, C] bool, True = real (not zero-padded) channel (optional)
        -> pooled [M, n_experts, D], attn [M, n_experts, C]
        """
        M = z.shape[0]
        scores = self.spatial_logit.unsqueeze(0).expand(M, -1, -1) / self.temperature  # [M, H, C]
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask[:, None, :], float('-inf'))
        attn = torch.softmax(scores, dim=-1)  # [M, H, C]
        pooled = torch.einsum('mhc,mcd->mhd', attn, z)  # [M, H, D]
        return pooled, attn

    def decorrelation_loss(self):
        """Mean squared pairwise cosine similarity between Experts' softmaxed spatial
        patterns (off-diagonal only) — cheap proxy for ICA's independence constraint,
        pushes Experts toward distinct channel weightings instead of redundant copies."""
        if self.n_experts < 2:
            return self.spatial_logit.new_zeros(())
        pattern = torch.softmax(self.spatial_logit, dim=-1)  # [H, C]
        pattern = F.normalize(pattern, dim=-1)
        sim = pattern @ pattern.t()  # [H, H]
        off_diag = ~torch.eye(self.n_experts, dtype=torch.bool, device=sim.device)
        return sim[off_diag].pow(2).mean()


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
