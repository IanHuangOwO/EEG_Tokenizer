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
        """ x: [B*C, N, D]

        fp32 island: `q * global_context * v` is a triple product of three
        unnormalized activations, so it overflows fp16 well before any single tensor
        looks large. See model/MeSAEFlat/MeSAEFlat_modules.py's copy of this class for
        the measured failure (it took out a whole tokenizer run) — this class is
        duplicated per model package by convention, so the guard is duplicated too.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
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

        self.spatial_active = False
        nn.init.zeros_(self.spatial_attn.out_proj.weight)
        nn.init.zeros_(self.spatial_attn.out_proj.bias)

    def enable_spatial(self):
        self.spatial_active = True

    def forward(self, x):
        B, C, N, D = x.shape
        x_flat = x.view(B * C, N, D)

        x_norm_t = self.norm_time(x_flat)
        attn_out_t = self.temporal_attn(x_norm_t)
        x_flat = x_flat + self.drop_t(attn_out_t)

        x_space = x_flat.view(B, C, N, D).permute(0, 2, 1, 3).reshape(B * N, C, D)
        if self.spatial_active:
            x_norm = self.norm_space(x_space)
            attn_out, _ = self.spatial_attn(x_norm, x_norm, x_norm)
            x_space = x_space + self.drop_s(attn_out)
        x_flat = x_space.view(B, N, C, D).permute(0, 2, 1, 3).reshape(B * C, N, D)

        x_flat = x_flat + self.ffn(self.norm_ffn(x_flat))
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
# VQ & Decoder
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


class Router(nn.Module):
    """
    Top-k softmax router over the routed expert pool. Each expert is scored against its
    own Expert View (per-Expert channel-pooled vector, see ExpertChannelPool) rather than
    a shared input — the gating decision and the quantized input are always the same
    vector, so routing stays interpretable ("why did this route to expert e" is answerable
    from what e actually encoded). The top-k scores are softmax-normalized (weights sum to
    1, regardless of k) and scattered into a dense [M, n_experts] gate mask (nonzero only
    at the selected experts).

    Also returns the Switch-Transformer-style load-balance loss: dense softmax prob `p`
    (has grad, computed over ALL experts so non-selected experts still get pushed on) times
    hard selection frequency `f` (detached) — 1.0 at perfectly uniform routing, scales
    with `n_experts` so the value is comparable across different pool sizes.
    """
    def __init__(self, dim, n_experts, top_k):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.scale = dim ** -0.5
        self.weight = nn.Parameter(torch.empty(n_experts, dim))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, expert_views):
        """expert_views: [M, n_experts, D] (each expert's own pooled view) -> gate_mask [M, n_experts], lb_loss scalar"""
        # scaled dot-product: dot-product variance grows linearly with `dim`, so without this
        # a wide `dim` saturates the softmax almost immediately -> one expert wins early and
        # entrenches (router collapse).
        gate_logits = (expert_views * self.weight).sum(dim=-1) * self.scale  # [M, H]
        topk_val, topk_idx = gate_logits.topk(self.top_k, dim=-1)
        topk_weight = torch.softmax(topk_val, dim=-1).to(gate_logits.dtype)
        gate_mask = torch.zeros_like(gate_logits).scatter_(-1, topk_idx, topk_weight)

        f = (gate_mask.detach() > 0).float().mean(dim=0)     # [H] hard selection frequency
        p = torch.softmax(gate_logits, dim=-1).mean(dim=0)   # [H] dense prob, has grad
        lb_loss = self.n_experts * ((f / (f.sum() + 1e-8)) * (p / (p.sum() + 1e-8))).sum()
        return gate_mask, lb_loss


class MeFSQ(nn.Module):
    def __init__(self, num_heads, vq_head_vocab_size, e_dim, num_discrete=5, sigmoid_gain=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.r = vq_head_vocab_size
        self.e_dim = e_dim
        self.num_discrete = num_discrete
        self.sigmoid_gain = sigmoid_gain

        self.norm = nn.LayerNorm(e_dim)
        # [H, D, r] — same shared D-dim input z broadcasts against every head's own [D, r]
        # slice (via einsum, no need to materialize a per-head copy of z) — smaller than the
        # old flat [D, H*r] matrix would suggest per-head specialization requires, at the
        # same total param count. Plain linear projection, no orthogonality constraint.
        self.A = nn.Parameter(torch.empty(num_heads, e_dim, vq_head_vocab_size))
        # kaiming_uniform_ on the whole 3D tensor at once folds vq_head_vocab_size (r) into
        # the fan-in calc (torch treats dim>2 tensors as [out, in, *receptive_field]), so
        # fan_in = e_dim*r instead of the real e_dim -> bigger r shrinks each per-head slice's
        # init scale for no good reason. Init each head's [e_dim, r] slice separately instead
        # (matches MultiHeadDecoder's existing per-head loop) so fan_in is just e_dim.
        for h in range(num_heads):
            nn.init.kaiming_uniform_(self.A[h], a=math.sqrt(5))

        self.register_buffer('avg_probs',        torch.zeros(num_heads, vq_head_vocab_size, num_discrete))
        self.register_buffer('max_prob_ema',     torch.tensor(0.0))
        self.register_buffer('ema_ste_gap',      torch.tensor(0.0))
        self.register_buffer('ema_head_ppl_std', torch.tensor(0.0))
        self.ema_decay = 0.99

    def forward(self, z):
        """z: [M, H, D] — a separate D-dim input per head (per-head stage-weighted fusion upstream)."""
        M, H, D = z.shape
        z = self.norm(z)

        r, N_d = self.r, self.num_discrete
        half_range = (N_d - 1) / 2.0

        q = torch.einsum('mhd,hdr->mhr', z, self.A)
        q_soft = (N_d - 1) * torch.sigmoid(self.sigmoid_gain * q) - half_range
        q_quant = torch.round(q_soft)
        v_q = q_soft + (q_quant - q_soft).detach()
        indices = (q_quant + half_range).long()

        if not self.training:
            with torch.no_grad():
                B_total = M
                flat_idx = indices.reshape(B_total, H * r).t().clamp(0, N_d - 1)
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

        return v_q, indices, q_soft


# ==========================================
# Finetune head
# ==========================================

class PerChannelHeadAttn(nn.Module):
    """
    Two-stage learnable-query attention pooling over an already channel-pooled Expert
    signal (see MeFSQPretrain.encode_post_vq_expert): the channel dim is collapsed by the
    base model itself (each Expert's own channel-attention View), so this head does not
    re-pool channels — pooling channels a second time here would just be redundant work
    over a dim that no longer carries per-channel information worth weighting.

    Stage 1 (temporal): a plain linear scorer over N patches, softmax-normalized — replaces
    plain mean-over-N so patch/time relevance becomes an interpretable weight (attn_n)
    instead of being averaged away.

    Stage 2 (head): a plain linear scorer over the H Experts, softmax-normalized, pooling
    to a single [B, d] vector fed into cls.

    Both stages used to be a learnable-query dot-product ("key" Linear(d,hidden) dotted
    with a fixed learned "query" vector). That's algebraically just a single linear scalar
    function of z — composing two linear maps with no nonlinearity between them adds no
    expressiveness over one Linear(d,1), since the query is a fixed parameter, not
    content-derived (real cross-attention would need the query itself computed from
    content for the extra layer to matter). Collapsed to Linear(d,1) here: same capacity,
    fewer params, one matmul instead of two.

    (Earlier version had a 3rd stage pooling over channels, fed a per-channel-decoded
    signal from encode_post_vq. That's retired: the per-channel decode/re-pool was
    redundant given the backbone already collapses channels into the Expert View before
    VQ — see encode_post_vq_expert.)
    """
    def __init__(self, head_dim, num_classes, dropout=0.1):
        super().__init__()
        # encode_pre_vq fed this pooler through pre_vq_norm_routed/shared (LayerNorm);
        # encode_post_vq_expert's raw vq_proj output has no such norm. Un-normalized-scale
        # input starves the scorer and cls of usable gradient — normalize here so the
        # pooler works regardless of which encode_* the caller feeds it.
        self.input_norm = nn.LayerNorm(head_dim)
        self.score_n = nn.Linear(head_dim, 1)
        self.score_h = nn.Linear(head_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(head_dim, num_classes)
        # cls reads z_h's raw, un-normalized amplitude (see input_norm's docstring above) —
        # that scale is unbounded by design, so default init gives large initial logits and
        # an elevated first-epoch loss average that has nothing to do with the LR/schedule.
        # Small init instead: predictions start near-uniform, cls is still free to grow
        # weights as large as it needs during training.
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.zeros_(self.cls.bias)

    def forward(self, z_per_head, pad_mask=None):
        """
        z_per_head: [B, N, H, d]
        pad_mask: [B, N] bool, True = valid (optional, for padded patches)
        Returns (logits [B, num_classes], attn_h [B, H], attn_n [B, H, N])
        """
        # input_norm feeds ONLY the scorer below, not the z_h that flows to cls —
        # LayerNorm-ing z_per_head itself would renormalize near-zero (dead/unselected
        # head) vectors up to unit variance too, amplifying noise to the same magnitude as
        # heads carrying real signal and wiping out the amplitude cue the classifier needs.
        # The scorer gets a stable scale to compute attention weights from; the actual
        # pooled values stay at their real amplitude.
        # A gated-off routed Expert (encode_post_vq_expert already zeroes its contribution
        # per patch, see that docstring) has z_per_head == 0 at every (n, d) for this batch
        # item — LayerNorm can't tell "genuinely zero" from "small real signal", so
        # score_h's bias alone would still hand it a non-trivial softmax share in stage 2
        # (visible as spurious attention/importance for an Expert that contributed nothing).
        # Mask those out here instead of relying on the pooled value being zero to save them.
        alive_h = z_per_head.abs().sum(dim=(1, 3)) > 0                          # [B, H]

        z_key = self.input_norm(z_per_head)

        # ---- stage 1: attention pool over N (patches), per head ----
        logits_n = self.score_n(z_key).squeeze(-1)                              # [B, N, H]
        if pad_mask is not None:
            logits_n = logits_n.masked_fill(~pad_mask.unsqueeze(-1), float('-inf'))
        attn_n = torch.softmax(logits_n, dim=1)                                 # softmax over N, per head
        attn_n = torch.nan_to_num(attn_n)                                       # all-invalid batch item -> nan; zero it
        z_h = torch.einsum('bnhd,bnh->bhd', z_per_head, attn_n)                 # [B, H, d]
        z_h_key = torch.einsum('bnhd,bnh->bhd', z_key, attn_n)                  # [B, H, d] — for stage 2 logits only

        # ---- stage 2: attention pool over H (Experts) ----
        logits_h = self.score_h(z_h_key).squeeze(-1)                            # [B, H]
        logits_h = logits_h.masked_fill(~alive_h, float('-inf'))
        attn_h = torch.softmax(logits_h, dim=1)                                 # [B, H]
        attn_h = torch.nan_to_num(attn_h)                                       # all-dead batch item -> nan; zero it
        pooled = (z_h * attn_h.unsqueeze(-1)).sum(dim=1)                        # [B, d]

        attn_n = attn_n.permute(0, 2, 1)                                        # [B, H, N] for interpretability
        return self.cls(self.drop(pooled)), attn_h, attn_n
