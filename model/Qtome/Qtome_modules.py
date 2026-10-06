import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


def overlap_add_patches(patches, stride):
    """patches: [..., N, L] patch_stride-spaced, patch_len-long patches on the last two
    dims (any number of leading batch dims) -> [..., T] the real continuous signal they
    were sliced from, T = (N-1)*stride + L.

    A plain reshape(..., N*L) only gives the real signal back when stride == L
    (non-overlapping patches) — patch_stride has been < patch_len (some overlap) since
    before this codebase's current patch_len/stride values, so that reshape silently
    stacks every patch's FULL length back to back regardless of real overlap: the
    shared region between consecutive patches gets duplicated (inflating the apparent
    duration) and, for a differentiable model output like recon (unlike raw, whose
    duplicate copies are byte-identical — the same real samples read twice), the two
    patches' independently-computed views of that same shared moment generally
    disagree, showing a real discontinuity at every patch boundary. See
    model/Qtome/plugin.py's _run_reconstruction_sae for the diagnostic-display use of
    this, and QtomePretrain._recon_loss for the differentiable training-loss use.

    Linear-crossfade overlap-add instead: each patch gets a trapezoidal window — ramps
    0->1 over the incoming overlap it shares with the PREVIOUS patch, flat 1 over its
    own unique hop, ramps 1->0 over the outgoing overlap it shares with the NEXT one;
    no ramp on a side with no neighbor (the first/last patch). Every output sample is
    the weight-normalized sum of every patch covering it — a true weighted average, not
    dependent on the window being exactly constant-overlap-add — so this degrades to
    the old exact reshape behavior when stride == L (overlap == 0: every weight is 1).
    One F.fold call (a sum over every patch covering each sample) for both the weighted signal
    and the weights."""
    *lead, N, L = patches.shape
    assert stride <= L, f"overlap_add_patches: stride ({stride}) > patch_len ({L}) leaves gaps unfilled"
    overlap = L - stride
    w = torch.ones(N, L, device=patches.device, dtype=patches.dtype)
    if overlap > 0:
        ramp = torch.linspace(0, 1, overlap + 2, device=patches.device, dtype=patches.dtype)[1:-1]
        w[1:, :overlap] = ramp
        w[:-1, -overlap:] = ramp.flip(0)
    return (fold_sum(patches * w, stride) / fold_sum(w, stride).clamp(min=1e-8)).reshape(*lead, -1)


def fold_sum(patches, stride):
    """[..., N, L] patches -> [..., T]: every sample the SUM of the patch samples covering it."""
    *lead, N, L = patches.shape
    T = (N - 1) * stride + L
    flat = patches.reshape(-1, N, L).transpose(1, 2)                        # [M, L, N]
    return F.fold(flat, (1, T), (1, L), stride=(1, stride)).reshape(*lead, T)


# ==========================================
# Embeddings
# ==========================================

def get_sinusoidal_pos(seq_len, dim, device):
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    sin_inp = torch.einsum("i,j->ij", t, inv_freq)
    pos_emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return pos_emb.unsqueeze(0)  # [1, SeqLen, Dim]


# Fixed log-spaced wavelengths for electrode positions in metres (a head is ~0.2 m across):
# 40 cm ~ hemisphere scale down to 2.5 cm ~ a dense cap's electrode spacing.
FOURIER_WAVELENGTHS_M = (0.40, 0.20, 0.10, 0.05, 0.025)


def fourier_features(p):
    """[..., 3] positions (m) -> [..., 3 + 3*2*len(FOURIER_WAVELENGTHS_M)]: raw xyz plus sin/cos
    of every axis at every wavelength. Gives an MLP a multi-scale basis over the scalp, which
    a raw-xyz MLP lacks (measured: it drifted to a near-constant, channel-independent output)."""
    w = 2 * math.pi / torch.tensor(FOURIER_WAVELENGTHS_M, device=p.device, dtype=p.dtype)   # [F]
    ang = (p.unsqueeze(-1) * w).flatten(-2)                                                 # [..., 3F]
    return torch.cat([p, ang.sin(), ang.cos()], dim=-1)


FOURIER_DIM = 3 + 3 * 2 * len(FOURIER_WAVELENGTHS_M)


class RelativeSpatialBias(nn.Module):
    """Per-block, per-head bias added to spatial attention scores from the DIRECTIONAL
    position difference of the two electrodes: b(i, j) = MLP(fourier(p_i - p_j)), so the model
    can learn asymmetric relations (posterior vs anterior, left vs right), not only distance.
    Depends only on the montage, so it's computed once per window and shared over time.
    Last layer zero-initialised: training starts exactly as without it. -> [B, depth, H, C, C]."""
    def __init__(self, depth, num_heads, hidden=32):
        super().__init__()
        self.depth, self.heads = depth, num_heads
        out = nn.Linear(hidden, depth * num_heads)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.mlp = nn.Sequential(nn.Linear(FOURIER_DIM, hidden), nn.GELU(), out)

    def forward(self, coords):
        B, C, _ = coords.shape
        rel = coords[:, :, None, :] - coords[:, None, :, :]                                   # [B, C, C, 3]  i - j
        b = self.mlp(fourier_features(rel.float()))                                          # [B, C, C, depth*H]
        return b.view(B, C, C, self.depth, self.heads).permute(0, 3, 4, 1, 2)                # [B, depth, H, C, C]


TEMPORAL_WAVELENGTHS = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)   # in fine patches (x patch_stride samples)


class RelativeTemporalBias(nn.Module):
    """Per-block, per-head bias added to temporal attention scores from the SIGNED time lag of
    the two tokens, b(i, j) = MLP([lag, sin/cos(2 pi lag / wavelength)]), lag in fine-patch units
    so it means the same time at every pooling stage (coarse tokens sit at their patches' mean
    position). Signed: past and future can differ. Last layer zero-initialised: training starts
    exactly as without it. lag [n, n] -> [depth, H, n, n]."""
    def __init__(self, depth, num_heads, hidden=32):
        super().__init__()
        self.depth, self.heads = depth, num_heads
        out = nn.Linear(hidden, depth * num_heads)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.mlp = nn.Sequential(nn.Linear(1 + 2 * len(TEMPORAL_WAVELENGTHS), hidden), nn.GELU(), out)

    def forward(self, lag):
        w = 2 * math.pi / torch.tensor(TEMPORAL_WAVELENGTHS, device=lag.device, dtype=lag.dtype)
        ang = lag[..., None] * w
        b = self.mlp(torch.cat([lag[..., None], ang.sin(), ang.cos()], dim=-1))              # [n, n, depth*H]
        n = lag.shape[0]
        return b.view(n, n, self.depth, self.heads).permute(2, 3, 0, 1)                       # [depth, H, n, n]


class SpatialTemporalEmbeddings(nn.Module):
    def __init__(self, patch_len, dim, max_patches=5000, spatial=True):
        super().__init__()
        self.proj = nn.Linear(patch_len, dim)
        self.norm = nn.LayerNorm(dim)
        # Learnable, warm-started from the sinusoidal code (not random init) - an
        # ablation showed the FIXED sinusoidal version was measurably inert (shuffling
        # or zeroing time_idx changed reconstruction MSE by <0.1%, noise-level, despite
        # carrying real magnitude comparable to the content embedding). A fixed code
        # assumes a generic Transformer inductive bias this task never demonstrably
        # used; starting from the same values and letting gradient move them gives it
        # a chance to find something this task actually rewards, without losing
        # whatever structure the sinusoidal init already provides for free.
        self.pos_emb = nn.Parameter(get_sinusoidal_pos(max_patches, dim, torch.device('cpu')))
        self.spatial_active = False
        # spatial=False: no coordinate embedding at all (the spatial-embedding ablation; the model
        # then pairs it with no RelativeSpatialBias either, see QtomePretrain).
        # bias=False on BOTH linears: a bias on either one is a channel-INDEPENDENT constant the
        # network can add regardless of coords (it collapsed to exactly that before); with no bias
        # anywhere, per-channel variation is the only thing this path can produce.
        self.coord_proj = None
        if spatial:
            _coord_out = nn.Linear(dim // 4, dim, bias=False)
            nn.init.zeros_(_coord_out.weight)
            self.coord_proj = nn.Sequential(nn.Linear(FOURIER_DIM, dim // 4, bias=False), nn.GELU(), _coord_out)

    def enable_spatial(self):
        self.spatial_active = True

    def forward(self, x, coords=None, time_idx=None, bool_masked_pos=None, mask_token=None):
        B, C, N, L = x.shape
        z = self.proj(x.reshape(B * C, N, L))  # [B*C, N, D]
        if bool_masked_pos is not None:
            # masked content -> mask_token BEFORE the position terms below (see QtomePretrain)
            z = torch.where(bool_masked_pos.reshape(B * C, N, 1), mask_token.reshape(1, 1, -1).to(z.dtype), z)

        if time_idx is not None:
            t = time_idx.clamp(0, self.pos_emb.shape[1] - 1)
            temp_emb = self.pos_emb[0][t]       # [B, N, D]
            z = z + temp_emb.unsqueeze(1).expand(B, C, N, -1).reshape(B * C, N, -1)
        else:
            z = z + self.pos_emb[:, :N, :]

        if coords is not None and self.spatial_active and self.coord_proj is not None:
            s = self.coord_proj(fourier_features(coords).reshape(B * C, -1)).unsqueeze(1)  # [B*C, 1, D]
            z = z + s

        return self.norm(z).reshape(B, C, N, -1)


# ==========================================
# TSA Encoder
# ==========================================

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
    FilterRouter, applied to a much larger token
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
    weight). Replaces the single dense FFN sub-layer in
    TSABlock.

    Each expert's inner width is a fraction of the dense FFN's hidden_dim (dim * mlp_ratio)
    so total *active* per-token compute (n_shared + top_k experts firing) stays roughly at
    parity with a single dense FFN of that hidden_dim — standard DeepSeekMoE
    fine-grained-expert sizing. mlp_ratio is the single knob controlling expert width.

    Routed Experts run only on the tokens routed to them (index_add back, gate-weighted).
    """
    def __init__(self, dim, hidden_dim, n_routed, n_shared, top_k, dropout=0.0):
        super().__init__()
        self.n_routed = n_routed
        self.n_shared = n_shared
        expert_hidden = max(8, hidden_dim // (n_shared + top_k))

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
        self._record_health(gate_mask)
        routed_sum = x_flat.new_zeros(x_flat.shape)                               # [T, D]
        for r, e in enumerate(self.routed_experts):                               # only the tokens routed to e
            idx = gate_mask[:, r].nonzero(as_tuple=True)[0]
            if len(idx):
                routed_sum = routed_sum.index_add(0, idx, (e(x_flat[idx]) * gate_mask[idx, r, None]).to(routed_sum.dtype))

        shared_sum = x_flat.new_zeros(x_flat.shape)
        for e in self.shared_experts:
            shared_sum = shared_sum + e(x_flat)

        out = (routed_sum + shared_sum).reshape(BC, N, D)
        return out, lb_loss

    @torch.no_grad()
    def _record_health(self, gate_mask):
        """Same router-health formulas as QtomePretrain.update_head_metrics (entropy of the
        LOAD distribution across routed Experts, entropy of the WITHIN-token gate weights,
        load std) — computed every forward call (cheap, R is small) and stashed on self so
        TSAEncoder.forward can average across all TSABlocks' MoEFFNs into one dashboard
        number, mirroring the SAE Filter router's diagnostic but kept as a separate metric
        (two distinct MoEs, two distinct health
        readouts)."""
        selected = (gate_mask > 0).float()
        load = selected.mean(dim=0)
        load_p = load / (load.sum() + 1e-8)
        self.last_router_entropy = -(load_p * torch.log(load_p + 1e-10)).sum()
        self.last_router_load_std = load.std()
        # .float() matters here: this runs inside forward(), under autocast during
        # training, so gate_mask is fp16 — 1e-10 underflows to exactly 0.0 in fp16, making
        # log(0+0)=-inf and 0*-inf=NaN for every masked-out (always-present) entry, which
        # _ema_update's NaN-guard then silently skips forever (see QtomePretrain.
        # update_head_metrics's gate_routed.detach().float().clamp(...) for the same fix
        # applied to the SAE Filter router's equivalent metric).
        gm = gate_mask.float().clamp(min=0)
        self.last_gate_entropy = -(gm * torch.log(gm + 1e-10)).sum(dim=-1).mean()


class TSABlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., dropout=0.0,
                 n_routed_ffn_experts=4, n_shared_ffn_experts=1, ffn_top_k=2):
        super().__init__()
        self.norm_time = nn.LayerNorm(dim)
        # Same nn.MultiheadAttention as spatial_attn below, over the PATCH axis N
        # instead of the channel axis C — so temporal mixing goes through PyTorch's
        # fused SDPA/flash kernels too. Softmax attention is a convex combination of v,
        # so its output is bounded by v's own range regardless of block depth or scale
        # drift. Position comes from SpatialTemporalEmbeddings' sinusoidal time
        # embedding, which MHA needs.
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_t = nn.Dropout(dropout)
        # Bounds this branch's raw output BEFORE scale_t multiplies it (applied to
        # attn_out_t/attn_out/ffn_out in forward(), right after _watch measures the
        # UNbounded version) — see the LayerScale comment below for why this is needed on
        # top of LayerScale: without it, a branch's own weights can grow arbitrarily to
        # counteract a tiny scale_t/s/ffn init, which is exactly what branch_max measures.
        self.norm_time_out = nn.LayerNorm(dim)

        self.norm_space = nn.LayerNorm(dim)
        # dropout here is on the attention WEIGHTS themselves (nn.MultiheadAttention's own
        # `dropout` arg), on top of drop_s below which drops the branch's output — two
        # different regularization points, same shared `dropout` value.
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_s = nn.Dropout(dropout)
        self.norm_space_out = nn.LayerNorm(dim)  # see norm_time_out

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = MoEFFN(dim, hidden_dim=int(dim * mlp_ratio), n_routed=n_routed_ffn_experts,
                           n_shared=n_shared_ffn_experts, top_k=ffn_top_k,
                           dropout=dropout)
        self.norm_ffn_out = nn.LayerNorm(dim)  # see norm_time_out

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
        # active for the whole run and throttles each branch's net CONTRIBUTION to x. It
        # does NOT bound the branch's own INTERNAL magnitude, though — LayerScale only
        # multiplies the branch's output on its way out, so gradient descent can (and does:
        # measured branch_max ~2500 on a converged FFN branch after 50 epochs, out_proj/fc2
        # weights growing to counteract scale_ffn's tiny init rather than the branch
        # learning to stay small) inflate the branch's own weights to reach whatever
        # effective magnitude it wants regardless of how small scale_t/s/ffn is. norm_*_out
        # (added right after each branch, before its scale multiply — see norm_time_out)
        # closes that loop: once the branch's raw output is itself normalized, scale_t/s/ffn
        # alone controls the residual contribution, removing the incentive to grow the
        # branch's internal weights in the first place. LayerScale still matters on top of
        # that (controls how much of the now-bounded branch reaches the residual stream).
        layerscale_init = 1e-4
        self.scale_t   = nn.Parameter(torch.full((dim,), layerscale_init))
        self.scale_s   = nn.Parameter(torch.full((dim,), layerscale_init))
        self.scale_ffn = nn.Parameter(torch.full((dim,), layerscale_init))

        # Both cross-patch (temporal, global context pooled over all N) and cross-channel
        # (spatial) mixing default off — Qtome's tokenizer stage trains the SAE on
        # patch-local features only, so the frozen dictionary can't leak already-seen
        # context into masked-stage reconstruction targets. Both out_proj-equivalents are
        # zero-inited so enabling later starts as a no-op and grows in under gradient,
        # instead of shocking a checkpoint that never saw either term active.
        self.temporal_active = False
        self.spatial_active = False
        self.last_branch_max = None  # see _watch
        nn.init.zeros_(self.temporal_attn.out_proj.weight)
        nn.init.zeros_(self.temporal_attn.out_proj.bias)
        nn.init.zeros_(self.spatial_attn.out_proj.weight)
        nn.init.zeros_(self.spatial_attn.out_proj.bias)

    def enable_temporal(self):
        self.temporal_active = True

    def enable_spatial(self):
        self.spatial_active = True

    def _watch(self, t):
        """Track the largest magnitude any branch produces BEFORE norm_*_out/LayerScale
        shrink it — called on the raw attn_out_t/attn_out/ffn_out, ahead of both.

        This is the blind spot that cost a tokenizer run. norm_time/norm_space/norm_ffn
        bound each branch's INPUT and norm_out bounds the block's OUTPUT, but nothing
        bounded what happened between them — and scale_t/scale_s/scale_ffn (init 1e-4)
        multiply the branch output on its way to the residual add, so an enormous
        interior arrived at the stream as a whisper. block_norm measures the post-scale
        contribution, i.e. the wrong side of that multiplication: it can stay tame right
        up to the moment a branch's interior crosses fp16's 65504 ceiling. norm_*_out
        (see the LayerScale comment above) now closes that gap architecturally, but this
        stays a raw pre-norm probe on purpose — it's the canary for the underlying matmul
        itself, independent of whatever norm_*_out does downstream.

        Deliberately NOT gated on `not self.training`, unlike last_block_norms: the
        overflow happened in TRAIN mode (eval on the same weights was finite), so an
        eval-only probe cannot see this class of failure at all. Cost is one amax
        reduction per branch — negligible against the block's matmuls — and it stays a
        tensor here so no per-step GPU sync happens; the single .item() is paid once per
        epoch in get_metrics.
        """
        m = t.detach().abs().amax()
        self.last_branch_max = m if self.last_branch_max is None else torch.maximum(self.last_branch_max, m)

    def _spatial_attention(self, x, valid_channels, spatial_bias):
        """x [B, N, C, D] -> [B, N, C, D]. spatial_attn's own weights, run through
        scaled_dot_product_attention so the per-block bias [B, H, C, C] and the padded-channel
        key mask broadcast over the N patches instead of being copied B*N times."""
        B, N, C, D = x.shape
        mha = self.spatial_attn
        H = mha.num_heads
        q, k, v = F.linear(x, mha.in_proj_weight, mha.in_proj_bias).chunk(3, dim=-1)
        q, k, v = (t.reshape(B, N, C, H, D // H).transpose(2, 3) for t in (q, k, v))   # [B, N, H, C, d]
        mask = None
        if spatial_bias is not None:
            mask = spatial_bias.to(q.dtype)[:, None]                                    # [B, 1, H, C, C]
        if valid_channels is not None:
            pad = torch.zeros(B, 1, 1, 1, C, dtype=q.dtype, device=q.device).masked_fill(
                ~valid_channels.bool()[:, None, None, None, :], float('-inf'))
            mask = pad if mask is None else mask + pad
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=mha.dropout if self.training else 0.0)
        return mha.out_proj(out.transpose(2, 3).reshape(B, N, C, D))

    def _temporal_attention(self, x, kpm, temporal_bias):
        """x [B*C, N, D] -> [B*C, N, D]. Without a bias: temporal_attn as a plain MHA call. With
        one ([H, N, N]): the same weights through scaled_dot_product_attention, the bias and the
        padded-patch key mask broadcast over B*C."""
        mha = self.temporal_attn
        if temporal_bias is None:
            return mha(x, x, x, key_padding_mask=kpm)[0]
        BC, N, D = x.shape
        H = mha.num_heads
        q, k, v = F.linear(x, mha.in_proj_weight, mha.in_proj_bias).chunk(3, dim=-1)
        q, k, v = (t.reshape(BC, N, H, D // H).transpose(1, 2) for t in (q, k, v))       # [BC, H, N, d]
        mask = temporal_bias.to(q.dtype)[None]                                            # [1, H, N, N]
        if kpm is not None:
            mask = mask + torch.zeros(BC, 1, 1, N, dtype=q.dtype, device=q.device).masked_fill(
                kpm[:, None, None, :], float('-inf'))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=mha.dropout if self.training else 0.0)
        return mha.out_proj(out.transpose(1, 2).reshape(BC, N, D))

    def forward(self, x, valid_channels=None, spatial_bias=None, valid_patches=None, temporal_bias=None):
        """valid_channels [B, C] bool (optional): zero-padded channels are left out of
        spatial attention as keys, so a montage's missing channels can't dilute the
        softmax (their own rows still get computed, then ignored downstream).
        valid_patches [B, N] bool (optional): the same for temporal attention over a
        window's zero-padded tail patches."""
        B, C, N, D = x.shape
        x_flat = x.view(B * C, N, D)
        self.last_branch_max = None

        if self.temporal_active:
            x_norm_t = self.norm_time(x_flat)
            kpm_t = None if valid_patches is None else \
                (~valid_patches.bool()).repeat_interleave(C, dim=0)             # [B*C, N], True = ignore
            attn_out_t = self._temporal_attention(x_norm_t, kpm_t, temporal_bias)
            self._watch(attn_out_t)
            attn_out_t = self.norm_time_out(attn_out_t)
            x_flat = x_flat + self.drop_t(self.scale_t * attn_out_t)

        x_space = x_flat.view(B, C, N, D).permute(0, 2, 1, 3).reshape(B * N, C, D)
        if self.spatial_active:
            x_norm = self.norm_space(x_space).view(B, N, C, D)
            attn_out = self._spatial_attention(x_norm, valid_channels, spatial_bias).reshape(B * N, C, D)
            self._watch(attn_out)
            attn_out = self.norm_space_out(attn_out)
            x_space = x_space + self.drop_s(self.scale_s * attn_out)
        x_flat = x_space.view(B, N, C, D).permute(0, 2, 1, 3).reshape(B * C, N, D)

        ffn_out, ffn_lb_loss = self.ffn(self.norm_ffn(x_flat))
        self._watch(ffn_out)
        ffn_out = self.norm_ffn_out(ffn_out)
        x_flat = x_flat + self.scale_ffn * ffn_out
        x_flat = self.norm_out(x_flat)
        return x_flat.view(B, C, N, D), ffn_lb_loss


class TemporalUpBlock(nn.Module):
    """Decoder block after an upsample: per channel (weights shared, channels never mix), two
    1-D convs along the patch axis, residual, pre/post LayerNorm and LayerScale as in TSABlock.
    x [B, C, N, D]; valid_patches [B, N] bool keeps padded tail patches out of the convs."""
    def __init__(self, dim, kernel=3, layerscale_init=1e-4):
        super().__init__()
        self.norm_in, self.norm_out = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(dim, dim, kernel, padding=kernel // 2)
        self.scale = nn.Parameter(torch.full((dim,), layerscale_init))

    def forward(self, x, valid_patches=None):
        B, C, N, D = x.shape
        y = self.norm_in(x)
        keep = None if valid_patches is None else valid_patches[:, None, :, None].to(y.dtype)  # [B, 1, N, 1]
        if keep is not None:
            y = y * keep
        y = y.reshape(B * C, N, D).transpose(1, 2)                      # [B*C, D, N]: merges adjacent B, C
        y = self.conv2(F.gelu(self.conv1(y))).transpose(1, 2).reshape(B, C, N, D)
        y = self.norm_out(y)
        if keep is not None:
            y = y * keep
        return x + self.scale * y


class TSAEncoder(nn.Module):
    """TSABlocks in stages of blocks_per_stage: the patch axis N is pooled by 2 after every stage
    but the last (depth 8, 2 per stage: 4 stages at N, N/2, N/4, N/8 -- 39 -> 20 -> 10 -> 5), and
    the way back up adds each stage's pre-pool output (a UNet skip, gated) onto the upsampled
    deeper result. Pooling is centred ([1,3,3,1]/8, coarse token i sits between fine patches 2i and
    2i+1) and upsampling interpolates linearly at the same positions, so no level shifts time.

    skip_mode 'gated' (the default) adds the skips; 'none' drops them, so everything reaching the
    output passes through the deepest stage; 'finest' keeps only the finest skip (per-patch detail)
    and drops the deeper ones, so everything coarser than a patch must pass the deep path. skip_drop p (training only, gated): each skip is
    dropped per sample with probability p and kept ones scaled by 1/(1-p) (drop-path), so the
    deep path must carry the patch detail part of the time; a list gives one p per skip, finest
    first (the skip_gate_0/1/2 order). decoder_blocks > 0 puts that many
    TemporalUpBlocks (per-channel temporal convs, no channel mixing) after each upsample.
    temporal_bias: a RelativeTemporalBias on every block's temporal attention (signed lag)."""
    def __init__(self, dim, depth=8, num_heads=8, mlp_ratio=4., dropout=0.0, blocks_per_stage=2,
                 n_routed_ffn_experts=4, n_shared_ffn_experts=1, ffn_top_k=2,
                 skip_mode='gated', decoder_blocks=0, skip_drop=0.0, temporal_bias=False):
        super().__init__()
        assert depth % blocks_per_stage == 0, f"depth {depth} is not a multiple of blocks_per_stage {blocks_per_stage}"
        assert skip_mode in ('gated', 'finest', 'none'), f"skip_mode {skip_mode!r}: 'gated', 'finest' or 'none'"
        block = lambda: TSABlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                                 n_routed_ffn_experts=n_routed_ffn_experts,
                                 n_shared_ffn_experts=n_shared_ffn_experts, ffn_top_k=ffn_top_k)
        self.blocks = nn.ModuleList([block() for _ in range(depth)])
        self.pool_after = [i for i in range(blocks_per_stage - 1, depth - 1, blocks_per_stage)]
        self.skip_mode = skip_mode
        self.temporal_bias = RelativeTemporalBias(depth, num_heads) if temporal_bias else None
        drops = list(skip_drop) if isinstance(skip_drop, (list, tuple)) else [skip_drop] * len(self.pool_after)
        assert len(drops) == len(self.pool_after), f"skip_drop {skip_drop}: one p per skip ({len(self.pool_after)})"
        assert all(0.0 <= p < 1.0 for p in drops) and (not any(drops) or skip_mode != 'none'), \
            f"skip_drop {skip_drop} needs skips (skip_mode 'gated' / 'finest') and 0 <= p < 1"
        self.skip_drop = [float(p) for p in drops]       # finest skip first
        if skip_mode != 'none':   # 'finest': one gate, for the finest skip (index 0)
            n_gates = len(self.pool_after) if skip_mode == 'gated' else 1
            self.skip_gates = nn.ParameterList([nn.Parameter(torch.tensor(3.0)) for _ in range(n_gates)])
        else:
            self.skip_gates = None
        # one list per upsample level, deepest level first (the order they run in)
        self.dec_blocks = nn.ModuleList([nn.ModuleList([TemporalUpBlock(dim) for _ in range(decoder_blocks)])
                                         for _ in self.pool_after]) if decoder_blocks else None

    def enable_spatial(self):
        for block in self.blocks:
            block.enable_spatial()

    def enable_temporal(self):
        for block in self.blocks:
            block.enable_temporal()

    @staticmethod
    def _pool(x):
        """[B, C, N, D] -> [B, C, ceil(N/2), D]: coarse token i = (x[2i-1] + 3 x[2i] + 3 x[2i+1]
        + x[2i+2]) / 8, centred on 2i + 0.5 (a lowpass before decimating, so content above the
        new Nyquist rate doesn't alias). Edges replicate; an odd N repeats its last token."""
        if x.shape[2] % 2:
            x = torch.cat([x, x[:, :, -1:]], dim=2)
        M = x.shape[2] // 2
        xp = torch.cat([x[:, :, :1], x, x[:, :, -1:]], dim=2)                 # xp[j] = x[j - 1]
        return (xp[:, :, 0:2 * M:2] + 3 * xp[:, :, 1:2 * M + 1:2]
                + 3 * xp[:, :, 2:2 * M + 2:2] + xp[:, :, 3:2 * M + 3:2]) / 8.0

    @staticmethod
    def _upsample(x, n):
        """[B, C, M, D] -> [B, C, n, D]: fine patch j reads the coarse sequence at (j - 0.5) / 2
        (coarse token i sits at fine position 2i + 0.5), linear interpolation, clamped at the ends."""
        M = x.shape[2]
        u = ((torch.arange(n, device=x.device, dtype=x.dtype) - 0.5) / 2).clamp(0, M - 1)
        i0 = u.floor().long().clamp(max=M - 1)
        i1 = (i0 + 1).clamp(max=M - 1)
        f = (u - i0.to(u.dtype)).view(1, 1, n, 1)
        return x[:, :, i0] * (1 - f) + x[:, :, i1] * f

    def upsample_bottleneck(self, n):
        """last_bottleneck (the deepest stage's output) back to n patches through the same linear
        upsamples, no skips, no decoder: a fixed full-rank linear map, so a linear probe on it
        sees exactly the bottleneck."""
        sizes = [n]
        for _ in self.pool_after:
            sizes.append((sizes[-1] + 1) // 2)
        x = self.last_bottleneck
        for m in reversed(sizes[:-1]):
            x = self._upsample(x, m)
        return x

    @staticmethod
    def _pool_valid(v):
        """[B, N] bool -> [B, ceil(N/2)]: a coarse token is real if either of its two patches is."""
        if v.shape[1] % 2:
            v = torch.cat([v, v[:, -1:]], dim=1)
        return v[:, 0::2] | v[:, 1::2]

    def forward(self, x, valid_channels=None, spatial_bias=None, valid_patches=None):
        """spatial_bias: optional [B, depth, H, C, C] from RelativeSpatialBias, block i gets [:, i].
        valid_patches: optional [B, N] bool, False on a window's zero-padded tail patches (left out
        of temporal attention as keys)."""
        skips = []  # (pre-pool tensor) per pool point, in block order
        # Per-block contribution norm (eval only): how much each block changes its input. A block
        # right before a pool point has its delta scaled by sigmoid(its skip gate), since that is
        # how strongly its output is re-added at the end.
        record_norms = not self.training
        if record_norms:
            self.last_block_norms = []
            gate_for_block = dict(zip(self.pool_after, self.skip_gates)) if self.skip_gates is not None else {}
        ffn_lb_loss = x.new_zeros(())
        vp = valid_patches
        pos = torch.arange(x.shape[2], device=x.device, dtype=torch.float32)            # token times, fine patches
        tb = self.temporal_bias(pos[:, None] - pos[None, :]) if self.temporal_bias is not None else None
        for i, block in enumerate(self.blocks):
            x_in = x
            x, blk_ffn_lb = block(x, valid_channels, None if spatial_bias is None else spatial_bias[:, i], vp,
                                  None if tb is None else tb[i])
            ffn_lb_loss = ffn_lb_loss + blk_ffn_lb
            if record_norms:
                with torch.no_grad():
                    delta = (x - x_in).norm(dim=-1).mean()
                    if i in gate_for_block:
                        delta = delta * torch.sigmoid(gate_for_block[i])
                    self.last_block_norms.append(delta.item())
            if i in self.pool_after:
                skips.append(x)
                x = self._pool(x)
                vp = None if vp is None else self._pool_valid(vp)
                if tb is not None:     # coarse token i sits at the mean time of fine tokens 2i, 2i+1
                    pos = torch.cat([pos, pos[-1:]]) if pos.shape[0] % 2 else pos
                    pos = (pos[0::2] + pos[1::2]) / 2
                    tb = self.temporal_bias(pos[:, None] - pos[None, :])

        self.last_bottleneck = x                                                        # [B, C, N_deep, D]
        vps = [valid_patches]                  # valid patches per level, finest first
        for _ in skips[1:]:
            vps.append(None if vps[-1] is None else self._pool_valid(vps[-1]))
        for level, skip in enumerate(reversed(skips)):
            x = self._upsample(x, skip.shape[2])
            i = len(skips) - 1 - level                                       # this skip's index, finest = 0
            if self.skip_mode == 'gated' or (self.skip_mode == 'finest' and i == 0):
                if self.training and self.skip_drop[i] > 0:
                    keep = (torch.rand(skip.shape[0], 1, 1, 1, device=skip.device) >= self.skip_drop[i])
                    skip = skip * keep.to(skip.dtype) / (1 - self.skip_drop[i])
                x = x + torch.sigmoid(self.skip_gates[i]) * skip
            for block in (self.dec_blocks[level] if self.dec_blocks is not None else []):
                x_in = x
                x = block(x, vps[-1 - level])
                if record_norms:
                    with torch.no_grad():
                        self.last_block_norms.append((x - x_in).norm(dim=-1).mean().item())

        # Router health and the worst interior branch magnitude (TSABlock._watch), over every block.
        with torch.no_grad():
            blocks = self.blocks
            self.last_ffn_router_entropy = torch.stack([b.ffn.last_router_entropy for b in blocks]).mean()
            self.last_ffn_router_load_std = torch.stack([b.ffn.last_router_load_std for b in blocks]).mean()
            self.last_ffn_gate_entropy = torch.stack([b.ffn.last_gate_entropy for b in blocks]).mean()
            watched = [b.last_branch_max for b in blocks if b.last_branch_max is not None]
            self.last_branch_max = torch.stack(watched).max() if watched else None

        return x, ffn_lb_loss


# ==========================================
# Decoder & channel pooling
# ==========================================

class AtomBank(nn.Module):
    """
    Static Q-atom dictionary over CHANNEL-GROUPED tokens: input is [G, C, D] where each group g is one
    patch POSITION (G = B*N) carrying all C channels' embeddings for that moment. Every Q-atom is active
    at every position; amplitude is read per channel. The instantaneous-mixing ICA picture made
    structural: x_c(t) = sum_s A[c, s] * source_s(t) -- D_hat_s is source s's waveform, and the [C]
    vector of a Q-atom's per-channel amps IS that source's mixing column (its topomap at that patch
    time). Routed (top-k) Q-atoms were removed (`routed-stamps` branch).

    phi_s(z_c) = rms_c * (a_s(z_c) * D_hat_s + b_s(z_c) * Hilbert(D_hat_s)): a fixed per-atom waveform
    TEMPLATE D_s (no z dependence, used UNIT-L2-NORMALIZED everywhere -- with a free-norm D, amp*D has
    a scale degeneracy) plus its DERIVED Hilbert quadrature partner (see _quadrature), combined by a
    per-CHANNEL gain pair (a, b) from the Q-atom's own small MLP on z -- amplitude sqrt(a^2+b^2), phase
    atan2(b, a) -- times that channel's raw-input RMS (the LayerNorm stack erases amplitude from z, so
    the gain multiplies it back in). Shape is a pure parameter and amplitude/phase the only
    z-dependent knobs, so "same waveform, different amplitude across channels" is structural
    (no per-token shape warping). A free [patch_len] template's frequency content is
    bound to the patch_len FFT grid (Df = fs/patch_len); oscillator atoms were withdrawn (0010).
    """
    def __init__(self, dim, patch_len, n_atoms=16, hidden_width=16, spatial_rank=0):
        super().__init__()
        self.n_atoms = n_atoms
        # spatial_rank K > 0: source-factorized gains. Each Q-atom's
        # [C] gain column is forced to rank K: src_sk = mean_c W_s[c, k] * g_cs (unmixing), g_cs <- sum_k
        # A_s[c, k] * src_sk (mixing). W and A are functions of the electrode position only (fixed scalp
        # fields, any montage); A_s[:, k] is source (s, k)'s topography, src_sk its (a, b) activation.
        self.spatial_rank = spatial_rank
        if spatial_rank:
            self.topo = nn.Sequential(nn.Linear(FOURIER_DIM, 64), nn.GELU(), nn.Linear(64, 2 * n_atoms * spatial_rank))
        # Normalizes z before it's used (z inherits whatever scale the encoder drifts to).
        self.input_norm = nn.LayerNorm(dim)
        # amp_s(z): per-atom MLP z -> hidden (GELU) -> quadrature gain pair (a, b). Because H is derived
        # from the SAME template, (a, b) can only re-phase and scale the shape, never morph it.
        self.W_down = nn.Parameter(torch.empty(n_atoms, dim, hidden_width))
        self.b_down = nn.Parameter(torch.zeros(n_atoms, hidden_width))
        nn.init.kaiming_uniform_(self.W_down, a=math.sqrt(5))
        self.w_amp = nn.Parameter(torch.empty(n_atoms, hidden_width, 2))
        self.b_amp = nn.Parameter(torch.zeros(n_atoms, 2))
        nn.init.kaiming_uniform_(self.w_amp, a=math.sqrt(5))
        # D_s: the Q-atom's waveform template, used unit-normalized; the raw parameter's norm is irrelevant.
        self.D = nn.Parameter(torch.randn(n_atoms, patch_len) * 0.02)

    def _amp(self, z):
        """z: [G, C, D] ALREADY input_norm'd tokens -> per-channel quadrature gains [G, C, n_atoms, 2],
        no rms. GELU between the two maps: without it they collapse into one of rank <= 2."""
        hidden = F.gelu(torch.einsum('gcd,hdk->gchk', z, self.W_down) + self.b_down)
        return torch.einsum('gchk,hkp->gchp', hidden, self.w_amp) + self.b_amp

    @staticmethod
    def _quadrature(D):
        """D: [M, L] unit templates -> each row's Hilbert quadrature partner [M, L], unit-normalized.
        Derived (rFFT, rotate every positive-frequency bin by -90 degrees, zero DC/Nyquist which have no
        quadrature, irFFT), NEVER a free parameter -- <D, H(D)> = 0 exactly, so a*D_hat + b*H_hat spans
        amplitude and constant phase of the template's analytic signal WITHOUT shape freedom.
        Re-normalized since zeroing DC/Nyquist drops that energy; an (almost-)pure-DC template's
        partner is degenerate -- its b head just learns ~0."""
        Fd = torch.fft.rfft(D.float(), dim=-1) * (-1j)
        Fd[..., 0] = 0
        if D.shape[-1] % 2 == 0:
            Fd[..., -1] = 0
        H = torch.fft.irfft(Fd, n=D.shape[-1], dim=-1)
        return F.normalize(H, dim=-1).to(D.dtype)

    def templates(self):
        """(D_hat, H_hat): unit templates and their quadrature partners, each [n_atoms, patch_len]."""
        D = F.normalize(self.D, dim=-1)
        return D, self._quadrature(D)

    def decode(self, amp):
        """amp: [G, C, n_atoms, 2] per-channel gains WITH rms (forward()'s out.amp) -> per-atom
        contribution [G, C, n_atoms, patch_len] = a*D_hat + b*H_hat, unsummed (forward's recon is its
        sum over Q-atoms). Pure re-expansion, no model re-evaluation."""
        D, H = self.templates()
        return amp[..., 0, None] * D + amp[..., 1, None] * H

    def _factorize(self, amp, coords, valid_channels):
        """amp [G, C, S, 2] per-channel gains (rms included), coords [G, C, 3] -> (amp with every Q-atom's
        column rank-K across channels [G, C, S, 2], src [G, S, K, 2] source activations). Unmixing is a mean
        over valid channels, so the scale doesn't depend on the montage size."""
        G, C, S, _ = amp.shape
        W, A = self.topo(fourier_features(coords.float())).view(G, C, 2, S, self.spatial_rank).to(amp.dtype).unbind(2)
        m = amp.new_ones(G, C) if valid_channels is None else valid_channels.to(amp.dtype)
        src = torch.einsum('gcsk,gcsp->gskp', W * m[..., None, None], amp) / m.sum(1).clamp(min=1)[:, None, None, None]
        return torch.einsum('gcsk,gskp->gcsp', A, src), src

    @torch.no_grad()
    def dense_amp(self, z, rms=None, coords=None, valid_channels=None):
        """z: [G, C, D] channel-grouped embeddings (same input forward() takes) -> amp [G, C, n_atoms, 2],
        input_norm'd, rms-scaled and (spatial_rank > 0) factorized the same way forward() is."""
        amp = self._amp(self.input_norm(z))
        amp = amp if rms is None else amp * rms.unsqueeze(-1)
        return self._factorize(amp, coords, valid_channels)[0] if self.spatial_rank else amp

    def forward(self, z, x_target=None, rms=None, valid_channels=None, coords=None):
        """
        z: [G, C, D] channel-grouped token embeddings (G = B*N patch positions), x_target: [G, C, patch_len]
        the real patch content (for mp_loss; None skips it), rms: [G, C, 1] per-channel raw-input RMS or
        None -- multiplied into every amp; callers running the real pipeline always pass it.
        valid_channels: [G, C] bool (True = real channel) or None -- excludes zero-padded channels from
        h and mp_loss; padded channels still decode.

        Returns recon [G, C, patch_len], amp [G, C, n_atoms, 2] (per-channel (a, b), rms included -- a
        Q-atom's [C] magnitude column is its phase-invariant topomap at this patch time), h [G, n_atoms]
        (post-rms amp magnitude averaged over valid channels: reconstruction-energy importance, orders
        mp_loss), mp_loss (scalar, valid-channel mean, for logging) and mp_map ([G, C], per position, what
        get_loss weights and trains on).
        """
        G, C, _ = z.shape
        amp = self._amp(self.input_norm(z))                                 # [G, C, S, 2]
        if rms is not None:
            amp = amp * rms.unsqueeze(-1)                                   # restores raw amplitude
        src = None
        if self.spatial_rank:
            amp, src = self._factorize(amp, coords, valid_channels)

        energy = amp.pow(2).sum(dim=-1)                                     # [G, C, S]
        if valid_channels is not None:
            vc = valid_channels.unsqueeze(-1).to(energy.dtype)              # [G, C, 1]
            h = ((energy * vc).sum(dim=1) / vc.sum(dim=1).clamp(min=1.0)).clamp(min=0).sqrt()
        else:
            h = energy.mean(dim=1).clamp(min=0).sqrt()                      # [G, S]

        D, H = self.templates()
        recon = torch.einsum('gck,kl->gcl', amp[..., 0], D) + torch.einsum('gck,kl->gcl', amp[..., 1], H)

        # Matching-Pursuit-style residual loss: rank the Q-atoms per position by h, then grade rank m
        # against x_target MINUS what ranks 0..m-1 already explained (detached, so gradient only pushes a
        # Q-atom toward what is still unexplained). A duplicate of a higher-ranked Q-atom sees a near-zero
        # residual and earns nothing. Doesn't change recon; only reshapes each Q-atom's training target.
        # Ranked per patch, not by Q-atom index (a fixed global hierarchy).
        mp_loss, mp_map = amp.new_zeros(()), None
        if x_target is not None:
            contrib = self.decode(amp)                                      # [G, C, S, L]
            order = h.argsort(dim=-1, descending=True)                      # [G, S]
            ranked = contrib.gather(2, order[:, None, :, None].expand_as(contrib))
            # Exclusive cumsum over rank of the detached contributions = every rank's residual at once.
            cum_excl = ranked.detach().cumsum(dim=2) - ranked.detach()
            diff2 = (ranked - (x_target.unsqueeze(2) - cum_excl)).pow(2)    # [G, C, S, L]
            mp_map = diff2.mean(dim=(2, 3))                                 # [G, C]
            if valid_channels is not None:
                vm = valid_channels.to(diff2.dtype)
                mp_loss = (mp_map * vm).sum() / vm.sum().clamp(min=1.0)
            else:
                mp_loss = mp_map.mean()

        return SimpleNamespace(recon=recon, amp=amp, h=h, mp_loss=mp_loss, mp_map=mp_map, src=src)


# ==========================================
# FINETUNE HEAD MODULES
# Everything above this line is the pretrain side.
# Shapes: a, b are the spatially mixed code amplitudes [B, N', K, S] (B trials, N' patches,
# K spatial filters, S Q-atoms); power = a^2 + b^2. Parameter names (p, q) and init match
# the QtomeFeatureHead so saved checkpoints load unchanged.
# ==========================================

def spatial_mix(spatial, t, dim):
    """Signed spatial filter (nn.Linear(C, K, bias=False)) over channel axis `dim`, or
    identity when spatial is None (channel concat)."""
    if spatial is None:
        return t
    return torch.movedim(spatial(torch.movedim(t, dim, -1).float()), -1, dim)


class PerAtomSpatial(nn.Module):
    """One signed spatial filter bank per Q-atom (filter-bank-CSP style): weight [S, K, C], so Q-atom s's code is mixed
    by its own K filters. [B, N', C, S] -> [B, N', K, S], the same shape the shared nn.Linear(C, K) gives.
    Init as nn.Linear's (uniform, bound 1/sqrt(C))."""
    def __init__(self, num_channels, k, num_atoms):
        super().__init__()
        b = 1.0 / math.sqrt(num_channels)
        self.weight = nn.Parameter(torch.empty(num_atoms, k, num_channels).uniform_(-b, b))

    def forward(self, t):
        return torch.einsum('bncs,skc->bnks', t.float(), self.weight)


class FlatTimePool(nn.Module):
    """Uniform weights over patches."""
    def forward(self, power):                                    # [B, N', K, S] -> [B, K, S]
        return power.mean(1)


class LearnedTimePool(nn.Module):
    """Low-rank softmax time weights w[s, n] = softmax_n(sum_r p[r, s] q[r, n]).
    Small init => starts equal to the flat mean."""
    def __init__(self, rank, num_atoms, num_patches):
        super().__init__()
        self.p = nn.Parameter(torch.randn(rank, num_atoms) * 0.02)
        self.q = nn.Parameter(torch.randn(rank, num_patches) * 0.02)

    def weights(self):                                           # [S, N']
        return torch.softmax(torch.einsum('rs,rn->sn', self.p, self.q), dim=-1)

    def forward(self, power):                                    # [B, N', K, S] -> [B, K, S]
        return torch.einsum('sn,bnks->bks', self.weights(), power)


class EvokedBranch(nn.Module):
    """Signed low-rank time filter T[s, n] = 1/N' + sum_r p[r, s] q[r, n] applied LINEARLY to
    a and b (phase-locked content survives a linear functional, not power)."""
    def __init__(self, rank, num_atoms, num_patches):
        super().__init__()
        self.p = nn.Parameter(torch.randn(rank, num_atoms) * 0.02)
        self.q = nn.Parameter(torch.randn(rank, num_patches) * 0.02)

    def forward(self, a, b):                                     # -> [B, K, 2*S]
        T = 1.0 / a.shape[1] + torch.einsum('rs,rn->sn', self.p, self.q)
        return torch.cat([torch.einsum('sn,bnks->bks', T, a),
                          torch.einsum('sn,bnks->bks', T, b)], dim=-1)


def phase_advance(a, b):
    """z[k, s] = sum_n u[n+1] conj(u[n]), u = a + i b, as real/imag parts (no complex dtype).
    Returns cat([z_re, z_im], -1) [B, K, 2*S]."""
    a_next, a_prev, b_next, b_prev = a[:, 1:], a[:, :-1], b[:, 1:], b[:, :-1]
    z_re = (a_next * a_prev + b_next * b_prev).sum(1)
    z_im = (b_next * a_prev - a_next * b_prev).sum(1)
    return torch.cat([z_re, z_im], dim=-1)


def _selfcheck_head_modules():
    """Runnable check against independent formulas; not run on import."""
    # self-check against independent formulas (complex dtype, explicit softmax)
    torch.manual_seed(0)
    B, N, K, S, R = 3, 39, 8, 25, 2
    a, b = torch.randn(B, N, K, S), torch.randn(B, N, K, S)
    u = torch.complex(a, b)
    z = (u[:, 1:] * u[:, :-1].conj()).sum(1)
    pa = phase_advance(a, b)
    assert torch.allclose(pa[..., :S], z.real, atol=1e-5) and torch.allclose(pa[..., S:], z.imag, atol=1e-5)
    ltp = LearnedTimePool(R, S, N)
    nn.init.normal_(ltp.p); nn.init.normal_(ltp.q)
    w = ltp.weights()
    assert torch.allclose(w.sum(-1), torch.ones(S), atol=1e-6)
    power = a.pow(2) + b.pow(2)
    ref = torch.einsum('sn,bnks->bks', w, power)
    assert torch.allclose(ltp(power), ref, atol=1e-6)
    nn.init.zeros_(ltp.p)                                        # zero logits => uniform => flat mean
    assert torch.allclose(ltp(power), FlatTimePool()(power), atol=1e-5)
    ev = EvokedBranch(R, S, N)
    nn.init.zeros_(ev.p)                                         # T = 1/N' => trial mean of a, b
    out = ev(a, b)
    assert torch.allclose(out[..., :S], a.mean(1), atol=1e-5) and torch.allclose(out[..., S:], b.mean(1), atol=1e-5)
    lin = nn.Linear(64, K, bias=False)
    t = torch.randn(B, N, 64, S)
    assert spatial_mix(lin, t, 2).shape == (B, N, K, S) and spatial_mix(None, t, 2) is t
    assert [n for n, _ in ltp.named_parameters()] == ['p', 'q'] and [n for n, _ in ev.named_parameters()] == ['p', 'q']

    # -- head config: one form, features = [{"type": ...}] with top-level defaults --
    base = dict(num_channels=22, num_atoms=25, sample_freq=200.0, patch_len=50, patch_stride=50)
    two = resolve_head_config(dict(features=[{'type': 'atom_power'}, {'type': 'raw_band', 'time_pool': 'flat'}],
                                   spatial_k=8, time_pool='learned', time_rank=2), num_patches=16, **base)
    assert needs_atom(two) and needs_raw(two) and feature_names(two) == ['atom_power', 'raw_band']
    assert _entry_cfg(two, 'atom_power')['time_pool'] == 'learned' and _entry_cfg(two, 'raw_band')['time_pool'] == 'flat'
    assert feature_dim(two) == _entry_dim(two, 'atom_power') + _entry_dim(two, 'raw_band')
    for bad in (dict(features=['atom_power']), dict(features=[{'type': 'atom_power'}] * 2),
                dict(features=[{'type': 'nope'}]), dict(features=[{'type': 'atom_power', 'typo': 1}]),
                dict(features=[{'type': 'atom_power'}], overrides={})):
        try:
            resolve_head_config(bad, num_patches=16, **base)
            assert False, f"{bad} should have raised"
        except ValueError:
            pass
    hcfg = resolve_head_config(dict(features=[{'type': 'atom_power', 'spatial_k': 4}, {'type': 'raw_band'},
                                              {'type': 'signed_ab'}, {'type': 'evoked', 'evoked_rank': 2}],
                                    time_pool='flat', dropout=0.0, time_rank=2),
                               num_patches=8, num_channels=6, num_classes=3, num_atoms=S,
                               sample_freq=200.0, patch_len=50, patch_stride=50)
    out = FeatureHead(hcfg)({'atom': torch.randn(2, 8, 6, S, 2), 'raw': torch.randn(2, 6, 8, 50)})
    assert out.shape == (2, 3), out.shape
    ps = resolve_head_config(dict(features=[{'type': 'atom_power', 'spatial_k': 2, 'spatial_per_atom': True}], time_pool='flat',
                                  dropout=0.0), num_patches=8, num_channels=6, num_classes=3, num_atoms=S,
                             sample_freq=200.0, patch_len=50, patch_stride=50)
    ph = FeatureHead(ps)
    x = torch.randn(2, 8, 6, S, 2)
    assert ph({'atom': x}).shape == (2, 3)
    W = ph.spatials['atom_power'].weight                        # [S, K, C]: Q-atom s mixed by its own filters only
    ref = torch.stack([x[..., s, 0] @ W[s].T for s in range(S)], -1)
    assert torch.allclose(ph.spatials['atom_power'](x[..., 0]), ref, atol=1e-5)

    print('head_modules self-check OK')


BANDS = ((8.0, 13.0), (13.0, 30.0))   # mu, beta
# Per-entry keys: set at the top level as the default for every entry, or inside one entry.
_ENTRY_KEYS = ('time_pool', 'time_rank', 'window', 'evoked_rank', 'spatial_k', 'atom_rank', 'latent_proj',
               'spatial_per_atom')
_HEAD_DEFAULTS = dict(features=[{'type': 'atom_power'}], spatial_k=8, time_pool='learned', time_rank=2,
                      window=None, evoked_rank=0, atom_rank=4, latent_proj='learned', spatial_per_atom=False, dropout=0.5,
                      latent_source='output')


def make_head_checkpoint(head, head_cfg, channel_idx, backbone_checkpoint):
    """Head-only checkpoint: state, resolved config (plus the real channels it was built for) and the
    frozen backbone it belongs to. Loaded by FinetuneModel.from_checkpoint."""
    return {'model_state_dict': head.state_dict(),
            'head_config': dict(head_cfg, channel_idx=list(channel_idx)),
            'backbone_checkpoint': backbone_checkpoint}


def feature_names(cfg):
    """The entry types of cfg['features'], in order."""
    return [f['type'] for f in cfg['features']]


def resolve_head_config(ft_params, **derived):
    """Head config = defaults + user keys + derived shapes, validated. features is a list of
    {"type": <ENTRY_TYPES name>, <_ENTRY_KEYS for that entry>}; top-level _ENTRY_KEYS are the
    defaults of every entry. Each entry gets its own spatial filter (its spatial_k, None/0 = no
    mixing, every real channel kept)."""
    user = dict(ft_params)
    unknown = set(user) - set(_HEAD_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown head keys {sorted(unknown)}; valid: {sorted(_HEAD_DEFAULTS)}")
    cfg = {**_HEAD_DEFAULTS, **user, **derived}
    feats = cfg['features']
    if not feats or not all(isinstance(f, dict) and f.get('type') in ENTRY_TYPES for f in feats):
        raise ValueError(f"features must be a non-empty list of {{'type': one of {FEATURES_ALL}, ...}}, got {feats!r}")
    names = feature_names(cfg)
    if len(names) != len(set(names)):
        raise ValueError(f"features has duplicate entries: {names}")
    for f in feats:
        bad = set(f) - {'type', *_ENTRY_KEYS}
        if bad:
            raise ValueError(f"features entry {f['type']!r}: unknown keys {sorted(bad)}; valid: {sorted(_ENTRY_KEYS)}")
    needs_np = False
    for name in names:
        e = _entry_cfg(cfg, name)
        k, etp = e['spatial_k'], e['time_pool']
        if k not in (None, 0) and not (isinstance(k, int) and k > 0):
            raise ValueError(f"spatial_k ('{name}') must be a positive int, or null/0 for no spatial mixing, got {k!r}")
        if etp not in ('flat', 'learned', 'window', 'none'):
            raise ValueError(f"features['{name}'] time_pool must be flat|learned|window|none, got {etp!r}")
        ENTRY_TYPES[name].check(e)
        if e['spatial_per_atom'] and (ENTRY_TYPES[name].source != 'atom' or not k):
            raise ValueError(f"spatial_per_atom ('{name}') needs a atom-code entry and spatial_k > 0")
        if etp == 'window' and not e['window']:
            raise ValueError(f"features['{name}'] time_pool='window' requires a window=[lo, hi]")
        if etp == 'learned' and int(e['time_rank']) < 1:
            raise ValueError(f"features['{name}'] time_pool='learned' requires time_rank >= 1")
        needs_np |= ENTRY_TYPES[name].needs_patches(e)
    if needs_np and cfg.get('num_patches') is None:
        raise ValueError("this configuration needs num_patches (trial length in patches)")
    return cfg


def _entry_cfg(cfg, name):
    """One entry's effective _ENTRY_KEYS (top-level default, the entry's own value wins) plus the
    shape keys every entry needs."""
    eff = {k: cfg.get(k, _HEAD_DEFAULTS[k]) for k in _ENTRY_KEYS}    # saved heads predate newer keys
    eff.update({k: v for k, v in next(f for f in cfg['features'] if f['type'] == name).items() if k != 'type'})
    for k in ('num_patches', 'num_atoms', 'sample_freq', 'patch_stride', 'patch_len', 'num_channels', 'latent_dim'):
        eff[k] = cfg.get(k)
    return eff


def needs_atom(cfg):
    """Whether any entry in cfg['features'] needs the frozen backbone (Q-atom codes, or z)."""
    return any(ENTRY_TYPES[f['type']].source in ('atom', 'latent') for f in cfg.get('features', _HEAD_DEFAULTS['features']))


def needs_raw(cfg):
    """Whether any entry in cfg['features'] needs the compiled raw signal."""
    return any(ENTRY_TYPES[f['type']].source == 'raw' for f in cfg.get('features', _HEAD_DEFAULTS['features']))


def needs_latent(cfg):
    """Whether any entry in cfg['features'] reads the encoder output z (cached with the Q-atoms)."""
    return any(ENTRY_TYPES[f['type']].source == 'latent' for f in cfg.get('features', _HEAD_DEFAULTS['features']))


def spatial_width(cfg, name):
    """K of one entry's spatial filter: spatial_k filters, or every real channel kept separate
    (spatial_k None/0 = no mixing)."""
    return _entry_cfg(cfg, name)['spatial_k'] or cfg['num_channels']


def _entry_dim(cfg, name):
    """Feature-vector width contributed by one features[] entry."""
    return ENTRY_TYPES[name].dim(_entry_cfg(cfg, name), spatial_width(cfg, name))


def feature_dim(cfg):
    """Width of the concatenated feature vector entering the readout."""
    return sum(_entry_dim(cfg, name) for name in feature_names(cfg))


class AtomExtractor(nn.Module):
    """Everything that needs the frozen backbone: Q-atom code (a, b) per patch, channel and
    Q-atom, scaled by patch RMS. Output [B, N', C_valid, S, 2] float32, padded channels dropped."""
    def __init__(self, backbone, channel_idx):
        super().__init__()
        self.backbone = backbone
        self.register_buffer('channel_idx', torch.as_tensor(channel_idx, dtype=torch.long))

    @torch.no_grad()
    def forward(self, x, coords, time_idx=None, valid_channels=None, impute_missing=False, return_z=False):
        """return_z: False, True / 'output' (the encoder output the Q-atoms read) or 'bottleneck' (the
        deepest stage, linearly upsampled back to the patch grid)."""
        """impute_missing (experiment, 2026-09-25): feed every missing channel as the pretrain
        mask_token at its own coordinate (coords must hold real positions for them), so spatial
        attention fills it in, and scale its Q-atom code by the mean patch RMS of its 3 nearest
        real channels (its own RMS is 0). Every channel then counts as valid."""
        B, C, N, L = x.shape
        vmask = (valid_channels if valid_channels is not None
                 else x.new_ones(B, C, dtype=torch.bool))
        rms = x.float().pow(2).mean(-1).sqrt()                                            # [B, C, N]
        masked = None
        if impute_missing:
            miss = ~vmask
            masked = miss[:, :, None].expand(B, C, N)
            d = torch.cdist(coords.float(), coords.float())                                # [B, C, C]
            d = d.masked_fill(miss[:, None, :], float('inf'))                              # only real channels as neighbours
            nn_idx = d.topk(min(3, int(vmask.sum(1).min())), dim=-1, largest=False).indices  # [B, C, k]
            nn_rms = torch.gather(rms[:, None].expand(B, C, C, N), 2,
                                  nn_idx[..., None].expand(-1, -1, -1, N)).mean(2)          # [B, C, N]
            rms = torch.where(miss[:, :, None], nn_rms, rms)
            vmask = torch.ones_like(vmask)
        z, _ = self.backbone.stage_features(x, coords, time_idx=time_idx, bool_masked_pos=masked,
                                            valid_channels=vmask)                       # [B, C, N, D]
        zg = z.permute(0, 2, 1, 3).reshape(B * N, C, -1)
        rg = rms.permute(0, 2, 1).reshape(B * N, C, 1)
        cg = coords.unsqueeze(1).expand(B, N, C, 3).reshape(B * N, C, 3)
        vg = vmask.unsqueeze(1).expand(B, N, C).reshape(B * N, C)
        amp = self.backbone.atoms.dense_amp(zg, rms=rg, coords=cg, valid_channels=vg)
        amp = amp.reshape(B, N, C, -1, 2).float() * vmask.float()[:, None, :, None, None]
        amp = amp[:, :, self.channel_idx]                                                 # [B, N, Cv, S, 2]
        if return_z:   # z for the latent_* head entries: [B, N, Cv, D]
            if return_z == 'bottleneck':
                z = self.backbone.encoder.upsample_bottleneck(N)
            return amp, z.permute(0, 2, 1, 3)[:, :, self.channel_idx].float()
        return amp

    def band_tables(self, sample_freq):
        """Per-atom template band energies (E_D, E_H), each [S, len(BANDS)], for feature='atom_band'."""
        with torch.no_grad():
            D_tab, H_tab = (t.float() for t in self.backbone.atoms.templates())
            fr = torch.fft.rfftfreq(D_tab.shape[-1], 1.0 / sample_freq)
            sel = [((fr >= lo) & (fr < hi)).to(D_tab.device) for lo, hi in BANDS]
            spec = lambda T: torch.stack([torch.fft.rfft(T, dim=-1).abs().pow(2)[:, m].sum(-1) for m in sel], -1)
            return spec(D_tab), spec(H_tab)


class NoTimePool(nn.Module):
    """Keep the patch axis: the features stay [B, N', K, F] and are flattened by the head."""
    def forward(self, power):
        return power


def _window_patches(cfg, N, device):
    """Bool mask [N] of patches fully inside cfg['window'] seconds (same rule as before)."""
    starts = torch.arange(N, device=device) * cfg['patch_stride']
    lo, hi = (w * cfg['sample_freq'] for w in cfg['window'])
    keep = (starts >= lo) & (starts + cfg['patch_len'] <= hi)
    assert keep.any(), f"window {cfg['window']} s keeps no patch"
    return keep


# ---------- head entry types: one class per features[] name (ENTRY_TYPES) ----------
# An entry owns its validation (check), its feature width (dim), whether it needs the trial
# length in patches (needs_patches) and its forward. `source` says what it reads: 'Q-atom' = the
# spatially mixed Q-atom codes (a, b), each [B, N', K, S]; 'raw' = the spatially mixed raw patches
# [B, K, N', L]. `e` is the entry's effective config
# (_entry_cfg). Parameter/buffer names and construction order are what saved heads were trained
# with: keep them when changing a class. A new head feature = one class + one ENTRY_TYPES line.

def _time_pool(e, n_feat):
    if e['time_pool'] == 'learned':
        return LearnedTimePool(int(e['time_rank']), n_feat, e['num_patches'])
    return NoTimePool() if e['time_pool'] == 'none' else FlatTimePool()


def _window(e, a, b):
    if e['time_pool'] != 'window':
        return a, b
    m = _window_patches(e, a.shape[1], a.device)
    return a[:, m], b[:, m]


def _pooled_dim(e, K, n_feat):
    return (e['num_patches'] if e['time_pool'] == 'none' else 1) * K * n_feat


class _Entry(nn.Module):
    source = 'atom'

    def __init__(self, e):
        super().__init__()
        self.e = e

    @staticmethod
    def check(e):
        pass

    @staticmethod
    def needs_patches(e):
        return e['time_pool'] in ('learned', 'none')


class AtomPowerEntry(_Entry):
    """log of time-pooled a^2 + b^2 per Q-atom: phase-invariant power (induced activity)."""
    def __init__(self, e):
        super().__init__(e)
        self.time = _time_pool(e, e['num_atoms'])

    @staticmethod
    def dim(e, K):
        return _pooled_dim(e, K, e['num_atoms'])

    def forward(self, a, b):
        a, b = _window(self.e, a, b)
        return torch.log(self.time(a.pow(2) + b.pow(2)) + 1e-12).flatten(1)


class AtomBandEntry(_Entry):
    """Q-atom power projected onto each Q-atom's template band energy (E_D/E_H, set from the
    backbone by the caller) -> mu / beta power."""
    def __init__(self, e):
        super().__init__(e)
        self.time = _time_pool(e, len(BANDS))
        self.register_buffer('E_D', torch.zeros(e['num_atoms'], len(BANDS)))
        self.register_buffer('E_H', torch.zeros(e['num_atoms'], len(BANDS)))

    @staticmethod
    def dim(e, K):
        return _pooled_dim(e, K, len(BANDS))

    def forward(self, a, b):
        a, b = _window(self.e, a, b)
        p = torch.einsum('bnks,sq->bnkq', a.pow(2), self.E_D) + torch.einsum('bnks,sq->bnkq', b.pow(2), self.E_H)
        return torch.log(self.time(p) + 1e-12).flatten(1)


class RawBandEntry(_Entry):
    """FFT band power (BANDS) of the mixed raw patches."""
    source = 'raw'

    def __init__(self, e):
        super().__init__(e)
        self.time = _time_pool(e, len(BANDS))

    @staticmethod
    def dim(e, K):
        return _pooled_dim(e, K, len(BANDS))

    def forward(self, x):
        e = self.e
        if e['time_pool'] == 'window':
            x = x[:, :, _window_patches(e, x.shape[2], x.device)]
        sp = torch.fft.rfft(x, dim=-1).abs().pow(2)                                 # [B, K, N, bins]
        fr = torch.fft.rfftfreq(x.shape[-1], 1.0 / e['sample_freq']).to(sp.device)
        p = torch.stack([sp[..., (lo <= fr) & (fr < hi)].sum(-1) for lo, hi in BANDS], -1)  # [B, K, N, 2]
        return torch.log(self.time(p.permute(0, 2, 1, 3)) + 1e-12).flatten(1)


class RawSignalEntry(_Entry):
    """The mixed raw signal itself, overlap-added and average-pooled to ~20 Hz."""
    source = 'raw'

    def __init__(self, e):
        super().__init__(e)
        self.pool_k = max(1, round(e['sample_freq'] / 20))

    @staticmethod
    def check(e):
        if e['time_pool'] != 'none':
            raise ValueError(f"features entry 'raw_signal' requires effective time_pool='none', got {e['time_pool']!r}")

    @staticmethod
    def needs_patches(e):
        return True

    @staticmethod
    def dim(e, K):
        pool = max(1, round(e['sample_freq'] / 20))
        return K * (((e['num_patches'] - 1) * e['patch_stride'] + e['patch_len']) // pool)

    def forward(self, x):
        sig = overlap_add_patches(x, self.e['patch_stride'])
        return torch.nn.functional.avg_pool1d(sig, self.pool_k, self.pool_k).flatten(1)


class PhaseAdvanceEntry(_Entry):
    """Phase advance between neighbouring patches, windowed by its own time_pool."""

    @staticmethod
    def dim(e, K):
        return 2 * K * e['num_atoms']

    def forward(self, a, b):
        return phase_advance(*_window(self.e, a, b)).flatten(1)


class EvokedEntry(_Entry):
    """Signed low-rank time filter over a and b; needs the full patch axis."""

    def __init__(self, e):
        super().__init__(e)
        self.branch = EvokedBranch(int(e['evoked_rank']), e['num_atoms'], e['num_patches'])

    @staticmethod
    def check(e):
        if e['time_pool'] == 'window':
            raise ValueError("features entry 'evoked' needs the full patch axis, not time_pool='window'")
        if int(e['evoked_rank']) < 1:
            raise ValueError("features entry 'evoked' requires evoked_rank >= 1 (effective)")

    @staticmethod
    def needs_patches(e):
        return True

    @staticmethod
    def dim(e, K):
        return 2 * K * e['num_atoms']

    def forward(self, a, b):
        return self.branch(a, b).flatten(1)


class SignedABEntry(_Entry):
    """signed_ab (2026-09-25): the signed, phase-locked counterpart of atom_power. Fully linear
    and factored: spatially mixed a and b (kept signed, so polarity and phase survive) ->
    learned Q-atom pooling S -> atom_rank -> learned time filters N' -> time_rank (init: flat
    average + small noise). A free linear readout on raw a/b would be the same function class
    but ~10^4 weights per class; the factoring is the regularisation. Carries evoked /
    phase-locked content only: induced (random-phase) power averages out, pair it with
    atom_power for that. -> [B, 2 * K * atom_rank * time_rank]."""
    def __init__(self, e):
        super().__init__(e)
        M, R, N = int(e['atom_rank']), int(e['time_rank']), e['num_patches']
        self.atom = nn.Linear(e['num_atoms'], M, bias=False)
        self.q = nn.Parameter(torch.full((R, N), 1.0 / N) + torch.randn(R, N) * 0.02)

    @staticmethod
    def check(e):
        if int(e['atom_rank']) < 1 or int(e['time_rank']) < 1:
            raise ValueError("features entry 'signed_ab' requires atom_rank >= 1 and time_rank >= 1")

    @staticmethod
    def needs_patches(e):
        return True

    @staticmethod
    def dim(e, K):
        return 2 * K * int(e['atom_rank']) * int(e['time_rank'])

    def forward(self, a, b):
        return torch.cat([torch.einsum('rn,bnkm->bkmr', self.q, self.atom(t)) for t in (a, b)], dim=1).flatten(1)


def _latent_proj(e, m):
    """z's D -> m projection. latent_proj 'learned': trained with the head. 'pca': frozen here, set
    by train_finetune.py's run_one to the top-m principal axes of the run's TRAINING-trial z
    (no trained parameters, like the Q-atom head's fixed templates)."""
    if e['latent_proj'] not in ('learned', 'pca'):
        raise ValueError(f"latent_proj must be learned|pca, got {e['latent_proj']!r}")
    proj = nn.Linear(e['latent_dim'], m, bias=False)
    proj.weight.requires_grad_(e['latent_proj'] == 'learned')
    return proj


class LatentPowerEntry(_Entry):
    """The atom_power pipeline on the encoder output z instead of the Q-atom codes: a learned
    projection shared over channels and time (D -> num_atoms, so the width equals atom_power's),
    squared, time-pooled, log."""
    source = 'latent'

    def __init__(self, e):
        super().__init__(e)
        self.proj = _latent_proj(e, e['num_atoms'])
        self.time = _time_pool(e, e['num_atoms'])

    @staticmethod
    def dim(e, K):
        return _pooled_dim(e, K, e['num_atoms'])

    def forward(self, z):
        z = _window(self.e, z, z)[0]
        return torch.log(self.time(self.proj(z).pow(2)) + 1e-12).flatten(1)


class LatentSignedEntry(_Entry):
    """The signed_ab pipeline on z: learned projection D -> 2 * atom_rank (signed_ab has a and b
    at atom_rank each, so the width matches), learned time filters -> time_rank."""
    source = 'latent'

    def __init__(self, e):
        super().__init__(e)
        M, R, N = 2 * int(e['atom_rank']), int(e['time_rank']), e['num_patches']
        self.proj = _latent_proj(e, M)
        self.q = nn.Parameter(torch.full((R, N), 1.0 / N) + torch.randn(R, N) * 0.02)

    check = SignedABEntry.check
    needs_patches = SignedABEntry.needs_patches

    @staticmethod
    def dim(e, K):
        return SignedABEntry.dim(e, K)

    def forward(self, z):
        return torch.einsum('rn,bnkm->bkmr', self.q, self.proj(z)).flatten(1)


ENTRY_TYPES = {'atom_power': AtomPowerEntry, 'atom_band': AtomBandEntry, 'raw_band': RawBandEntry,
               'raw_signal': RawSignalEntry, 'phase_advance': PhaseAdvanceEntry, 'evoked': EvokedEntry,
               'signed_ab': SignedABEntry, 'latent_power': LatentPowerEntry, 'latent_signed': LatentSignedEntry}
FEATURES_ALL = tuple(ENTRY_TYPES)


class FeatureHead(nn.Module):
    """Composable finetune head, no backbone inside: one ENTRY_TYPES submodule per
    cfg['features'] entry, each behind its own spatial filter, concatenated ->
    BatchNorm/Dropout/Linear readout."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        names = feature_names(cfg)
        self.entries = nn.ModuleDict({name: ENTRY_TYPES[name](_entry_cfg(cfg, name)) for name in names})
        # spatial_k None/0 = no mixing (ablation control): spatial_mix(None, ...) is identity,
        # so each real channel stays its own feature row instead of being pooled to K filters.
        self.spatials = nn.ModuleDict({name: (PerAtomSpatial(cfg['num_channels'], k, cfg['num_atoms'])
                                              if _entry_cfg(cfg, name)['spatial_per_atom']
                                              else nn.Linear(cfg['num_channels'], k, bias=False))
                                       for name in names if (k := _entry_cfg(cfg, name)['spatial_k'])})
        n_feat = feature_dim(cfg)
        # BatchNorm stands in for the probe's StandardScaler: log-powers are far from unit scale.
        self.cls = nn.Sequential(nn.BatchNorm1d(n_feat), nn.Dropout(cfg['dropout']),
                                 nn.Linear(n_feat, cfg['num_classes']))

    def forward(self, inp):
        """inp: {'raw': [B, C, N', L] or absent, 'Q-atom': [B, N', C, S, 2] or absent}."""
        # Feature math in fp32: under autocast, squared amplitudes overflow in fp16 and the 1e-12
        # epsilon rounds to 0 (NaN loss). The readout stays outside, as before.
        with torch.autocast(device_type=next(iter(inp.values())).device.type, enabled=False):
            raw = inp['raw'].float() if 'raw' in inp else None
            amp = inp['atom'].float() if 'atom' in inp else None
            latent = inp['latent'].float() if 'latent' in inp else None                # [B, N', C, D]
            outs = []
            for name, mod in self.entries.items():
                spatial = self.spatials[name] if name in self.spatials else None
                if mod.source == 'raw':
                    outs.append(mod(spatial_mix(spatial, raw, 1)))                           # [B, K, N', L]
                elif mod.source == 'latent':
                    outs.append(mod(spatial_mix(spatial, latent, 2)))                        # [B, N', K, D]
                elif isinstance(spatial, PerAtomSpatial):
                    outs.append(mod(spatial(amp[..., 0]), spatial(amp[..., 1])))              # each [B, N', K, S]
                else:
                    outs.append(mod(spatial_mix(spatial, amp[..., 0], 2),
                                    spatial_mix(spatial, amp[..., 1], 2)))                  # each [B, N', K, S]
            feat = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
        return self.cls(feat)
