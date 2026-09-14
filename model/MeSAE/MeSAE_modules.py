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
    model/MeSAE/plugin.py's _run_reconstruction_sae for the diagnostic-display use of
    this, and MeSAEPretrain._recon_loss for the differentiable training-loss use.

    Linear-crossfade overlap-add instead: each patch gets a trapezoidal window — ramps
    0->1 over the incoming overlap it shares with the PREVIOUS patch, flat 1 over its
    own unique hop, ramps 1->0 over the outgoing overlap it shares with the NEXT one;
    no ramp on a side with no neighbor (the first/last patch). Every output sample is
    the weight-normalized sum of every patch covering it — a true weighted average, not
    dependent on the window being exactly constant-overlap-add — so this degrades to
    the old exact reshape behavior when stride == L (overlap == 0: every weight is 1).
    Built via pad-then-stack-then-sum (not in-place slice accumulation) so it stays
    autograd-safe when patches requires grad."""
    *lead, N, L = patches.shape
    assert stride <= L, f"overlap_add_patches: stride ({stride}) > patch_len ({L}) leaves gaps unfilled"
    overlap = L - stride
    T = (N - 1) * stride + L
    flat = patches.reshape(-1, N, L)
    ramp = torch.linspace(0, 1, overlap + 2, device=patches.device, dtype=patches.dtype)[1:-1] \
        if overlap > 0 else None
    # Accumulate directly instead of stacking N full-length [.., T] tensors first — plain
    # addition is already a valid non-in-place autograd op, so this keeps the same
    # autograd-safety guarantee without the extra O(N*T) list/stack.
    out = flat.new_zeros(flat.shape[0], T)
    wsum = flat.new_zeros(T)
    for n in range(N):
        w = torch.ones(L, device=patches.device, dtype=patches.dtype)
        if overlap > 0:
            if n > 0:
                w[:overlap] = ramp
            if n < N - 1:
                w[-overlap:] = ramp.flip(0)
        pad = (n * stride, T - (n * stride + L))
        out = out + F.pad(flat[:, n, :] * w, pad)
        wsum = wsum + F.pad(w, pad)
    return (out / wsum.clamp(min=1e-8)).reshape(*lead, T)


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
        # bias=False on BOTH linears: a bias on either one is a channel-INDEPENDENT
        # constant the network can add regardless of coords — exactly the collapse
        # coord_scale (below) only partly fixed (dropping just coord_out's bias alone
        # isn't enough: the same constant just relocates into coord_proj[0]'s bias,
        # GELU passes it through nearly unchanged, and coord_out still linearly maps
        # that unchanged constant to the same output vector for every channel). With
        # no bias anywhere in the path, coords=0 -> output=0 exactly, so real per-
        # channel variation is the ONLY thing this path can produce, structurally.
        _coord_out = nn.Linear(dim // 4, dim, bias=False)
        nn.init.zeros_(_coord_out.weight)
        self.coord_proj = nn.Sequential(
            nn.Linear(3, dim // 4, bias=False),
            nn.GELU(),
            _coord_out,
        )
        # Real channel coords sit in a tiny range (head-radius units, magnitude ~0.1 —
        # e.g. standard_1020 channels sit ~0.08-0.12 from head center). Against coord_
        # proj[0]'s default Linear init (weight scale ~1/sqrt(3)), that's a barely-there
        # signal: measured on a trained checkpoint, coord_proj's per-channel output was
        # 99.1% cosine-similar across channels — 92% of its norm was a channel-independent
        # constant (effectively just a learned bias), only ~8% actually varied with
        # position. Recon loss never pushed back because per-channel CONTENT already
        # differs plenty (different electrode signal), so the coord path had no pressure
        # to earn its keep. A learnable scale multiplying coords before the MLP gives
        # position-dependent variation more leverage relative to that constant term, no
        # magic number tied to one montage's specific radius (adapts to whatever
        # coordinate frame the run's channels actually live in). Init 10.0: typical
        # coord magnitude ~0.1 -> scaled input ~O(1), a normal-sized MLP input instead of
        # a tenth of one.
        self.coord_scale = nn.Parameter(torch.tensor(10.0))

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
            s = self.coord_proj((coords * self.coord_scale).reshape(B * C, 3)).unsqueeze(1)  # [B*C, 1, D]
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
    weight — unlike the 0.2x-weighted shared stamps in the StampBank, true DeepSeekMoE
    shared Experts aren't down-weighted). Replaces the single dense FFN sub-layer in
    TSABlock. See docs/adr/0008-moe-ffn-for-mesae.md.

    Each expert's inner width is a fraction of the dense FFN's hidden_dim (dim * mlp_ratio)
    so total *active* per-token compute (n_shared + top_k experts firing) stays roughly at
    parity with a single dense FFN of that hidden_dim — standard DeepSeekMoE
    fine-grained-expert sizing. mlp_ratio is the single knob controlling expert width.

    # ponytail: dense routed-expert compute (every routed Expert runs on every token, then
    # masked by the gate — same "compute all, mask by gate" convention FilterRouter/
    # ExpertChannelPool already use in this file), not real sparse dispatch. Fine at this
    # expert count; switch to grouped/sparse dispatch if expert count or throughput ever
    # makes this the bottleneck.
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
        routed_out = torch.stack([e(x_flat) for e in self.routed_experts], dim=1)  # [T, R, D]
        routed_sum = (routed_out * gate_mask.unsqueeze(-1)).sum(dim=1)  # [T, D]

        shared_sum = x_flat.new_zeros(x_flat.shape)
        for e in self.shared_experts:
            shared_sum = shared_sum + e(x_flat)

        out = (routed_sum + shared_sum).reshape(BC, N, D)
        return out, lb_loss

    @torch.no_grad()
    def _record_health(self, gate_mask):
        """Same router-health formulas as MeSAEPretrain.update_head_metrics (entropy of the
        LOAD distribution across routed Experts, entropy of the WITHIN-token gate weights,
        load std) — computed every forward call (cheap, R is small) and stashed on self so
        TSAEncoder.forward can average across all TSABlocks' MoEFFNs into one dashboard
        number, mirroring the SAE Filter router's diagnostic but kept as a separate metric
        (see docs/adr/0008-moe-ffn-for-mesae.md: two distinct MoEs, two distinct health
        readouts)."""
        selected = (gate_mask > 0).float()
        load = selected.mean(dim=0)
        load_p = load / (load.sum() + 1e-8)
        self.last_router_entropy = -(load_p * torch.log(load_p + 1e-10)).sum()
        self.last_router_load_std = load.std()
        # .float() matters here: this runs inside forward(), under autocast during
        # training, so gate_mask is fp16 — 1e-10 underflows to exactly 0.0 in fp16, making
        # log(0+0)=-inf and 0*-inf=NaN for every masked-out (always-present) entry, which
        # _ema_update's NaN-guard then silently skips forever (see MeSAEPretrain.
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
        # (spatial) mixing default off — MeSAE's tokenizer stage trains the SAE on
        # patch-local features only, so the frozen dictionary can't leak already-seen
        # context into masked-stage reconstruction targets (see
        # docs/adr/0003-mesae-two-stage-masked-training.md). Both out_proj-equivalents are
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

    def forward(self, x):
        B, C, N, D = x.shape
        x_flat = x.view(B * C, N, D)
        self.last_branch_max = None

        if self.temporal_active:
            x_norm_t = self.norm_time(x_flat)
            attn_out_t, _ = self.temporal_attn(x_norm_t, x_norm_t, x_norm_t)
            self._watch(attn_out_t)
            attn_out_t = self.norm_time_out(attn_out_t)
            x_flat = x_flat + self.drop_t(self.scale_t * attn_out_t)

        x_space = x_flat.view(B, C, N, D).permute(0, 2, 1, 3).reshape(B * N, C, D)
        if self.spatial_active:
            x_norm = self.norm_space(x_space)
            attn_out, _ = self.spatial_attn(x_norm, x_norm, x_norm)
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


class TSAEncoder(nn.Module):
    def __init__(self, dim, depth=12, num_heads=8, mlp_ratio=4., dropout=0.0,
                 pool_after_blocks=(),
                 n_routed_ffn_experts=4, n_shared_ffn_experts=1, ffn_top_k=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            TSABlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                     n_routed_ffn_experts=n_routed_ffn_experts, n_shared_ffn_experts=n_shared_ffn_experts,
                     ffn_top_k=ffn_top_k)
            for _ in range(depth)
        ])
        # UNet-style temporal down/up: triangular-kernel-filtered pool N in half after each
        # listed block, then (once, after the last block) nearest-repeat upsample + gated
        # skip-add back through the same points in reverse, restoring the original N.
        # Parameter-free pooling (fixed [1,2,1]/4 kernel + repeat), so downstream
        # (SAE/decoder/loss) never sees a shape change. The gated residual add on the
        # way back up always runs.
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
        #
        # A block immediately before a pool point (i in pool_after_blocks) has its raw
        # delta counted twice downstream: once propagated through the pooled/bottleneck
        # path, and again as a direct gated re-add at the very end (see the skip loop
        # below) — the raw delta alone doesn't reflect that second, gate-weighted path.
        # Folded in here by scaling those blocks' recorded delta by sigmoid(gate), so
        # block_norm_i reads as this block's actual surviving contribution, not just
        # what it computed before the gate ever touches it.
        record_norms = not self.training
        if record_norms:
            self.last_block_norms = []
            sorted_pool = sorted(self.pool_after_blocks)
            gate_for_block = dict(zip(sorted_pool, self.skip_gates))
        ffn_lb_loss = x.new_zeros(())
        for i, block in enumerate(self.blocks):
            x_in = x
            x, blk_ffn_lb = block(x)
            ffn_lb_loss = ffn_lb_loss + blk_ffn_lb
            if record_norms:
                with torch.no_grad():
                    delta = (x - x_in).norm(dim=-1).mean()
                    if i in gate_for_block:
                        delta = delta * torch.sigmoid(gate_for_block[i])
                    self.last_block_norms.append(delta.item())
            if i in self.pool_after_blocks:
                skips.append(x)
                x = self._pool(x)

        for skip, gate in zip(reversed(skips), reversed(self.skip_gates)):
            N_pre = skip.shape[2]
            x = x.repeat_interleave(2, dim=2)  # upsample
            x = x[:, :, :N_pre, :]              # trim off any pool-time padding
            x = x + torch.sigmoid(gate) * skip

        # Average each TSABlock's MoEFFN router-health readout (see MoEFFN._record_health)
        # across all blocks into one number per encoder pass — every block runs every
        # forward, so a simple mean is a fair per-batch summary of "how is the FFN router
        # doing across the whole encoder", not just one layer's snapshot.
        with torch.no_grad():
            self.last_ffn_router_entropy = torch.stack([b.ffn.last_router_entropy for b in self.blocks]).mean()
            self.last_ffn_router_load_std = torch.stack([b.ffn.last_router_load_std for b in self.blocks]).mean()
            self.last_ffn_gate_entropy = torch.stack([b.ffn.last_gate_entropy for b in self.blocks]).mean()
            # Worst interior branch magnitude anywhere in the stack (see TSABlock._watch).
            # Max, not mean: one block crossing the float ceiling takes out the whole run,
            # so an average across 12 blocks would bury exactly the signal this exists for.
            watched = [b.last_branch_max for b in self.blocks if b.last_branch_max is not None]
            self.last_branch_max = torch.stack(watched).max() if watched else None

        return x, ffn_lb_loss


# ==========================================
# Decoder & channel pooling
# ==========================================

class StampBank(nn.Module):
    """
    Sparse source dictionary over CHANNEL-GROUPED tokens: input is [G, C, D] where each
    group g is one patch POSITION (G = B*N) carrying all C channels' embeddings for
    that moment. Selection runs once per group (shared by every channel); amplitude is
    read per channel. This is the instantaneous-mixing ICA picture made structural:
    x_c(t) = sum_s A[c, s] * source_s(t) — D_hat_i is source_s's waveform, and the
    [C] vector of per-channel amps for a selected stamp IS that source's mixing
    column (its topomap at that patch time), dense across channels by construction.

    Group selection binds one source to ONE stamp across the whole scalp.

    The bank is n_routed (compete via score + top-k) plus n_shared (always
    included, every group, no top-k). No fixed down-weight on the shared pool's
    recon contribution: a scalar multiplier on amp is not a real regularizer here —
    amp is a free, unconstrained linear gain, so the optimizer just inflates it to
    cancel any fixed scale back out at convergence. Whatever separates shared from
    routed has to come from a real structural difference (always-on vs. gated,
    wider bottleneck) — see shared_hidden_width — not a multiplier gradient descent
    can undo for free.

    Selection is TopK-SAE style, aggregated over channels: per-atom group score =
    mean over VALID channels of amp_i(z_c)^2 (matched-filter energy summed over the
    scalp — an atom strong on a few channels or moderate on many both rank fairly),
    one top-k per group. The coefficient IS the score: amp is trained by recon MSE at
    every channel, so ranking directly off it gives every atom a real, continuously
    updated selection signal. h (both h_routed and h_shared) is the post-rms amp
    magnitude sqrt(a^2+b^2) averaged over channels — diagnostics/viz-ranking only,
    never touches recon.

    No load-balance loss: the routing score IS the reconstruction coefficient, so
    pushing the load distribution toward uniform is pushing reconstruction amplitudes
    toward uniform — unlike MoEFFN, whose gate is a free parameter with no other job,
    where uniformity costs only routing preference. A plain LB term here would be
    another auxiliary loss whose optimum ("every atom contributes equal energy on
    every patch") recon cannot veto. It would also fight legitimate power-law usage: measured
    load entropy on healthy runs is 0.73-0.81 of its maximum (alive 0.54-0.97) —
    deliberately non-uniform, as a content-addressed dictionary should be, since real
    source prevalence is unequal (alpha everywhere, a rare artifact rarely). Collapse
    is instead guarded by fire_ema/dead_threshold/aux_loss below, which are curative
    and content-AWARE (a revived atom is aimed at the residual, i.e. at content
    nothing else covers) where LB would be preventive and content-blind. If prevention
    is ever genuinely needed, the safe shape is a HINGED entropy FLOOR (relu(0.70 -
    H/log(n_routed)), inactive across the healthy band, fires only on a real
    collapse), not a push toward uniform. That fraction is logged as
    stamp_router_entropy_frac.

    phi_i(z_c) = rms_c * (a_i(z_c) * D_hat_i + b_i(z_c) * Hilbert(D_hat_i)): a fixed
    per-atom waveform TEMPLATE D_i (nn.Parameter [patch_len], no z dependence, used
    UNIT-L2-NORMALIZED everywhere — see the D_routed init comment for the amp/norm
    degeneracy this kills) plus its DERIVED Hilbert quadrature partner (never a free
    parameter, see _quadrature), combined by a per-CHANNEL, per-atom gain pair
    (a, b) from the atom's own narrow hidden_i bottleneck — amplitude
    sqrt(a^2+b^2), phase atan2(b, a): the stamp can present its source at any
    arrival phase without shape freedom (see the w_amp init comment) — times that
    channel's raw-input RMS (the LayerNorm stack erases amplitude from z, so the
    gain multiplies it back in explicitly — see forward()).
    Deliberately NOT a generator that can bend its own shape per token (that was the
    prior design: hidden_i @ W_out_i + b_out_i, a full per-atom linear map from the
    bottleneck to [patch_len]) — replaced because the target signal this is meant to
    capture (a shared source, e.g. line noise, arriving at every channel as the SAME
    waveform at a channel-specific amplitude/polarity, near-zero phase lag) is
    structurally amplitude-varying, not shape-varying. Forcing shape to be a pure
    parameter and amplitude to be the only z-dependent knob makes "same waveform,
    different amplitude across channels" a structural guarantee instead of something
    training has to discover on its own, and is provably phase-safe: scalar-multiplying
    a real time-domain vector scales every frequency bin's magnitude by the same
    factor and leaves phase untouched (amp<0 is a clean 180-degree flip, not
    distortion) — unlike scaling a waveform's real/imag FFT components independently,
    which does distort phase (that failure mode doesn't apply here since there's no
    real/imag split anywhere in this module, only a real time-domain vector, see
    dense_probe's docstring for the earlier scalar-weighting attempts that got
    entangled with the ROUTING scalar h instead of using a free one).

    Cost trade against the old per-atom W_out design: loses the ability for an atom to
    warp its own shape per token (e.g. a genuine conduction-delay phase difference
    across channels, or an amplitude-dependent shape change like a spike broadening as
    it grows — see docs/adr/0009's discussion of this exact tradeoff). Also a real
    fingerprint simplification: D_i now IS each atom's shape, unconditionally — no more
    fabricated-probe fingerprint() vs real-data dense_probe() split to work around a
    generator whose shape depended on its input (see both methods below).

    A free [patch_len]-length D_i's implicit frequency content is bound to the
    patch_len-length FFT grid (Df = fs/patch_len). Parametric oscillator atoms once
    carved out shared slots to escape that grid for line noise; withdrawn after
    measurement — see docs/adr/0010-oscillator-atoms-withdrawn.md.
    """
    def __init__(self, dim, patch_len, n_routed_stamps=796, n_shared_stamps=4,
                 top_k=32, hidden_width=8, shared_hidden_width=16,
                 dead_threshold_frac=0.1, aux_k_cap_frac=0.04, ema_decay=0.999):
        super().__init__()
        # Pool sizes are declared separately, not a total minus a slice: n_stamps is
        # the derived sum. Every internal use below wants the total, so it stays.
        self.n_routed = n_routed_stamps
        self.n_shared = n_shared_stamps
        self.n_stamps = n_routed_stamps + n_shared_stamps
        self.top_k = min(top_k, self.n_routed)
        # Normalizes z before it's used for anything (scoring, the bottleneck's
        # generator input, z_h) — z inherits whatever scale the encoder currently
        # drifts to (documented block_norm growth across blocks/epochs elsewhere in this
        # codebase).
        self.input_norm = nn.LayerNorm(dim)
        self.dim = dim
        # Bottleneck width over the D-dim z input. Shared stamps get a wider
        # bottleneck than routed (16 vs 8 by default): they're always-on across every
        # patch/dataset (never gated out), so they need more room to represent structure
        # common across all data types rather than specializing narrowly like a routed
        # stamp can afford to.
        self.hidden_width = hidden_width
        self.shared_hidden_width = shared_hidden_width

        # No selection scorer params — selection is |amp_i(z)| directly (see class
        # docstring and forward()): amp is trained by recon MSE, so scoring off the
        # atom's own bottleneck output IS what the atom would contribute.

        # phi: bottleneck generator, per-atom W_down/b_down (down-project + GELU) decoding
        # through a per-ATOM W_out/b_out straight to [patch_len] — every atom gets its own
        # full down+up map now, no group-shared decode table. Routed and shared use
        # separate W_down/b_down/W_out/b_out tables (different hidden widths).
        self.W_down_routed = nn.Parameter(torch.empty(self.n_routed, dim, hidden_width))
        self.b_down_routed = nn.Parameter(torch.zeros(self.n_routed, hidden_width))
        self.W_down_shared = nn.Parameter(torch.empty(self.n_shared, dim, shared_hidden_width))
        self.b_down_shared = nn.Parameter(torch.zeros(self.n_shared, shared_hidden_width))
        nn.init.kaiming_uniform_(self.W_down_routed, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_down_shared, a=math.sqrt(5))

        # amp_i(z): QUADRATURE PAIR of gains (a, b) read off the atom's own hidden_i
        # bottleneck — contribution = a*D_hat + b*Hilbert(D_hat), so the pair encodes
        # amplitude A=sqrt(a^2+b^2) and phase phi=atan2(b, a) of the template with the
        # generator staying fully linear (phase is the ANGLE of a learned 2-vector,
        # never a raw scalar rotated through trig — no sin/cos optimization basins).
        # Because the partner is the Hilbert transform of the SAME template (derived,
        # not free — see _quadrature), (a, b) can only re-phase and scale the shape,
        # never morph it: that tie is what separates this from the rejected
        # "independently scale real/imag" design, which warps the waveform. Doubles as
        # the selection score via a^2+b^2 (phase-invariant matched-filter energy — an
        # atom now matches its source at ANY arrival phase, killing the need for
        # phase-shifted template copies in the pool). Free/unbounded/signed, no clamp.
        self.w_amp_routed = nn.Parameter(torch.empty(self.n_routed, hidden_width, 2))
        self.b_amp_routed = nn.Parameter(torch.zeros(self.n_routed, 2))
        self.w_amp_shared = nn.Parameter(torch.empty(self.n_shared, shared_hidden_width, 2))
        self.b_amp_shared = nn.Parameter(torch.zeros(self.n_shared, 2))
        nn.init.kaiming_uniform_(self.w_amp_routed, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_amp_shared, a=math.sqrt(5))

        # D_i: the atom's own waveform template, a plain parameter with NO z dependence.
        # Used UNIT-L2-NORMALIZED at every consumption site (F.normalize in
        # _generate_routed/decode_selected/dense_probe/fingerprint), never raw: with a
        # free-norm D, amp*D has a scale degeneracy — the model can shrink amp and grow
        # ||D|| with recon unchanged, which (a) games sparsity_loss's L1-on-amp down to
        # nothing without any real sparsification (classic sparse-coding pitfall, fixed
        # the standard way: unit-norm dictionary atoms), and (b) makes amp values
        # incomparable across atoms — with unit D, amp is the one true coefficient
        # (actual per-channel source amplitude, the thing a topomap of one stamp across
        # channels is supposed to read). The raw parameter keeps whatever norm it drifts
        # to; only its direction ever matters.
        # Normal-init at a modest std (not kaiming, there's no fan-in/fan-out here: this
        # is a direct [patch_len] output vector, not a weight matrix).
        self.D_routed = nn.Parameter(torch.randn(self.n_routed, patch_len) * 0.02)
        self.D_shared = nn.Parameter(torch.randn(self.n_shared, patch_len) * 0.02)

        self.dead_threshold = dead_threshold_frac * (self.top_k / self.n_routed)
        self.aux_k_cap = max(1, int(aux_k_cap_frac * self.n_routed))
        self.ema_decay = ema_decay
        self.register_buffer('fire_ema', torch.zeros(self.n_routed))

    def _amp_dense(self, z):
        """z: [G, C, D] ALREADY input_norm'd channel-grouped tokens -> per-channel,
        per-atom QUADRATURE gain pairs (a, b), dense over both pools:
        (amp_routed [G, C, n_routed, 2], amp_shared [G, C, n_shared, 2]) — NO rms applied
        (callers multiply it in where the real contribution scale is needed; the group
        selection score deliberately skips it, see forward()). Same "compute-all"
        convention MoEFFN uses (see its ponytail note): with amp needed dense for group
        scoring anyway, there is no sparse decode path left to save — the old
        per-selected-atom gather einsums (_decode_atoms/_generate_routed) collapsed
        into this one dense computation plus a cheap gather in forward().

        GELU sits between W_down and w_amp: without it, hidden is affine-in-affine (two
        linear maps back to back), which collapses algebraically into one linear map
        z -> amp of rank <= min(hidden_width, 2) — since amp is only 2-dim, hidden_width
        past 2 bought zero extra capacity, just wasted params. The GELU makes hidden_width
        a real nonlinear bottleneck (an actual per-atom small MLP) instead of a disguised
        linear readout."""
        hidden_r = F.gelu(torch.einsum('gcd,hdk->gchk', z, self.W_down_routed) + self.b_down_routed)
        amp_r = torch.einsum('gchk,hkp->gchp', hidden_r, self.w_amp_routed) + self.b_amp_routed
        hidden_s = F.gelu(torch.einsum('gcd,hdk->gchk', z, self.W_down_shared) + self.b_down_shared)
        amp_s = torch.einsum('gchk,hkp->gchp', hidden_s, self.w_amp_shared) + self.b_amp_shared
        return amp_r, amp_s

    @staticmethod
    def _quadrature(D):
        """D: [M, L] unit templates -> each row's Hilbert quadrature partner [M, L],
        unit-normalized. Derived (rFFT, rotate every positive-frequency bin by -90
        degrees, zero DC/Nyquist which have no quadrature, irFFT), NEVER a free
        parameter — <D, H(D)> = 0 exactly, so (a*D_hat + b*H_hat) spans amplitude
        A=sqrt(a^2+b^2) and constant phase phi=atan2(b,a) of the template's analytic
        signal WITHOUT any shape freedom (see the w_amp init comment: the tie is what
        makes two coefficients mean phase, not morphing). Re-normalized since zeroing
        DC/Nyquist drops whatever energy the template had there; an (almost-)pure-DC
        template's partner is degenerate — its b head just learns ~0."""
        Fd = torch.fft.rfft(D.float(), dim=-1) * (-1j)
        Fd[..., 0] = 0
        if D.shape[-1] % 2 == 0:
            Fd[..., -1] = 0
        H = torch.fft.irfft(Fd, n=D.shape[-1], dim=-1)
        return F.normalize(H, dim=-1).to(D.dtype)

    def _template_tables(self):
        """(D_all, H_all): unit templates in GLOBAL layout (routed then shared) and
        their quadrature partners, each [n_stamps, patch_len]."""
        D_all = F.normalize(torch.cat([self.D_routed, self.D_shared], dim=0), dim=-1)
        return D_all, self._quadrature(D_all)

    def decode_selected(self, idx, amp):
        """idx: [G, top_k+n_shared] GLOBAL indices (routed then shared, forward()'s
        layout), amp: [G, C, top_k+n_shared, 2] per-channel quadrature gain pairs WITH
        rms already in (forward()'s out.amp) -> contribution [G, C, top_k+n_shared, patch_len] = a*D_hat + b*H_hat
        per slot, each slot's own raw decoded output per channel (unsummed — viz reads
        this to show per-stamp per-channel content; a stamp's [C] magnitude column
        sqrt(a^2+b^2) at one slot is its phase-invariant topomap at that patch time).
        Pure re-expansion of forward()'s already-computed quantities — no model
        re-evaluation, so callers can't accidentally decode with different
        selection/scale than training produced."""
        D_all, H_all = self._template_tables()
        D_sel, H_sel = D_all[idx], H_all[idx]
        return (amp[..., 0].unsqueeze(-1) * D_sel.unsqueeze(1)
                + amp[..., 1].unsqueeze(-1) * H_sel.unsqueeze(1))

    # Redundant/degenerate atoms are handled two ways: mp_loss (below) denies them
    # reward for re-explaining a higher-ranked atom's content, and interchangeable
    # atoms that still slip through die naturally (amp shrinks at zero recon cost ->
    # group score fades -> dead -> aux rescue re-aims them at uncovered residual).
    # k_eff and stamp_router_entropy_frac stay logged as the collapse tripwire.

    @torch.no_grad()
    def fingerprint(self):
        """Every stamp's raw waveform template, routed then shared. Free-vector atoms
        have NO z dependence at all — D_i IS the shape, unconditionally. Shown at one
        arbitrary reference phase (t=0, see _template_tables), since only their
        frequency is a stable identity; arrival phase isn't. amp_i(z) never touches
        shape for either kind, only overall scale/sign — see class docstring — so this
        alone is the complete, correct answer to "what does this atom look like".
        Returned unit-normalized, matching what the decode path actually uses. Dense
        over all n_stamps. Returns [n_stamps, patch_len]."""
        return self._template_tables()[0]

    @torch.no_grad()
    def dense_amp(self, z, rms=None):
        """z: [G, C, D] channel-grouped embeddings (same input StampBank.forward takes)
        -> amp [G, C, n_stamps, 2], dense over EVERY atom (routed+shared, no top-k),
        input_norm'd and rms-scaled the same way forward() is. Every atom's own
        matched-filter response to real content, whether or not it would win the top-k
        race — used where a stable, always-populated per-stamp axis matters more than
        reconstruction sparsity (MeSAEFinetune.encode_post_stamp_expert's per-stamp
        channel pool). Shared by dense_probe below, which decodes this further into
        waveform space."""
        z = self.input_norm(z)
        amp_r, amp_s = self._amp_dense(z)  # [G, C, n_routed, 2], [G, C, n_shared, 2]
        amp = torch.cat([amp_r, amp_s], dim=2)  # [G, C, n_stamps, 2]
        if rms is not None:
            amp = amp * rms.unsqueeze(-1)
        return amp

    def dense_probe(self, z, rms=None):
        """The real per-channel, per-atom CONTRIBUTION each stamp would produce if it
        had fired — amp_i(z_c) * rms_c * D_hat_i, dense over all n_stamps
        (diagnostic-only, not the training path, which only decodes the selected
        top_k+n_shared). Distinct from fingerprint() (the atom's own shape, no z at
        all): this shows the SCALED contribution a real token would get, fingerprint()
        the unscaled template underneath it.
        z: [G, C, D] channel-grouped embeddings (same input StampBank.forward takes),
        rms: [G, C, 1] or None -> contribution [G, C, n_stamps, patch_len].
        """
        amp = self.dense_amp(z, rms=rms)  # [G, C, n_stamps, 2]
        D_all, H_all = self._template_tables()  # each [n_stamps, L]
        return (amp[..., 0].unsqueeze(-1) * D_all.view(1, 1, self.n_stamps, -1)
                + amp[..., 1].unsqueeze(-1) * H_all.view(1, 1, self.n_stamps, -1))

    def forward(self, z, x_target=None, rms=None, valid_channels=None):
        """
        z: [G, C, D] channel-grouped token embeddings (G = B*N patch positions, all C
        channels of one patch time per group — see class docstring), x_target:
        [G, C, patch_len] the real patch content (only needed for the dead-atom aux
        rescue, training only), rms: [G, C, 1] per-channel raw-input RMS or None —
        multiplied into every amp; callers running the real pipeline should always
        pass it.
        valid_channels: [G, C] bool, True = real (not zero-padded) channel,
        or None — used ONLY for the group selection score (a zero-padded channel's amp
        is encoder-bias noise that shouldn't vote on which sources this patch
        contains); padded channels still decode/reconstruct like any other, and the
        loss-side exclusion stays get_loss's job.

        Returns recon [G, C, patch_len], idx [G, top_k+n_shared] (GLOBAL stamp ids,
        routed then shared — ONE selection per patch position, shared by all C
        channels), amp [G, C, top_k+n_shared, 2] (per-channel quadrature gain pairs
        (a, b), rms included — a slot's [C] magnitude column sqrt(a^2+b^2) is that
        stamp's phase-invariant mixing/topomap vector at this patch time, atan2(b, a)
        its per-channel phase), h [G, top_k+n_shared] (that slot's post-rms amp
        magnitude averaged over channels — real reconstruction-energy importance,
        diagnostics/viz-ranking only, still never touches recon),
        dense_routed [G, n_routed] (zeros at unselected — the diagnostic object
        MeSAETrainer/MeSAECodebookChecker read for router-health/usage
        panels, at patch-position granularity), aux_loss, k_eff (diagnostic only).

        No load-balance loss — see class docstring. Dead-atom collapse is handled by
        fire_ema/dead_threshold/aux_loss below ("fired" now means "selected for a
        patch position", not "for a (channel, patch) token").
        """
        G, C, D = z.shape
        z = self.input_norm(z)  # stabilize scale before scoring/generation, see __init__

        amp_r_dense, amp_s_dense = self._amp_dense(z)  # [G, C, n_routed, 2], [G, C, n_shared, 2]

        # Group selection score: mean over VALID channels of a^2+b^2 (the pair's energy
        # — PHASE-INVARIANT matched filtering: an atom matches its source at any
        # arrival phase, see the w_amp init comment) — matched-filter
        # energy of each atom totaled over the scalp (see class docstring). rms
        # deliberately NOT applied: unlike the per-token case (where it was a single
        # scalar and ranking-invariant), per-channel rms WOULD reweight the ranking
        # toward loud channels — but amp already carries each channel's learned gain;
        # double-weighting by raw loudness would let one hot channel drown out a
        # source spread moderately over many, exactly the topomap-binarizing failure
        # group selection exists to fix.
        a2 = amp_r_dense.pow(2).sum(dim=-1)                         # [G, C, n_routed] — a^2+b^2
        if valid_channels is not None:
            vc = valid_channels.unsqueeze(-1).to(a2.dtype)          # [G, C, 1]
            group_score = (a2 * vc).sum(dim=1) / vc.sum(dim=1).clamp(min=1.0)  # [G, n_routed]
        else:
            group_score = a2.mean(dim=1)                            # [G, n_routed]

        _, topk_idx = group_score.topk(self.top_k, dim=-1)   # [G, top_k]

        shared_idx = torch.arange(self.n_routed, self.n_stamps, device=z.device)
        shared_idx = shared_idx.unsqueeze(0).expand(G, -1)               # [G, n_shared]

        idx = torch.cat([topk_idx, shared_idx], dim=-1)   # [G, top_k+n_shared]

        # Per-channel gains for the group's selected set: every channel decodes the
        # SAME stamps with its OWN amp — the [C] column per slot is the mixing vector.
        amp_sel_r = amp_r_dense.gather(
            2, topk_idx.view(G, 1, self.top_k, 1).expand(G, C, self.top_k, 2))
        amp = torch.cat([amp_sel_r, amp_s_dense], dim=2)  # [G, C, top_k+n_shared, 2]
        if rms is not None:
            amp = amp * rms.unsqueeze(-1)  # [G, C, 1, 1] broadcast — restores raw amplitude

        # h = post-rms amp magnitude sqrt(a^2+b^2) averaged over (valid) channels —
        # the real reconstruction-energy importance of each selected slot, replacing
        # the old within-group softmax (see class docstring for why: that value never
        # fed recon/loss/dead-atom detection, and gave every shared stamp the same
        # flat constant regardless of its real contribution).
        slot_energy = amp.pow(2).sum(dim=-1)                             # [G, C, K]
        if valid_channels is not None:
            vc = valid_channels.unsqueeze(-1).to(slot_energy.dtype)      # [G, C, 1]
            slot_mag = (slot_energy * vc).sum(dim=1) / vc.sum(dim=1).clamp(min=1.0)
        else:
            slot_mag = slot_energy.mean(dim=1)                           # [G, K]
        h = slot_mag.clamp(min=0).sqrt()                                 # [G, K]
        dense_routed = torch.zeros_like(group_score).scatter_(-1, topk_idx, h[:, :self.top_k])  # [G, n_routed]

        D_all, H_all = self._template_tables()
        D_sel, H_sel = D_all[idx], H_all[idx]  # each [G, top_k+n_shared, patch_len]
        # a*D_hat + b*Hilbert(D_hat) summed over slots — no [G,C,K,L] materialized
        recon = (torch.einsum('gck,gkl->gcl', amp[..., 0], D_sel)
                 + torch.einsum('gck,gkl->gcl', amp[..., 1], H_sel))

        # Matching-Pursuit-style residual loss (see class docstring's Sequential
        # residual fit section below) — routed slots ONLY (shared stamps are an
        # always-on baseline, not competing for content, so residual-ordering them
        # doesn't apply). One-shot top-k-by-energy selection has NO mechanism against
        # two correlated/near-duplicate atoms co-scoring high on the SAME target and
        # getting selected together every time (unlike Matching Pursuit/OMP, which
        # explicitly re-scores against the RESIDUAL after each pick, so a near-
        # duplicate of an already-picked atom scores ~0 on what's left). This term
        # doesn't change selection or recon (both stay exactly as above) — it only
        # reshapes each routed slot's TRAINING TARGET: rank the top_k routed slots
        # by h (descending, real reconstruction-energy order, not the fixed idx
        # order), then grade slot rank m against x_target MINUS what ranks 0..m-1
        # already explained (detached, so gradient only ever pushes a slot toward
        # what's genuinely still unexplained, never perturbs the residual itself).
        # A true duplicate of a higher-ranked atom sees a near-zero residual and gets
        # no reward for repeating it — sidesteps the narrowband-collapse risk a blunt
        # pairwise spectral-overlap penalty would have (see session discussion): nothing
        # here penalizes two atoms sharing content, only rewards atoms for covering
        # content NO ONE ELSE at a higher rank already covered.
        mp_loss = amp.new_zeros(())
        if x_target is not None:
            L = D_sel.shape[-1]
            contrib_all = (amp[..., 0].unsqueeze(-1) * D_sel.unsqueeze(1)
                           + amp[..., 1].unsqueeze(-1) * H_sel.unsqueeze(1))   # [G, C, K, L]
            order_routed = h[:, :self.top_k].argsort(dim=-1, descending=True)  # [G, top_k]
            # Shared slots pinned ahead of routed, always. They are always-on, so every
            # patch's reconstruction contains them whether or not anything asked for them;
            # grading a routed atom against a residual that still holds that baseline
            # rewards it for re-explaining content already covered. Not a knob — the
            # alternative is simply wrong. Measured: real 50Hz shared-pool share 0.336 ->
            # 0.163, single routed owner 0.414 -> 0.661, mse_patch 0.0295 -> 0.0232.
            shared_cols = torch.arange(self.top_k, idx.shape[1],
                                       device=h.device).unsqueeze(0).expand(G, -1)
            order = torch.cat([shared_cols, order_routed], dim=1)
            n_rank = order.shape[1]
            order_c = order.unsqueeze(1).unsqueeze(-1).expand(G, C, n_rank, L)
            contrib_ranked = contrib_all.gather(2, order_c)             # rank 0 = first claim

            vmask = None
            if valid_channels is not None:
                vmask = valid_channels.unsqueeze(-1).to(contrib_ranked.dtype)  # [G, C, 1]

            # Vectorized over rank (was a Python for-loop over n_rank, every training
            # step): rank m's residual is x_target minus every STRICTLY-higher-ranked
            # slot's (detached) contribution, i.e. an EXCLUSIVE cumsum over the rank axis
            # of the detached contributions. cumsum of detached == detach of cumsum, so
            # this is exactly the sequential loop's math, just computed for every rank at
            # once instead of one at a time (verified numerically equal, incl. gradients,
            # against the loop form before this rewrite).
            cum_excl = contrib_ranked.detach().cumsum(dim=2) - contrib_ranked.detach()  # [G,C,n_rank,L]
            resid_all = x_target.unsqueeze(2) - cum_excl
            diff2_all = (contrib_ranked - resid_all).pow(2)                 # [G, C, n_rank, L]
            if vmask is not None:
                per_rank = (diff2_all * vmask.unsqueeze(2)).sum(dim=(0, 1, 3)) \
                    / vmask.sum().clamp(min=1.0) / L
            else:
                per_rank = diff2_all.mean(dim=(0, 1, 3))
            mp_loss = per_rank.mean()

        # Magnitude sqrt(a^2+b^2) per selected routed slot — the phase-invariant
        # amplitude, what sparsity/k_eff should see (penalize/count loudness, never
        # phase).
        amp_routed_sel = amp[:, :, :self.top_k, :]
        amp_mag_routed = amp_routed_sel.pow(2).sum(dim=-1).clamp(min=1e-12).sqrt()

        # Scale-invariant parsimony diagnostic: effective atom count per token,
        # k_eff = (sum|a|)^2 / sum(a^2) — 1.0 when one atom carries everything,
        # top_k when all selected atoms contribute equally. Unlike the sparsity loss
        # value (whose optimum depends on how many real sources a patch contains),
        # this reads directly as "how many atoms genuinely carry the reconstruction"
        # with no data-loudness floor. Diagnostic only (no_grad), token-averaged over
        # valid channels.
        with torch.no_grad():
            a = amp_mag_routed
            keff = a.sum(dim=-1).pow(2) / (a.pow(2).sum(dim=-1) + 1e-8)  # [G, C]
            if valid_channels is not None and valid_channels.any():
                k_eff = keff[valid_channels].mean()
            else:
                k_eff = keff.mean()

        aux_loss = recon.new_zeros(())
        if self.training:
            with torch.no_grad():
                fired = dense_routed.detach().gt(0).float().mean(dim=0)
                self.fire_ema.mul_(self.ema_decay).add_(fired, alpha=1 - self.ema_decay)
                dead_mask = self.fire_ema < self.dead_threshold  # [n_routed]

            if dead_mask.any() and x_target is not None:
                dead_score = group_score.masked_fill(~dead_mask.unsqueeze(0), float('-inf'))
                aux_k = min(self.aux_k_cap, int(dead_mask.sum().item()))
                aux_val, aux_idx = dead_score.topk(aux_k, dim=-1)  # [G, aux_k]
                # rescue only ever draws from the routed pool (dead atoms are a routed-only
                # concept, shared stamps are always "alive" by construction). Training a
                # rescued atom's amp toward the residual raises exactly the quantity that
                # gets it selected (group_score is amp^2-based) — the rescue revives atoms
                # for real, per channel, at group granularity.
                amp_aux = amp_r_dense.gather(
                    2, aux_idx.view(G, 1, aux_k, 1).expand(G, C, aux_k, 2))
                if rms is not None:
                    amp_aux = amp_aux * rms.unsqueeze(-1)
                D_r_hat = F.normalize(self.D_routed, dim=-1)
                D_aux = D_r_hat[aux_idx]                       # [G, aux_k, patch_len]
                H_aux = self._quadrature(D_r_hat)[aux_idx]
                recon_aux = (torch.einsum('gck,gkl->gcl', amp_aux[..., 0], D_aux)
                             + torch.einsum('gck,gkl->gcl', amp_aux[..., 1], H_aux))
                residual = (x_target - recon).detach()
                aux_loss = F.mse_loss(recon_aux, residual) / (residual.pow(2).mean() + 1e-8)

        return SimpleNamespace(
            recon=recon, idx=idx, amp=amp, h=h, dense_routed=dense_routed,
            aux_loss=aux_loss, k_eff=k_eff, mp_loss=mp_loss,
        )


class PerChannelHeadAttn(nn.Module):
    """
    Two-stage attention pooling — identical to model/MeFSQ/MeFSQ_modules.py's
    PerChannelHeadAttn (duplicated here rather than cross-imported, same convention as
    ExpertChannelPool/MultiHeadDecoder above: each model package stays self-contained).
    Fully backbone-agnostic — only needs z_per_head [B, N, H, d] at forward time, so it
    works unchanged whether H indexes MeFSQ Experts or MeSAE stamps.

    The channel dim is collapsed by the backbone itself (each Filter's own
    channel-attention View, see MeSAEPretrain.encode_post_sae_expert) before this head
    ever sees the signal, so there's no channel stage here to re-pool (see
    docs/agents/ / CONTEXT.md finetune val-chance bug for the overparameterization a
    channel-concat classifier caused).

    Stage 1 (temporal): a plain linear scorer over N patches, softmax-normalized.
    Stage 2 (unit): a plain linear scorer over the H units, softmax-normalized, pooling to
    a single [B, d] vector fed into cls. Linear(d,1) at each stage: with a fixed
    (non-content-derived) query, a learnable-query dot-product scorer is just a
    composition of two linear maps with no nonlinearity between them, so it adds no
    expressiveness over a single Linear(d,1) — same capacity, fewer params.
    """
    def __init__(self, head_dim, num_classes, dropout=0.1):
        super().__init__()
        # Un-normalized-scale input (raw decoder-output EEG amplitude) starves the scorer
        # and cls of usable gradient — normalize here so the pooler works regardless of
        # which encode_* the caller feeds it.
        self.input_norm = nn.LayerNorm(head_dim)
        self.score_n = nn.Linear(head_dim, 1)
        self.score_h = nn.Linear(head_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(head_dim, num_classes)
        # cls reads z_h's raw, un-normalized amplitude (see input_norm comment above) — that
        # scale is unbounded by design, so default init gives large initial logits and an
        # elevated first-epoch loss average that has nothing to do with the LR/schedule.
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
        # A gated-off routed Filter (encode_post_sae_expert already zeroes its contribution
        # per patch, see that docstring) has z_per_head == 0 at every (n, d) for this batch
        # item — LayerNorm can't tell "genuinely zero" from "small real signal", so
        # score_h's bias alone would still hand it a non-trivial softmax share in stage 2
        # (visible as spurious attention/importance for a Filter that contributed nothing).
        # Mask those out here instead of relying on the pooled value being zero to save them.
        alive_h = z_per_head.abs().sum(dim=(1, 3)) > 0                          # [B, H]

        z_key = self.input_norm(z_per_head)

        # ---- stage 1: attention pool over N (patches), per unit ----
        logits_n = self.score_n(z_key).squeeze(-1)                              # [B, N, H]
        if pad_mask is not None:
            logits_n = logits_n.masked_fill(~pad_mask.unsqueeze(-1), float('-inf'))
        attn_n = torch.softmax(logits_n, dim=1)                                 # softmax over N, per unit
        attn_n = torch.nan_to_num(attn_n)                                       # all-invalid batch item -> nan; zero it
        z_h = torch.einsum('bnhd,bnh->bhd', z_per_head, attn_n)                 # [B, H, d]
        z_h_key = torch.einsum('bnhd,bnh->bhd', z_key, attn_n)                  # [B, H, d] — for stage 2 logits only

        # ---- stage 2: attention pool over H (units) ----
        logits_h = self.score_h(z_h_key).squeeze(-1)                            # [B, H]
        logits_h = logits_h.masked_fill(~alive_h, float('-inf'))
        attn_h = torch.softmax(logits_h, dim=1)                                 # [B, H]
        attn_h = torch.nan_to_num(attn_h)                                       # all-dead batch item -> nan; zero it
        pooled = (z_h * attn_h.unsqueeze(-1)).sum(dim=1)                        # [B, d]

        attn_n = attn_n.permute(0, 2, 1)                                        # [B, H, N] for interpretability
        return self.cls(self.drop(pooled)), attn_h, attn_n
