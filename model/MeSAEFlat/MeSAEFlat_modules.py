import math
from types import SimpleNamespace

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
        """Same router-health formulas as MeSAEFlatPretrain.update_head_metrics (entropy of the
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
        # _ema_update's NaN-guard then silently skips forever (see MeSAEFlatPretrain.
        # update_head_metrics's gate_routed.detach().float().clamp(...) for the same fix
        # applied to the SAE Filter router's equivalent metric).
        gm = gate_mask.float().clamp(min=0)
        self.last_gate_entropy = -(gm * torch.log(gm + 1e-10)).sum(dim=-1).mean()


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
            # Incoming residual-stream norm BEFORE each block's own norm_out — separates
            # "this block genuinely rewrote a lot" from "norm_out yanked an already-drifted
            # x back to unit scale, showing up as a large delta regardless of this block's
            # own contribution" (see TSABlock.norm_out's docstring on compounding growth).
            self.last_block_input_norms = []
        ffn_lb_loss = x.new_zeros(())
        for i, block in enumerate(self.blocks):
            x_in = x
            if record_norms:
                with torch.no_grad():
                    self.last_block_input_norms.append(x_in.norm(dim=-1).mean().item())
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

        # Average each TSABlock's MoEFFN router-health readout (see MoEFFN._record_health)
        # across all blocks into one number per encoder pass — every block runs every
        # forward, so a simple mean is a fair per-batch summary of "how is the FFN router
        # doing across the whole encoder", not just one layer's snapshot.
        with torch.no_grad():
            self.last_ffn_router_entropy = torch.stack([b.ffn.last_router_entropy for b in self.blocks]).mean()
            self.last_ffn_router_load_std = torch.stack([b.ffn.last_router_load_std for b in self.blocks]).mean()
            self.last_ffn_gate_entropy = torch.stack([b.ffn.last_gate_entropy for b in self.blocks]).mean()

        return x, ffn_lb_loss


# ==========================================
# Decoder & channel pooling
# ==========================================

class StampBank(nn.Module):
    """
    Per-token sparse dictionary over flat (channel, patch) tokens — no cross-channel
    pooling. Each caller-supplied row of `z` is already one (channel, patch)'s own
    D-dim embedding; every atom reads that SAME row directly (no more per-atom
    channel-attention view — see the retired `u`/`_pool` in prior revisions of this
    class, docs/adr/0009-spatiotemporal-stamp-dictionary-for-mesae.md). A "stamp" is
    now just phi_i: R^D -> R^patch_len, a small generator with no spatial component.

    n_stamps splits into n_routed (compete via score + top-k, `docs/adr/0007`'s
    routed/shared split carried over from the Filter level) and n_shared (always
    included, fixed constant `shared_weight`).

    Selection is a plain per-atom linear scorer: score_i = z . w_score_i + b_score_i,
    top-k over routed atoms, softmax over the selected top-k as h_routed (a
    within-token selection confidence, not a self-reconstruction fit — see
    stamp_router_confidence in MeSAEFlatPretrain.get_metrics). Kept as an nn.Parameter
    per-atom weight matrix (not a plain nn.Linear) purely for parity with the
    prior per-atom-view convention; since every atom now shares the same input z,
    this is algebraically a single Linear(dim, n_routed) — `w_score`/`b_score` are
    just its weight/bias laid out per-row. No load-balance loss: unlike MoEFFN
    (compute-all-then-mask, needs traffic spread for compute balance), this module
    dispatches sparsely (gather by idx) — no compute-balance problem, and forcing
    uniform routing would fight the legitimate power-law usage a content-addressed
    dictionary should have. Collapse is instead guarded by fire_ema/dead_threshold/
    aux_loss below.

    phi_i(z) = amp_i(z) * D_i: a fixed per-atom waveform TEMPLATE D_i (nn.Parameter
    [patch_len], no z dependence at all) scaled by a per-token, per-atom scalar gain
    amp_i(z) (from the same narrow hidden_i bottleneck w_score already reads).
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
    """
    def __init__(self, dim, patch_len, n_stamps=800, n_shared_stamps=4,
                 top_k=32, hidden_width=8, shared_hidden_width=16, shared_weight=0.2,
                 dead_threshold_frac=0.1, aux_k_cap_frac=0.04, ema_decay=0.999):
        super().__init__()
        self.n_stamps = n_stamps
        self.n_shared = n_shared_stamps
        self.n_routed = n_stamps - n_shared_stamps
        self.top_k = min(top_k, self.n_routed)
        self.shared_weight = shared_weight
        # Normalizes z before it's used for anything (scoring, the bottleneck's
        # generator input, z_h) — z inherits whatever scale the encoder currently
        # drifts to (documented block_norm growth across blocks/epochs elsewhere in this
        # codebase). Same role TopKSAE.input_norm played for its own encoder.
        self.input_norm = nn.LayerNorm(dim)
        self.dim = dim
        # Bottleneck width over the D-dim z input. Shared stamps get a wider
        # bottleneck than routed (16 vs 8 by default): they're always-on across every
        # patch/dataset (never gated out), so they need more room to represent structure
        # common across all data types rather than specializing narrowly like a routed
        # stamp can afford to.
        self.hidden_width = hidden_width
        self.shared_hidden_width = shared_hidden_width

        # Per-atom selection scorer, routed atoms only (shared atoms are never
        # scored/gated, see class docstring) — score_i = hidden_i . w_score_i + b_score_i,
        # where hidden_i = z @ W_down_i + b_down_i is the SAME low-dim (hidden_width, e.g.
        # 8) bottleneck atom i's own generator already uses (see forward()'s dense
        # hidden_score computation and _decode_atoms). NOT a full-D w_score (was
        # torch.empty(n_routed, dim)): with n_routed atoms competing in a D-dim space (D
        # >> hidden_width), random-init w_score directions overlap heavily and whichever
        # happens to align best with the "typical" token wins everywhere, permanently —
        # observed as dead_feature_rate locking to ~1 - top_k/n_routed by epoch 1 and
        # never moving for an entire 50-epoch run. Scoring off each atom's own narrow
        # hidden_width-dim view instead forces atoms to compete in their own personal
        # subspace rather than the shared full-D one, and ties "how this atom is scored"
        # to "what this atom would actually decode" (self-relevance).
        self.w_score = nn.Parameter(torch.empty(self.n_routed, hidden_width))
        self.b_score = nn.Parameter(torch.zeros(self.n_routed))
        nn.init.kaiming_uniform_(self.w_score, a=math.sqrt(5))

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

        # amp_i(z): scalar gain read off the same hidden_i bottleneck w_score uses,
        # one more Linear(hidden_width, 1)-equivalent per atom. Free/unbounded/signed —
        # no softplus/sigmoid clamp: a genuinely loud channel (e.g. right on top of a
        # line-noise source) needs to be able to grow past 1.0, and a sign flip is a
        # legitimate 180-degree phase flip, not something to forbid.
        self.w_amp_routed = nn.Parameter(torch.empty(self.n_routed, hidden_width))
        self.b_amp_routed = nn.Parameter(torch.zeros(self.n_routed))
        self.w_amp_shared = nn.Parameter(torch.empty(self.n_shared, shared_hidden_width))
        self.b_amp_shared = nn.Parameter(torch.zeros(self.n_shared))
        nn.init.kaiming_uniform_(self.w_amp_routed, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_amp_shared, a=math.sqrt(5))

        # D_i: the atom's own waveform template, a plain parameter with NO z dependence
        # — phi_i(z) = amp_i(z) * D_i, see class docstring. Normal-init at a modest std
        # (not kaiming, there's no fan-in/fan-out here: this is a direct [patch_len]
        # output vector, not a weight matrix multiplying some input) so early-training
        # contributions start comparable in scale to a typical normalized patch.
        self.D_routed = nn.Parameter(torch.randn(self.n_routed, patch_len) * 0.02)
        self.D_shared = nn.Parameter(torch.randn(self.n_shared, patch_len) * 0.02)

        self.dead_threshold = dead_threshold_frac * (self.top_k / self.n_routed)
        self.aux_k_cap = max(1, int(aux_k_cap_frac * self.n_routed))
        self.ema_decay = ema_decay
        self.register_buffer('fire_ema', torch.zeros(self.n_routed))

    def _decode_atoms(self, idx, z, W_down_sel, b_down_sel, w_amp_sel, b_amp_sel, D_sel):
        """idx: [T, K] GLOBAL atom indices (0..n_stamps-1 regardless of routed/shared),
        z: [T, D] the token itself (same row for every atom, no per-atom view anymore),
        W_down_sel/b_down_sel/w_amp_sel/b_amp_sel/D_sel: all already gathered per-atom
        from whichever (routed or shared) table matches these atoms (see __init__).
        -> contribution [T, K, patch_len] = amp_i(z) * D_i, see class docstring.

        D_sel never depends on z (gathered straight from the parameter table, same
        template regardless of which token selected it) — only amp_i is a function of
        the token, via the same hidden_i bottleneck w_score reads."""
        z_sel = z.unsqueeze(1).expand(-1, idx.shape[1], -1)  # [T, K, D] — identical row per atom

        hidden = torch.einsum('tkd,tkdh->tkh', z_sel, W_down_sel) + b_down_sel  # [T, K, hidden]
        amp = torch.einsum('tkh,tkh->tk', hidden, w_amp_sel) + b_amp_sel  # [T, K]
        contribution = amp.unsqueeze(-1) * D_sel  # [T, K, patch_len]
        return contribution, z_sel

    def _generate_routed(self, idx, z):
        """idx: [T, K] routed-only indices (values in [0, n_routed)), used both as the
        GLOBAL index and to gather W_down_routed/b_down_routed/w_amp_routed/D_routed
        directly. z: [T, D] the token itself. Used by both the main forward path's
        routed half and the dead-atom aux rescue (rescue only ever draws from the
        routed pool, never shared — see forward())."""
        W_down_sel = self.W_down_routed[idx]
        b_down_sel = self.b_down_routed[idx]
        w_amp_sel = self.w_amp_routed[idx]
        b_amp_sel = self.b_amp_routed[idx]
        D_sel = self.D_routed[idx]
        return self._decode_atoms(idx, z, W_down_sel, b_down_sel, w_amp_sel, b_amp_sel, D_sel)

    def decode_selected(self, idx, h, z):
        """idx: [T, top_k+n_shared] GLOBAL indices, first top_k routed then n_shared shared
        (same layout forward() builds), h: [T, top_k+n_shared] their selection strengths,
        z: [T, D] the token itself -> contribution [T, top_k+n_shared, patch_len] (each
        slot's OWN RAW decoded output, unweighted by h and unsummed — real per-token-
        per-slot content, not yet collapsed into one reconstruction; viz/extract.py's
        per-patch grid reads this directly as a diagnostic of each atom's own generator
        response, deliberately raw so a runaway atom is visible even though _generate
        below gates it down before it ever reaches recon — see docs/agents/ or the "PSD
        averages away per-patch stamp identity" investigation), z_h [T, top_k+n_shared, D]
        (pre-generator finetune feature, z itself * h — every slot now reads the
        identical token, so z_h only differs across slots by its own h weight). Splits
        into routed/shared halves since they use separate W_down/W_out tables (see
        __init__) — sparse dispatch either way, only the K=top_k+n_shared selected atoms'
        params ever get gathered, never all n_stamps."""
        idx_routed, idx_shared = idx[:, :self.top_k], idx[:, self.top_k:]

        contribution_r, z_sel_r = self._generate_routed(idx_routed, z)

        local_shared = idx_shared - self.n_routed  # local index into the shared table
        W_down_s_sel = self.W_down_shared[local_shared]
        b_down_s_sel = self.b_down_shared[local_shared]
        w_amp_s_sel = self.w_amp_shared[local_shared]
        b_amp_s_sel = self.b_amp_shared[local_shared]
        D_s_sel = self.D_shared[local_shared]
        contribution_s, z_sel_s = self._decode_atoms(
            idx_shared, z, W_down_s_sel, b_down_s_sel, w_amp_s_sel, b_amp_s_sel, D_s_sel)

        contribution = torch.cat([contribution_r, contribution_s], dim=1)     # [T, top_k+n_shared, patch_len]
        z_sel = torch.cat([z_sel_r, z_sel_s], dim=1)                          # [T, top_k+n_shared, D]
        z_h = z_sel * h.unsqueeze(-1)
        return contribution, z_h

    def _generate(self, idx, h, z):
        """recon [T, patch_len] (plain UNWEIGHTED sum of decode_selected's per-slot
        contributions), z_h [T, top_k+n_shared, D] — see decode_selected for the per-slot
        contributions this sums.

        Tried both weighting the raw per-slot contribution by h (bled into recon's
        spectral shape, not just its strength) and normalizing each slot to unit norm
        before weighting by h (an explicit amplitude knob, but the per-slot norm division
        makes the generator's output a nonlinear function of z) — reverted both, back to
        a plain sum here, to keep the whole phi_i(z) generator (W_down_i -> W_out) a
        genuinely linear map end to end, per stamp. h still scales z_h (the
        finetune-facing feature) unchanged."""
        contribution, z_h = self.decode_selected(idx, h, z)
        recon = contribution.sum(dim=1)
        return recon, z_h

    def decorrelation_loss(self):
        """Pairwise cosine sim of each routed atom's own W_down direction (flattened
        [dim, hidden_width] -> [dim*hidden_width]), off-diagonal — decorrelates each
        atom's INPUT direction now that there's no spatial topography `u` to decorrelate
        (see class docstring: every atom reads the identical token z, so the only
        per-atom diversity left lives in W_down/W_out). Same off-diagonal-mean formula as
        the retired spatial version."""
        if self.n_routed < 2:
            return self.W_down_routed.new_zeros(())
        w = F.normalize(self.W_down_routed.reshape(self.n_routed, -1), dim=-1)
        sim = w @ w.t()
        off_diag = ~torch.eye(self.n_routed, dtype=torch.bool, device=sim.device)
        return sim[off_diag].pow(2).mean()

    @torch.no_grad()
    def fingerprint(self):
        """Every stamp's raw waveform template D_i, concatenated routed-then-shared.
        Unlike the old generator-based design (phi_i(z) = hidden_i @ W_out_i + b_out_i,
        an actual function of some probe z), D_i now has NO z dependence at all — it IS
        the shape, unconditionally, no fabricated probe needed and no "which probe do
        we use" question to answer (the old zero-probe default reduced the bottleneck
        to its bias terms only, understating diversity early in training — see git
        history for dense_probe's original docstring on that problem). amp_i(z) never
        touches shape, only overall scale/sign — see class docstring — so D_i alone is
        the complete, correct answer to "what does this atom look like". Dense over all
        n_stamps. Returns [n_stamps, patch_len]."""
        return torch.cat([self.D_routed, self.D_shared], dim=0)

    @torch.no_grad()
    def dense_probe(self, z):
        """The real per-token, per-atom CONTRIBUTION each stamp would produce if it had
        fired on this token — amp_i(z) * D_i, dense over all n_stamps (diagnostic-only,
        not the sparse-dispatch training path, which only ever gathers the
        top_k+n_shared selected atoms via decode_selected). Distinct from fingerprint()
        (the atom's own shape, no z at all) by design now: this shows the SCALED
        contribution a real token would get, fingerprint() shows the unscaled template
        underneath it.
        z: [T, D] real token embeddings (same input StampBank.forward takes) ->
        contribution [T, n_stamps, patch_len].
        """
        z = self.input_norm(z)

        hidden_r = torch.einsum('td,hdk->thk', z, self.W_down_routed) + self.b_down_routed  # [T, n_routed, hidden]
        amp_r = torch.einsum('thk,hk->th', hidden_r, self.w_amp_routed) + self.b_amp_routed  # [T, n_routed]
        contrib_r = amp_r.unsqueeze(-1) * self.D_routed.unsqueeze(0)  # [T, n_routed, patch_len]

        hidden_s = torch.einsum('td,hdk->thk', z, self.W_down_shared) + self.b_down_shared
        amp_s = torch.einsum('thk,hk->th', hidden_s, self.w_amp_shared) + self.b_amp_shared
        contrib_s = amp_s.unsqueeze(-1) * self.D_shared.unsqueeze(0)  # [T, n_shared, patch_len]

        contribution = torch.cat([contrib_r, contrib_s], dim=1)  # [T, n_stamps, patch_len]
        return contribution

    def forward(self, z, x_target=None):
        """
        z: [T, D] flat (channel, patch) token embeddings, x_target: [T, patch_len] the
        real patch content for that token (only needed for the dead-atom aux rescue,
        training only).

        Returns recon [T,patch_len], z_h [T, top_k+n_shared, D] (pre-generator finetune
        feature), h [T, top_k+n_shared] (selection strengths, routed then shared — same
        ordering convention `docs/adr/0007`'s gate used), dense_routed [T, n_routed]
        (zeros at unselected — the diagnostic object MeSAEFlatTrainer/MeSAEFlatCodebookChecker
        read for router-health/usage-histogram panels), decorr_loss, aux_loss. No `attn`
        anymore — there is no cross-channel pool left to produce a channel-attention map
        from (see class docstring); callers must not assume this field exists.

        No load-balance loss — see class docstring: sparse dispatch, no compute-balance
        problem, and forcing uniform routing would fight legitimate power-law usage. Dead-
        atom collapse is handled instead by fire_ema/dead_threshold/aux_loss below.
        """
        T = z.shape[0]
        z = self.input_norm(z)  # stabilize scale before scoring/generation, see __init__

        # Dense over all n_routed atoms (not just the ones that end up selected) — same
        # "compute-all, dispatch sparse" convention MoEFFN already uses in this file (see
        # its docstring's ponytail note). This is the SAME z@W_down_i+b_down_i bottleneck
        # _decode_atoms recomputes for just the selected atoms below — mild redundancy,
        # traded for keeping the sparse decode path untouched (see w_score's docstring for
        # why scoring needs every atom's own low-dim view, not just the winners').
        hidden_score = torch.einsum('td,hdk->thk', z, self.W_down_routed) + self.b_down_routed  # [T, n_routed, hidden]
        score = torch.einsum('thk,hk->th', hidden_score, self.w_score) + self.b_score  # [T, n_routed]

        topk_val, topk_idx = score.topk(self.top_k, dim=-1)
        h_routed = torch.softmax(topk_val, dim=-1)  # [T, top_k] — within-token selection confidence
        dense_routed = torch.zeros_like(score).scatter_(-1, topk_idx, h_routed)  # [T, n_routed]

        shared_idx = torch.arange(self.n_routed, self.n_stamps, device=z.device)
        shared_idx = shared_idx.unsqueeze(0).expand(T, -1)               # [T, n_shared]
        h_shared = score.new_full((T, self.n_shared), self.shared_weight)

        idx = torch.cat([topk_idx, shared_idx], dim=-1)   # [T, top_k+n_shared]
        h   = torch.cat([h_routed, h_shared], dim=-1)

        recon, z_h = self._generate(idx, h, z)

        decorr_loss = self.decorrelation_loss()

        aux_loss = recon.new_zeros(())
        if self.training:
            with torch.no_grad():
                fired = dense_routed.detach().gt(0).float().mean(dim=0)
                self.fire_ema.mul_(self.ema_decay).add_(fired, alpha=1 - self.ema_decay)
                dead_mask = self.fire_ema < self.dead_threshold  # [n_routed]

            if dead_mask.any() and x_target is not None:
                dead_score = score.masked_fill(~dead_mask.unsqueeze(0), float('-inf'))
                aux_k = min(self.aux_k_cap, int(dead_mask.sum().item()))
                aux_val, aux_idx = dead_score.topk(aux_k, dim=-1)
                # rescue only ever draws from the routed pool (dead atoms are a routed-only
                # concept, shared stamps are always "alive" by construction) — use
                # _generate_routed directly rather than the mixed routed+shared _generate.
                # Unweighted sum: trains rescued atoms' decoder (W_down/W_out) only, not
                # their score. Softmax-weighting by aux_val was tried to route gradient into
                # score too, but regressed real training (see MeSAE_modules.py's own
                # StampBank for the same revert) — reverted back to this simpler, working
                # version.
                contribution_aux, _ = self._generate_routed(aux_idx, z)
                recon_aux = contribution_aux.sum(dim=1)
                residual = (x_target - recon).detach()
                aux_loss = F.mse_loss(recon_aux, residual) / (residual.pow(2).mean() + 1e-8)

        return SimpleNamespace(
            recon=recon, z_h=z_h, h=h, idx=idx, dense_routed=dense_routed,
            decorr_loss=decorr_loss, aux_loss=aux_loss,
        )


class PerChannelHeadAttn(nn.Module):
    """
    Two-stage attention pooling — identical to model/MeFSQ/MeFSQ_modules.py's
    PerChannelHeadAttn (duplicated here rather than cross-imported, same convention as
    ExpertChannelPool/MultiHeadDecoder above: each model package stays self-contained).
    Fully backbone-agnostic — only needs z_per_head [B, N, H, d] at forward time, so it
    works unchanged whether H indexes MeFSQ Experts or MeSAE Filters.

    The channel dim is collapsed by the backbone itself (each Filter's own
    channel-attention View, see MeSAEFlatPretrain.encode_post_sae_expert) before this head ever
    sees the signal, so there's no channel stage here to re-pool.

    Stage 1 (temporal): a plain linear scorer over N patches, softmax-normalized.
    Stage 2 (unit): a plain linear scorer over the H units, softmax-normalized, pooling to
    a single [B, d] vector fed into cls.

    Both stages used to be a learnable-query dot-product ("key" Linear(d,hidden) dotted
    with a fixed learned "query" vector). That's algebraically just a single linear scalar
    function of z — composing two linear maps with no nonlinearity between them adds no
    expressiveness over one Linear(d,1), since the query is a fixed parameter, not
    content-derived. Collapsed to Linear(d,1): same capacity, fewer params.

    (Earlier version had a 3rd stage pooling over channels, fed a per-channel-decoded
    signal from encode_post_sae — retired for the same reason as the channel-concat
    classifier it replaced: redundant given the backbone already collapses channels into
    the Filter View before the SAE. See docs/agents/ / CONTEXT.md finetune val-chance bug
    for the original overparameterization this design avoids.)
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
