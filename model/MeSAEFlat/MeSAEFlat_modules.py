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
    Sparse source dictionary over CHANNEL-GROUPED tokens: input is [G, C, D] where each
    group g is one patch POSITION (G = B*N) carrying all C channels' embeddings for
    that moment. Selection runs once per group (shared by every channel); amplitude is
    read per channel. This is the instantaneous-mixing ICA picture made structural:
    x_c(t) = sum_s A[c, s] * source_s(t) — D_hat_i is source_s's waveform, and the
    [C] vector of per-channel amps for a selected stamp IS that source's mixing
    column (its topomap at that patch time), dense across channels by construction.

    The prior design selected top-k independently PER (channel, patch) token — which
    router-thresholded the topomap: a channel where the source arrived weakly lost the
    top-k race to whatever was louder there, its mixing coefficient became a hard 0,
    and the residue got absorbed by different stamps per channel, smearing one
    physical source across several channel-dependent stamps. Group selection is what
    binds one source to ONE stamp across the whole scalp.

    n_stamps splits into n_routed (compete via score + top-k, `docs/adr/0007`'s
    routed/shared split carried over from the Filter level) and n_shared (always
    included, fixed constant `shared_weight`).

    Selection is TopK-SAE style, aggregated over channels: per-atom group score =
    mean over VALID channels of amp_i(z_c)^2 (matched-filter energy summed over the
    scalp — an atom strong on a few channels or moderate on many both rank fairly),
    one top-k per group. The coefficient IS the score — the earlier per-token |amp|
    selection already established why (the retired w_score scorer had ZERO gradient
    from recon, ranking frozen at init, dead_feature_rate locked ~0.7, aux rescue
    training decoders the ranking would never pick); group aggregation keeps that
    property since amp is trained by recon MSE at every channel. h_routed = softmax
    over the selected group scores (a within-group selection confidence for
    diagnostics only, never touches recon).

    No load-balance loss, for a reason that got STRONGER after |amp| self-selection:
    the routing score IS the reconstruction coefficient now, so pushing the load
    distribution toward uniform is pushing reconstruction amplitudes toward uniform —
    unlike MoEFFN, whose gate is a free parameter with no other job, where uniformity
    costs only routing preference. A plain LB term here would be another auxiliary
    loss whose optimum ("every atom contributes equal energy on every patch") recon
    cannot veto, the exact failure pattern catalogued in the note above
    _spatial_weights. It would also fight legitimate power-law usage: measured load
    entropy on healthy runs is 0.73-0.81 of its maximum (v4/v5/v6, alive 0.54-0.97) —
    deliberately non-uniform, as a content-addressed dictionary should be, since real
    source prevalence is unequal (alpha everywhere, a rare artifact rarely).
    (An earlier version of this note also argued "sparse dispatch, no compute-balance
    problem" — that leg is now obsolete: _amp_dense computes every routed atom densely
    for group scoring. The statistical argument above is the load-bearing one.)
    Collapse is instead guarded by fire_ema/dead_threshold/aux_loss below, which are
    curative and content-AWARE (a revived atom is aimed at the residual, i.e. at
    content nothing else covers) where LB would be preventive and content-blind. If
    prevention is ever genuinely needed — group selection makes each atom's selection
    opportunities C times scarcer than the retired per-(channel,patch) routing did, so
    death is structurally likelier now — the safe shape is a HINGED entropy FLOOR
    (relu(0.70 - H/log(n_routed)), inactive across the healthy band, fires only on a
    real collapse like v8's 0.53), not a push toward uniform. That fraction is logged
    as stamp_router_entropy_frac.

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

        # No selection scorer params — selection is |amp_i(z)| directly (see class
        # docstring and forward()): the retired w_score/b_score never received gradient
        # from recon (h_routed only ever fed z_h), so ranking stayed at random init all
        # run; |amp| is trained by recon MSE and completes the earlier "self-relevance"
        # direction (scoring off the atom's own bottleneck) to its logical end — the
        # score IS what the atom would contribute.

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
        into this one dense computation plus a cheap gather in forward()."""
        hidden_r = torch.einsum('gcd,hdk->gchk', z, self.W_down_routed) + self.b_down_routed
        amp_r = torch.einsum('gchk,hkp->gchp', hidden_r, self.w_amp_routed) + self.b_amp_routed
        hidden_s = torch.einsum('gcd,hdk->gchk', z, self.W_down_shared) + self.b_down_shared
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
        """(D_all [n_stamps, L], H_all [n_stamps, L]) — unit templates (routed then
        shared, see the D_routed init comment) and their Hilbert quadrature partners
        (_quadrature), rebuilt each call so both track the live parameters."""
        D_all = torch.cat([F.normalize(self.D_routed, dim=-1),
                           F.normalize(self.D_shared, dim=-1)], dim=0)  # [n_stamps, L]
        return D_all, self._quadrature(D_all)

    def decode_selected(self, idx, amp):
        """idx: [G, top_k+n_shared] GLOBAL indices (routed then shared, forward()'s
        layout), amp: [G, C, top_k+n_shared, 2] per-channel quadrature gain pairs WITH
        rms already in (forward()'s out.amp) -> contribution
        [G, C, top_k+n_shared, patch_len] = a*D_hat + b*H_hat per slot, each slot's own
        raw decoded output per channel (unsummed — viz reads this to show per-stamp
        per-channel content; a stamp's [C] magnitude column sqrt(a^2+b^2) at one slot
        is its phase-invariant topomap at that patch time). Pure re-expansion of
        forward()'s already-computed quantities — no model re-evaluation, so callers
        can't accidentally decode with different selection/scale than training
        produced."""
        D_all, H_all = self._template_tables()
        return (amp[..., 0].unsqueeze(-1) * D_all[idx].unsqueeze(1)
                + amp[..., 1].unsqueeze(-1) * H_all[idx].unsqueeze(1))

    # decorrelation_loss and independence_loss are GONE, deliberately, not lost:
    # - decorrelation_loss (W_down direction repulsion) was life support for the
    #   retired untrainable w_score router (random directions overlapping, one atom
    #   winning everywhere). With |amp| group selection trained by recon, redundant
    #   atoms die naturally: interchangeable -> sparsity_loss shrinks one's amp at
    #   zero recon cost -> group score fades -> dead -> aux rescue re-aims it at the
    #   RESIDUAL (content nobody covers). The rescue is the decorrelation engine now.
    # - independence_loss (co-selected template Gram entropy) was blind to the real
    #   observed collapse (same-Hz-bin phase-tiled atoms are orthogonal in time
    #   domain -> full entropy, zero penalty), and the collapse it aimed at is an
    #   OBJECTIVE problem, not a redundancy one: under time-domain MSE on 1/f EEG,
    #   packing every atom into the loudest band is genuinely optimal. Fixed at the
    #   objective instead — spectrally whitened recon loss (MeSAEFlat._recon_loss),
    #   ICA's own mandatory whitening step — so frequency diversity pays for itself.

    # Four auxiliary dictionary-shaping losses were tried and RETIRED, each with
    # measured evidence — recorded so they don't get reinvented. The surviving one
    # (hinged spatial smoothness, below) is the only one shaped so it cannot reach its
    # own degenerate optimum:
    #  - decorr (W_down direction repulsion): life support for the retired untrainable
    #    w_score router. With |amp| group selection trained by recon, redundant atoms
    #    die on their own (interchangeable -> dead -> aux rescue re-aims them at the
    #    residual), a stronger mechanism than geometric repulsion.
    #  - indep (co-selected template Gram entropy): structurally blind to the collapse
    #    it targeted (same-Hz-bin phase-tiled atoms are orthogonal in time domain), and
    #    that collapse was an OBJECTIVE problem anyway — under time-domain MSE on 1/f
    #    EEG, packing every atom into the loudest band is optimal. Fixed properly by
    #    the spectrally whitened recon loss (MeSAEFlat._recon_loss).
    #  - L1 sparsity (raw, then signal-normalized): amp is a SMOOTH linear function of
    #    the bottleneck trained by plain SGD, with no proximal/soft-threshold step, so
    #    the subgradient never zeroed anything. v6: optimal global rescale alpha=1.27
    #    (recon 27% too small — LASSO shrinkage) with k_eff stuck ~23 of 30. Shrinkage
    #    without selection.
    #  - Hoyer sparsity: fixed the shrinkage (scale-invariant) but scale invariance
    #    also removed the reconstruction floor that had bounded L1, leaving a reachable
    #    degenerate optimum ("one atom carries everything"). v7 went there monotonically
    #    from epoch 1: k_eff 26.3 -> 5.2 -> 2.3 -> 1.7, router_entropy 0.00, dead 0.99.
    # The common pattern: every failure had a degenerate optimum the recon term could
    # not veto, and each collapsed a different axis — Hoyer the PER-TOKEN code (k_eff),
    # unbounded smoothness the POOL (alive fraction). A HINGED penalty (act only above
    # a target, zero pressure below) is the one safe shape, which is why the smoothness
    # term below is written that way. top_k remains the real sparsity budget, whitening
    # the real diversity mechanism, the aux rescue the real anti-collapse mechanism;
    # k_eff stays a logged diagnostic so drift is visible.

    @staticmethod
    def _spatial_weights(coords, sigma_scale=1.5):
        """coords: [C, 3] electrode positions -> W [C, C] gaussian spatial affinity,
        zero diagonal. sigma is set from the median nearest-neighbour distance times
        sigma_scale, so the kernel adapts to whatever montage/scale the caller uses
        (canonical 10-10 coords here) instead of a hard-coded length.

        Nearest-neighbour distances are taken over POSITIVE distances only, and only
        among channels with a real position: zero-padded channels all carry coords
        exactly (0, 0, 0) (see IO/dataset.py's channel mapping), so a plain
        median-nearest-neighbour collapses to 0 the moment a montage has more padded
        than mapped channels — which drives sigma to its clamp and produces a
        degenerate all-zero kernel (observed as a NaN smoothness loss on a
        2-subject Dial run: 56 of 64 channels padded)."""
        d = torch.cdist(coords, coords)                              # [C, C]
        real = coords.norm(dim=-1) > 1e-8                            # padded channels sit at the origin
        if real.sum() >= 2:
            dr = d[real][:, real]
            big = dr + torch.eye(dr.shape[0], device=d.device, dtype=d.dtype) * 1e9
            nn_d = big.min(dim=-1).values
        else:
            nn_d = d.flatten()
        pos = nn_d[nn_d > 1e-8]
        sigma = (pos.median() if pos.numel() else d.max().clamp(min=1e-3)) * sigma_scale
        W = torch.exp(-d.pow(2) / (2 * sigma.pow(2).clamp(min=1e-12)))
        return W - torch.diag(torch.diag(W))                         # no self-loops

    def smoothness_loss(self, amp_routed, coords, valid_channels=None, target=0.30):
        """amp_routed: [G, C, top_k, 2] the selected routed atoms' per-channel
        QUADRATURE pairs (the mixing columns), coords: [C, 3] electrode positions,
        valid_channels: [G, C] bool or None. Returns the energy-weighted graph
        Rayleigh quotient of each stamp's mixing column over the electrode graph,
        averaged over groups.

        Physical motivation: volume conduction makes a real dipolar source's scalp
        field spatially LOW-PASS — neighbouring electrodes see nearly the same
        thing. Nothing in this architecture enforces that: every channel's amp is
        estimated independently from its own token, so a stamp's mixing column is
        free to come out salt-and-pepper, which no physical source produces (a
        suspected cause of ICLabel classifying many stamps 'Other' — its clean
        classes are trained on real, smooth dipolar topographies).

        Both quadrature components are smoothed together: a zero-lag dipole has
        BOTH smooth amplitude and smooth phase across the scalp.

            R = u^T (D - W) u / u^T D u    per (group, slot), u = [C, 2] column

        SCALE-INVARIANT by construction (a Rayleigh quotient): it constrains the
        mixing column's SHAPE, never its magnitude, so it adds no shrinkage bias.
        Bounded in [0, 2]: 0 = perfectly flat field, ~1 = a random/uncorrelated
        field, high = rapid channel-to-channel sign/amplitude flips.

        HINGED at `target` — the loss is relu(R - target), i.e. exactly zero gradient
        once the field is already as smooth as physics calls for. This is the whole
        difference between this version and the unbounded one that had to be reverted:
        R=0 (a perfectly UNIFORM field) is a degenerate optimum, and an unbounded
        penalty goes straight to it. v8 did exactly that at weight 0.05 — R driven to
        0.018, per-channel contrast crushed from ~1.0 to 0.11 (uniformly bright
        topomaps), and the codebook collapsed with it (alive 16/84, template spectral
        entropy 0.35 -> 0.47) because spatial pattern is a main axis along which two
        stamps differ: force every column flat and stamps can only differ by waveform,
        so most become redundant and die.

        DEFAULT OFF (smooth_weight 0.0 in config) — measured as unnecessary, and the
        history is worth keeping. v10 (90 routed, top_k 18, no smoothness) settles at
        R ~0.40 on its own, i.e. already AT the physical reference (BETA_4s raw =
        0.396), and it produced the best reconstruction of any run to date (val
        whitened 0.0712 vs 0.083/0.086/0.096 for v8/v4/v6). With nothing to correct,
        a hinge at target=0.30 only pushes the field ~25% BELOW physics — that target
        was mis-set on the same "too smooth" side as the unbounded version it
        replaced. Keep the term available (smooth_R is logged unconditionally, so a
        genuinely rough run is still visible) but leave the weight at 0 unless
        smooth_R is measured well above the physical band, and then set target at or
        above ~0.40 rather than below it.

        target=0.30 is empirical, not a guess. Measured R of REAL EEG spatial patterns
        (same Rayleigh quotient, u_c = that channel's raw patch waveform): EEGMMIdb
        0.254 (62/64 real channels), BETA_4s 0.396 (58/64), Dial 0.206 (8/64, least
        reliable). Untouched model runs sit at R 0.44-0.58 (v4/v5/v6, pools 60-298) —
        genuinely ~1.5-2x rougher than physical fields, which is what justifies the
        prior at all, but nowhere near the salt-and-pepper (~1.0) this was originally
        assumed to be fixing. 0.30 sits in the measured band, deliberately toward the
        rougher end so the term under-corrects rather than over-corrects.

        Slots are weighted by their DETACHED relative energy share — a near-silent
        slot's direction is numerically meaningless, and detaching keeps the whole
        term pure-shape (no gradient path that could push magnitudes around)."""
        G, C, K, _ = amp_routed.shape
        if C < 3 or K < 1:
            return amp_routed.new_zeros(())
        # fp32: the [G,C,C]x[G,C,K,2] contraction below sums C^2*K*2 terms per group
        # and overflows fp16 under autocast (inf - inf = NaN), same convention as the
        # other numerically sensitive blocks in this file.
        with torch.autocast(device_type=amp_routed.device.type, enabled=False):
            u = amp_routed.float()
            W = self._spatial_weights(coords.float())                # [C, C]
            if valid_channels is not None:
                m = valid_channels.float()                           # [G, C]
                Wm = W.unsqueeze(0) * m.unsqueeze(1) * m.unsqueeze(2)  # [G, C, C]
            else:
                Wm = W.unsqueeze(0).expand(G, C, C)
            deg = Wm.sum(dim=-1)                                     # [G, C]

            e = u.pow(2).sum(dim=-1)                                 # [G, C, K] per-channel energy
            den = torch.einsum('gc,gck->gk', deg, e)                 # u^T D u
            # u^T W u, summed over the 2 quadrature components
            cross = torch.einsum('gcd,gckp,gdkp->gk', Wm, u, u)
            R = (den - cross) / (den + 1e-8)                         # [G, K] in [0, 2]

            with torch.no_grad():
                share = den / (den.sum(dim=-1, keepdim=True) + 1e-8)  # detached energy weights
                # Raw (un-hinged) R, stashed for logging — once the hinge is satisfied
                # the loss itself is identically 0 and would hide where R actually sits
                # (0.29 and 0.02 both read as 0). Same self-attribute convention as
                # MoEFFN._record_health. Compare against the measured physical band in
                # the docstring: ~0.25-0.40 is EEG-like, ~1.0 is random, ~0 is degenerate.
                self.last_smooth_R = (R * share).sum(dim=-1).mean()
            # hinge per (group, slot): no pressure on columns already at/below target
            excess = (R - target).clamp(min=0.0)
            return (excess * share).sum(dim=-1).mean()

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
        the complete, correct answer to "what does this atom look like". Returned
        unit-normalized, matching what the decode path actually uses (see the D_routed
        init comment — the raw parameter's norm is dead weight, never consumed). Dense
        over all n_stamps. Returns [n_stamps, patch_len]."""
        return torch.cat([F.normalize(self.D_routed, dim=-1),
                          F.normalize(self.D_shared, dim=-1)], dim=0)

    @torch.no_grad()
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
        z = self.input_norm(z)
        amp_r, amp_s = self._amp_dense(z)  # [G, C, n_routed, 2], [G, C, n_shared, 2]
        amp = torch.cat([amp_r, amp_s], dim=2)  # [G, C, n_stamps, 2]
        if rms is not None:
            amp = amp * rms.unsqueeze(-1)
        D_all, H_all = self._template_tables()  # each [n_stamps, L]
        return (amp[..., 0].unsqueeze(-1) * D_all.view(1, 1, self.n_stamps, -1)
                + amp[..., 1].unsqueeze(-1) * H_all.view(1, 1, self.n_stamps, -1))

    def forward(self, z, x_target=None, rms=None, valid_channels=None, coords=None,
                smooth_target=0.30):
        """
        z: [G, C, D] channel-grouped token embeddings (G = B*N patch positions, all C
        channels of one patch time per group — see class docstring), x_target:
        [G, C, patch_len] the real patch content (only needed for the dead-atom aux
        rescue, training only), rms: [G, C, 1] per-channel raw-input RMS or None —
        multiplied into every amp; callers running the real pipeline should always
        pass it. valid_channels: [G, C] bool, True = real (not zero-padded) channel,
        or None — used ONLY for the group selection score (a zero-padded channel's amp
        is encoder-bias noise that shouldn't vote on which sources this patch
        contains); padded channels still decode/reconstruct like any other, and the
        loss-side exclusion stays get_loss's job. coords: [C, 3] electrode positions
        or None (None skips smoothness_loss, e.g. probes that don't need it);
        smooth_target: the hinge point for that loss, see smoothness_loss.

        Returns recon [G, C, patch_len], idx [G, top_k+n_shared] (GLOBAL stamp ids,
        routed then shared — ONE selection per patch position, shared by all C
        channels), amp [G, C, top_k+n_shared, 2] (per-channel quadrature gain pairs
        (a, b), rms included — a slot's [C] magnitude column sqrt(a^2+b^2) is that
        stamp's phase-invariant mixing/topomap vector at this patch time, atan2(b, a)
        its per-channel phase), h [G, top_k+n_shared] (selection confidence,
        diagnostics only),
        dense_routed [G, n_routed] (zeros at unselected — the diagnostic object
        MeSAEFlatTrainer/MeSAEFlatCodebookChecker read for router-health/usage
        panels, now at patch-position granularity), aux_loss, smooth_loss (hinged
        spatial prior, see smoothness_loss), smooth_R (its raw un-hinged value, for
        logging), k_eff (diagnostic only).
        z_h is gone too: it was dead weight (nothing read it in either training
        stage, MeSAEFlatFinetune is NotImplemented on this branch).

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

        topk_val, topk_idx = group_score.topk(self.top_k, dim=-1)   # [G, top_k]
        h_routed = torch.softmax(topk_val, dim=-1)  # [G, top_k] — within-group selection confidence
        dense_routed = torch.zeros_like(group_score).scatter_(-1, topk_idx, h_routed)  # [G, n_routed]

        shared_idx = torch.arange(self.n_routed, self.n_stamps, device=z.device)
        shared_idx = shared_idx.unsqueeze(0).expand(G, -1)               # [G, n_shared]
        h_shared = group_score.new_full((G, self.n_shared), self.shared_weight)

        idx = torch.cat([topk_idx, shared_idx], dim=-1)   # [G, top_k+n_shared]
        h   = torch.cat([h_routed, h_shared], dim=-1)

        # Per-channel gains for the group's selected set: every channel decodes the
        # SAME stamps with its OWN amp — the [C] column per slot is the mixing vector.
        amp_sel_r = amp_r_dense.gather(
            2, topk_idx.view(G, 1, self.top_k, 1).expand(G, C, self.top_k, 2))
        amp = torch.cat([amp_sel_r, amp_s_dense], dim=2)  # [G, C, top_k+n_shared, 2]
        if rms is not None:
            amp = amp * rms.unsqueeze(-1)  # [G, C, 1, 1] broadcast — restores raw amplitude

        D_sel, H_sel = (t[idx] for t in self._template_tables())  # each [G, top_k+n_shared, patch_len]
        # a*D_hat + b*Hilbert(D_hat) summed over slots — no [G,C,K,L] materialized
        recon = (torch.einsum('gck,gkl->gcl', amp[..., 0], D_sel)
                 + torch.einsum('gck,gkl->gcl', amp[..., 1], H_sel))

        # Magnitude sqrt(a^2+b^2) per selected routed slot — the phase-invariant
        # amplitude, what sparsity/k_eff should see (penalize/count loudness, never
        # phase).
        amp_routed_sel = amp[:, :, :self.top_k, :]
        amp_mag_routed = amp_routed_sel.pow(2).sum(dim=-1).clamp(min=1e-12).sqrt()
        # The one surviving auxiliary term, and the only one shaped so it CANNOT run to
        # its own degenerate optimum (hinged — see smoothness_loss and the note above
        # _spatial_weights for the four that could, and did).
        if coords is not None:
            smooth_loss = self.smoothness_loss(amp_routed_sel, coords,
                                               valid_channels=valid_channels, target=smooth_target)
            smooth_R = self.last_smooth_R
        else:
            smooth_loss = amp.new_zeros(())
            smooth_R = amp.new_zeros(())

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
            aux_loss=aux_loss, smooth_loss=smooth_loss, smooth_R=smooth_R, k_eff=k_eff,
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
