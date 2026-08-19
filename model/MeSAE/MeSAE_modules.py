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
    Replaces ExpertChannelPool + FilterRouter + TopKSAE + MultiHeadDecoder with one
    dictionary of rank-1 spatiotemporal atoms ("stamps") — see
    docs/adr/0009-spatiotemporal-stamp-dictionary-for-mesae.md for the full derivation.
    Each stamp_i = (u_i in R^C static spatial topography, phi_i: R^D -> R^(C*patch_len)
    small generator). u_i is the topography (encode-side only, see _pool); phi_i turns
    that atom's own channel-pooled view of the patch into its contribution.

    n_stamps splits into n_routed (compete via score + top-k, `docs/adr/0007`'s
    routed/shared split carried over from the Filter level) and n_shared (always
    included, fixed constant `shared_weight`).

    Selection is a plain per-atom linear scorer: score_i = pooled_i . w_score_i +
    b_score_i, top-k over routed atoms, softmax over the selected top-k as h_routed
    (a within-patch selection confidence, not a self-reconstruction fit — see
    stamp_router_confidence in MeSAEPretrain.get_metrics). Deliberately NOT a shared
    nn.Linear(dim, n_routed) gate (MoEFFN's FFNRouter convention) — that assumes every
    expert reads the same input, but every atom here already has its OWN input
    (pooled_i, from its own channel-attention view, see _pool), so the scorer needs its
    own per-atom weight vector too. No load-balance loss: unlike MoEFFN (compute-all-
    then-mask, needs traffic spread for compute balance), this module dispatches
    sparsely (gather by idx) — no compute-balance problem, and forcing uniform routing
    would fight the legitimate power-law usage a content-addressed dictionary should
    have. Collapse is instead guarded by fire_ema/dead_threshold/aux_loss below.

    phi_i(pooled_i) is a small bottleneck MLP straight to the output, no tied
    reconstruction step: hidden_i = gelu(pooled_i @ W_down_i + b_down_i) -> [hidden_width],
    contribution_i = hidden_i @ W_out(group) + b_out(group) -> [C, patch_len]. W_down is
    per-atom (all cross-atom diversity lives here); W_out/b_out are shared within each
    group (routed vs shared, separate tables since hidden widths differ) rather than
    per-atom, same cost argument as the retired tied-autoencoder version's shared W_out
    (~D*C*patch_len params, independent of n_stamps, vs ~n_stamps x that for per-atom).
    Consequence: two atoms converging to a similar hidden_i now produce IDENTICAL output
    (deterministic shared function) — redundancy in hidden space maps 1:1 to redundancy
    in output space, a direct read on real redundancy for
    MeSAECodebookChecker's fingerprint-affinity panel.
    """
    def __init__(self, dim, num_channels, patch_len, n_stamps=800, n_shared_stamps=4,
                 top_k=32, hidden_width=8, shared_hidden_width=16, shared_weight=0.1,
                 cluster_bias=2.0, dead_threshold_frac=0.1, aux_k_cap_frac=0.04, ema_decay=0.999):
        super().__init__()
        self.n_stamps = n_stamps
        self.n_shared = n_shared_stamps
        self.n_routed = n_stamps - n_shared_stamps
        self.top_k = min(top_k, self.n_routed)
        self.shared_weight = shared_weight
        # Normalizes pooled_i before it's used for anything (scoring, the bottleneck's
        # generator input, z_h) — pooled_i inherits whatever scale the encoder currently
        # drifts to (documented block_norm growth across blocks/epochs elsewhere in this
        # codebase). Same role TopKSAE.input_norm played for its own encoder.
        self.input_norm = nn.LayerNorm(dim)
        self.dim = dim
        # Bottleneck width over the D-dim pooled_i input. Shared stamps get a wider
        # bottleneck than routed (16 vs 8 by default): they're always-on across every
        # patch/dataset (never gated out), so they need more room to represent structure
        # common across all data types rather than specializing narrowly like a routed
        # stamp can afford to.
        self.hidden_width = hidden_width
        self.shared_hidden_width = shared_hidden_width

        # u: spatial topography, all n_stamps rows (routed + shared share the same table,
        # shared rows just never get scored). Round-robin cluster init (h % num_channels,
        # not a linspace bucket) so EVERY stamp gets a real one-hot-biased starting
        # channel — n_stamps > num_channels is the common case (e.g. 300 stamps / 64
        # channels), and linspace(0, num_channels, n_stamps+1) rounds most consecutive
        # bounds[h]:bounds[h+1] slices to empty, silently skipping cluster_bias for most
        # stamps (only num_channels of them ever got a real bias). Those un-biased stamps
        # started at near-uniform channel attention (near-identical near-mean-pooled
        # input to every one of them), a real structural cause of router collapse
        # (confirmed: attn spread ~0.01-0.07 around the 1/num_channels uniform value).
        logit = torch.randn(n_stamps, num_channels) * 0.02
        channel_idx = torch.arange(n_stamps) % num_channels
        logit[torch.arange(n_stamps), channel_idx] += cluster_bias
        self.u = nn.Parameter(logit)

        # Per-atom linear selection scorer, routed atoms only (shared atoms are never
        # scored/gated, see class docstring) — score_i = pooled_i . w_score_i + b_score_i.
        self.w_score = nn.Parameter(torch.empty(self.n_routed, dim))
        self.b_score = nn.Parameter(torch.zeros(self.n_routed))
        nn.init.kaiming_uniform_(self.w_score, a=math.sqrt(5))

        # phi: bottleneck generator, per-atom W_down/b_down (down-project + gelu only, no
        # tied up-project — see class docstring), decoding through a per-GROUP shared
        # W_out/b_out straight to [C, patch_len]. Routed and shared use separate
        # W_down/b_down/W_out/b_out tables (different hidden widths).
        self.W_down_routed = nn.Parameter(torch.empty(self.n_routed, dim, hidden_width))
        self.b_down_routed = nn.Parameter(torch.zeros(self.n_routed, hidden_width))
        self.W_down_shared = nn.Parameter(torch.empty(self.n_shared, dim, shared_hidden_width))
        self.b_down_shared = nn.Parameter(torch.zeros(self.n_shared, shared_hidden_width))
        self.W_out_routed = nn.Parameter(torch.empty(hidden_width, num_channels, patch_len))
        self.b_out_routed = nn.Parameter(torch.zeros(num_channels, patch_len))
        self.W_out_shared = nn.Parameter(torch.empty(shared_hidden_width, num_channels, patch_len))
        self.b_out_shared = nn.Parameter(torch.zeros(num_channels, patch_len))
        nn.init.kaiming_uniform_(self.W_down_routed, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_down_shared, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out_routed, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out_shared, a=math.sqrt(5))

        self.dead_threshold = dead_threshold_frac * (self.top_k / self.n_routed)
        self.aux_k_cap = max(1, int(aux_k_cap_frac * self.n_routed))
        self.ema_decay = ema_decay
        self.register_buffer('fire_ema', torch.zeros(self.n_routed))

    def _pool(self, z, valid_mask=None):
        """z: [M, C, D] -> pooled [M, n_stamps, D], attn [M, n_stamps, C]. Same math as the
        retired ExpertChannelPool, just n_stamps rows instead of n_filters."""
        scores = self.u.unsqueeze(0).expand(z.shape[0], -1, -1)  # [M, n_stamps, C]
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask[:, None, :], float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        pooled = torch.einsum('mhc,mcd->mhd', attn, z)
        return pooled, attn

    def _decode_atoms(self, idx, pooled, W_down_sel, b_down_sel, W_out, b_out):
        """idx: [M, K] GLOBAL atom indices (into pooled, 0..n_stamps-1 regardless of
        routed/shared), W_down_sel/b_down_sel: already gathered from whichever (routed or
        shared) table matches these atoms, W_out/b_out: that group's SHARED decode table
        (no gather — same matrix for every atom in the group, see __init__).
        -> contribution [M, K, C, patch_len], pooled_sel [M, K, D]."""
        D = pooled.shape[-1]
        pooled_sel = pooled.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))  # [M, K, D]

        pre = torch.einsum('mkd,mkdh->mkh', pooled_sel, W_down_sel) + b_down_sel  # [M, K, hidden]
        hidden = F.gelu(pre)
        contribution = torch.einsum('mkh,hcp->mkcp', hidden, W_out) + b_out  # [M, K, C, patch_len]
        return contribution, pooled_sel

    def _generate_routed(self, idx, pooled):
        """idx: [M, K] routed-only indices (values in [0, n_routed), used both as the
        GLOBAL index (pooled's routed range matches global indices 0..n_routed-1 1:1) and
        to gather W_down_routed/b_down_routed directly. Used by both the main forward
        path's routed half and the dead-atom aux rescue (rescue only ever draws from the
        routed pool, never shared — see forward())."""
        W_down_sel = self.W_down_routed[idx]
        b_down_sel = self.b_down_routed[idx]
        return self._decode_atoms(idx, pooled, W_down_sel, b_down_sel, self.W_out_routed, self.b_out_routed)

    def decode_selected(self, idx, h, pooled):
        """idx: [M, top_k+n_shared] GLOBAL indices, first top_k routed then n_shared shared
        (same layout forward() builds), h: [M, top_k+n_shared] their selection strengths
        (used only for z_h/diagnostics, not fed into generation), pooled: [M, n_stamps, D]
        (dense, from _pool) -> contribution [M, top_k+n_shared, C, patch_len] (each slot's
        OWN decoded output, unsummed — real per-patch-per-slot content, not yet collapsed
        into one reconstruction; viz/extract.py's per-patch grid reads this directly, see
        docs/agents/ or the "PSD averages away per-patch stamp identity" investigation),
        z_h [M, top_k+n_shared, D] (pre-generator finetune feature, gathered pooled view *
        h). Splits into routed/shared halves since they use separate W_down/W_out tables
        (see __init__) — sparse dispatch either way, only the K=top_k+n_shared selected
        atoms' params ever get gathered, never all n_stamps."""
        idx_routed, idx_shared = idx[:, :self.top_k], idx[:, self.top_k:]

        contribution_r, pooled_sel_r = self._generate_routed(idx_routed, pooled)

        W_down_s_sel = self.W_down_shared[idx_shared - self.n_routed]  # local index into the shared table
        b_down_s_sel = self.b_down_shared[idx_shared - self.n_routed]
        contribution_s, pooled_sel_s = self._decode_atoms(
            idx_shared, pooled, W_down_s_sel, b_down_s_sel, self.W_out_shared, self.b_out_shared)

        contribution = torch.cat([contribution_r, contribution_s], dim=1)     # [M, top_k+n_shared, C, patch_len]
        pooled_sel = torch.cat([pooled_sel_r, pooled_sel_s], dim=1)           # [M, top_k+n_shared, D]
        z_h = pooled_sel * h.unsqueeze(-1)
        return contribution, z_h

    def _generate(self, idx, h, pooled):
        """recon [M, C, patch_len] (plain unweighted sum of decode_selected's per-slot
        contributions), z_h [M, top_k+n_shared, D] — see decode_selected for the per-slot
        version.

        h-weighting the sum was tried (each slot's contribution scaled by its own
        selection strength before summing, to stop atoms cancelling out at large opposite-
        sign magnitudes) but regressed router health — a low-confidence atom earning
        almost no gradient credit for a genuinely useful contribution pushed the top-k
        softmax to sharpen onto fewer atoms over training (see git history on this
        method). Reverted back to the plain sum; the cancellation issue it was meant to
        fix is a real but separate problem, not solved here."""
        contribution, z_h = self.decode_selected(idx, h, pooled)
        recon = contribution.sum(dim=1)
        return recon, z_h

    def decorrelation_loss(self):
        """Spatial-only for now (pairwise cosine sim of u rows, off-diagonal) — same
        formula as the retired ExpertChannelPool.decorrelation_loss, just n_stamps wide.
        ADR leaves a temporal term (over flatten(outer(u_i, phi_i(probe))) instead of u_i
        alone) as an open follow-up, gated on whether MeSAECodebookChecker's
        decoder_fingerprint_matrix panel shows atoms staying redundant despite this."""
        if self.n_stamps < 2:
            return self.u.new_zeros(())
        w = F.normalize(self.u, dim=-1)
        sim = w @ w.t()
        off_diag = ~torch.eye(self.n_stamps, dtype=torch.bool, device=sim.device)
        return sim[off_diag].pow(2).mean()

    @torch.no_grad()
    def fingerprint(self, probe=None):
        """Every stamp's full [C, patch_len] pattern at a fixed canonical D-dim probe
        (default zero vector — reduces the bottleneck to its bias terms, still a
        deterministic per-atom parameter fingerprint, just no longer "what this atom
        always outputs regardless of input"). Dense over all n_stamps (diagnostic-only
        call, not the training/sparse-dispatch path — see
        MeSAECodebookChecker.decoder_fingerprint_matrix). Returns [n_stamps, C, patch_len].
        """
        if probe is None:
            probe = self.u.new_zeros(self.n_stamps, self.dim)
        else:
            probe = probe.expand(self.n_stamps, self.dim)

        pre_r = torch.einsum('kd,kdh->kh', probe[:self.n_routed], self.W_down_routed) + self.b_down_routed
        contrib_r = torch.einsum('kh,hcp->kcp', F.gelu(pre_r), self.W_out_routed) + self.b_out_routed

        pre_s = torch.einsum('kd,kdh->kh', probe[self.n_routed:], self.W_down_shared) + self.b_down_shared
        contrib_s = torch.einsum('kh,hcp->kcp', F.gelu(pre_s), self.W_out_shared) + self.b_out_shared

        return torch.cat([contrib_r, contrib_s], dim=0)  # [n_stamps, C, patch_len]

    @torch.no_grad()
    def dense_probe(self, z, valid_mask=None):
        """Like fingerprint(), but fed each atom's REAL pooled_i (its actual channel-
        weighted view of this real patch) instead of a fabricated zero vector — the
        genuine "run this stamp's own generator on its real input" view. fingerprint()'s
        zero probe collapses pre = 0@W_down + b_down down to just b_down, so W_down's
        actual response to content never gets exercised — with b_down still zero-init and
        slow to grow, that made every atom's fingerprint mostly reflect shared near-zero
        bias behavior rather than the (possibly already quite different) weight matrices
        underneath, systematically understating diversity early in training. Dense over
        all n_stamps (diagnostic-only, not the sparse-dispatch training path) but needs
        real (x, coords) data, unlike fingerprint().
        z: [M, C, D] real pooled-channel patch embeddings (same input StampBank.forward
        takes) -> contribution [M, n_stamps, C, patch_len], attn [M, n_stamps, C]
        (encode-side channel attention only — no longer feeds the output — kept here
        purely as a diagnostic, caller decides how to use it).
        """
        pooled, attn = self._pool(z, valid_mask)
        pooled = self.input_norm(pooled)

        pre_r = torch.einsum('mhd,hdk->mhk', pooled[:, :self.n_routed], self.W_down_routed) + self.b_down_routed
        contrib_r = torch.einsum('mhk,kcp->mhcp', F.gelu(pre_r), self.W_out_routed) + self.b_out_routed

        pre_s = torch.einsum('mhd,hdk->mhk', pooled[:, self.n_routed:], self.W_down_shared) + self.b_down_shared
        contrib_s = torch.einsum('mhk,kcp->mhcp', F.gelu(pre_s), self.W_out_shared) + self.b_out_shared

        contribution = torch.cat([contrib_r, contrib_s], dim=1)  # [M, n_stamps, C, patch_len]
        return contribution, attn

    def forward(self, z, x_target=None, valid_mask=None):
        """
        z: [M, C, D] pooled-channel patch embeddings, x_target: [M, C, patch_len] the real
        patch (only needed for the dead-atom aux rescue, training only), valid_mask: [M, C].

        Returns recon [M,C,patch_len], attn [M,n_stamps,C] (channel-attention, for topomaps
        — full dense table, not gathered), z_h [M, top_k+n_shared, D] (pre-generator
        finetune feature), h [M, top_k+n_shared] (selection strengths, routed then shared —
        same ordering convention `docs/adr/0007`'s gate used), dense_routed [M, n_routed]
        (zeros at unselected — the diagnostic object MeSAETrainer/MeSAECodebookChecker read
        for router-health/usage-histogram panels), decorr_loss, aux_loss.

        No load-balance loss — see class docstring: sparse dispatch, no compute-balance
        problem, and forcing uniform routing would fight legitimate power-law usage. Dead-
        atom collapse is handled instead by fire_ema/dead_threshold/aux_loss below.
        """
        M = z.shape[0]
        pooled, attn = self._pool(z, valid_mask)  # [M, n_stamps, D], [M, n_stamps, C]
        pooled = self.input_norm(pooled)  # stabilize scale before scoring/generation, see __init__

        pooled_routed = pooled[:, :self.n_routed]
        score = torch.einsum('mhd,hd->mh', pooled_routed, self.w_score) + self.b_score  # [M, n_routed]

        topk_val, topk_idx = score.topk(self.top_k, dim=-1)
        h_routed = torch.softmax(topk_val, dim=-1)  # [M, top_k] — within-patch selection confidence
        dense_routed = torch.zeros_like(score).scatter_(-1, topk_idx, h_routed)  # [M, n_routed]

        shared_idx = torch.arange(self.n_routed, self.n_stamps, device=z.device)
        shared_idx = shared_idx.unsqueeze(0).expand(M, -1)               # [M, n_shared]
        h_shared = score.new_full((M, self.n_shared), self.shared_weight)

        idx = torch.cat([topk_idx, shared_idx], dim=-1)   # [M, top_k+n_shared]
        h   = torch.cat([h_routed, h_shared], dim=-1)

        recon, z_h = self._generate(idx, h, pooled)

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
                # score too, but regressed real training (see git history on this block) —
                # reverted back to this simpler, working version.
                contribution_aux, _ = self._generate_routed(aux_idx, pooled)
                recon_aux = contribution_aux.sum(dim=1)
                residual = (x_target - recon).detach()
                aux_loss = F.mse_loss(recon_aux, residual) / (residual.pow(2).mean() + 1e-8)

        return SimpleNamespace(
            recon=recon, attn=attn, pooled=pooled, z_h=z_h, h=h, idx=idx, dense_routed=dense_routed,
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
    channel-attention View, see MeSAEPretrain.encode_post_sae_expert) before this head ever
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
