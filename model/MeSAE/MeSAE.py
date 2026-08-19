from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MeSAE.MeSAE_modules import SpatialTemporalEmbeddings, TSAEncoder, StampBank, PerChannelHeadAttn


def _ema_update(buf, val, decay=0.99):
    """In-place EMA update, skipped if val is NaN/Inf. A plain `.mul_(decay).add_(val,
    alpha=1-decay)` permanently poisons buf the moment val is ever NaN even once (a single
    bad batch, e.g. a transient fp16-autocast overflow early in training): NaN propagates
    through every future update (decay*NaN + (1-decay)*anything = NaN), so a diagnostic
    stuck NaN for an entire run can trace back to one early outlier batch long since
    recovered from. Skipping non-finite updates lets the EMA keep tracking real values."""
    if torch.isfinite(val):
        buf.mul_(decay).add_(val, alpha=1 - decay)


class MeSAEPretrain(nn.Module):
    """
    Spatiotemporal stamp-dictionary EEG tokenizer — parallel to MeFSQPretrain, not a
    variant of it. Goal is explainable, per-patch embeddings (not a discrete vocabulary):
    channel-count invariant (cross-dataset unification still matters) but NOT
    length-invariant (each patch keeps its own embedding, for temporal localization of
    events within a trial).

    Pipeline: encoder -> StampBank (dictionary of rank-1 spatiotemporal atoms — static
    per-stamp channel topography x a small per-stamp generator MLP conditioned on that
    stamp's own selection strength) -> reconstruction, summed directly in patch space (no
    separate decoder stage). See docs/adr/0009-spatiotemporal-stamp-dictionary-for-mesae.md
    for the full derivation and docs/adr/0007-routed-filter-gating-for-mesae.md for the
    routed/shared split StampBank carries over from the retired per-Filter design.

    Trains in two sequential stages (see docs/adr/0003-mesae-two-stage-masked-training.md
    and CONTEXT.md: Tokenizer stage / Masked stage):
    - Tokenizer stage: temporal/spatial mixing OFF (enable_temporal/enable_spatial not yet
      called), no masking (bool_masked_pos=None) — encoder + StampBank train jointly on
      patch-local features only, so the stamp dictionary can't be built from
      already-context-leaked input.
    - Masked stage: call enable_temporal(), enable_spatial(), freeze_stamps() (in that
      order), then train with bool_masked_pos set — only embed/encoder/mask_token keep
      learning, predicting masked patches through the now-frozen StampBank.
    """
    def __init__(
        self,
        embed_dim=100,
        enc_depth=12,
        mlp_ratio=4.0,
        patch_len=20,
        spatial_heads=10,
        dropout=0.0,
        pool_after_blocks=(),
        num_channels=1,
        n_stamps=800,
        n_shared_stamps=4,
        stamp_top_k=32,
        stamp_hidden_width=8,
        stamp_shared_hidden_width=16,
        dead_threshold_frac=0.1,
        aux_k_cap_frac=0.04,
        stamp_ema_decay=0.999,
        n_routed_ffn_experts=4,
        n_shared_ffn_experts=1,
        ffn_top_k=2,
        ffn_expert_hidden=None,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.head_dim = embed_dim
        self.num_channels = num_channels

        self.embed   = SpatialTemporalEmbeddings(patch_len, embed_dim)
        self.encoder = TSAEncoder(embed_dim, depth=enc_depth, num_heads=spatial_heads, mlp_ratio=mlp_ratio,
                                   dropout=dropout, pool_after_blocks=pool_after_blocks,
                                   n_routed_ffn_experts=n_routed_ffn_experts, n_shared_ffn_experts=n_shared_ffn_experts,
                                   ffn_top_k=ffn_top_k, ffn_expert_hidden=ffn_expert_hidden)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.stamps = StampBank(
            embed_dim, num_channels, patch_len,
            n_stamps=n_stamps, n_shared_stamps=n_shared_stamps, top_k=stamp_top_k,
            hidden_width=stamp_hidden_width, shared_hidden_width=stamp_shared_hidden_width,
            dead_threshold_frac=dead_threshold_frac,
            aux_k_cap_frac=aux_k_cap_frac, ema_decay=stamp_ema_decay,
        )
        # convenience aliases — viz/checker code reads these off the model directly
        # (e.g. base_checker.py compute_unit_colors), same convention the retired
        # n_filters/n_routed_filters/n_shared_filters attrs used.
        self.n_stamps = self.stamps.n_stamps
        self.n_routed_stamps = self.stamps.n_routed
        self.n_shared_stamps = self.stamps.n_shared
        self.shared_weight = self.stamps.shared_weight
        self.stamps_frozen = False

        # EMA router-health buffers — same 3 metrics as MeFSQ's Router
        # (ema_stamp_router_entropy/ema_stamp_router_load_std/ema_stamp_gate_entropy), see
        # update_stamp_router_metrics below for what each number means.
        self.register_buffer('ema_stamp_router_entropy',  torch.tensor(0.0))
        self.register_buffer('ema_stamp_router_load_std', torch.tensor(0.0))
        self.register_buffer('ema_stamp_gate_entropy',    torch.tensor(0.0))

        # Same 3 EMA metrics, but for the FFN MoE routers (MoEFFN/FFNRouter, one per
        # TSABlock, averaged across blocks by TSAEncoder.forward) — a distinct MoE from the
        # stamp router above, see docs/adr/0008-moe-ffn-for-mesae.md and
        # update_ffn_router_metrics below.
        self.register_buffer('ema_ffn_router_entropy',  torch.tensor(0.0))
        self.register_buffer('ema_ffn_router_load_std', torch.tensor(0.0))
        self.register_buffer('ema_ffn_gate_entropy',    torch.tensor(0.0))

    def enable_spatial(self):
        self.embed.enable_spatial()
        self.encoder.enable_spatial()

    def enable_temporal(self):
        self.encoder.enable_temporal()

    def freeze_stamps(self):
        """
        End of Tokenizer stage: lock StampBank so the Masked stage's frozen reconstruction
        target stops moving. aux_loss (dead-atom rescue) must be dropped from the
        Masked-stage loss entirely once this is called — rescuing a frozen dictionary's dead
        atoms is meaningless, see get_loss. Mirrors MeFSQ's freeze_vq_and_decoder(), but
        two-stage/sequential rather than joint-warmup-then-freeze (see
        docs/adr/0003-mesae-two-stage-masked-training.md).
        """
        for p in self.stamps.parameters():
            p.requires_grad_(False)
        self.stamps_frozen = True

    @torch.no_grad()
    def update_stamp_router_metrics(self, h_routed_dense):
        """
        EMA router-health monitoring, called once per step (see
        MeSAETrainer.update_diagnostics) — logic ported from MeFSQ's
        MeFSQ.update_head_metrics, adapted for StampBank's raw (not softmax-normalized)
        selection strengths. h_routed_dense: [M, n_routed_stamps] (StampBank's
        `dense_routed`, zeros at unselected).

        stamp_router_entropy: entropy of the routed pool's LOAD distribution (how evenly,
        across this batch's patches, selection is spread over the n_routed_stamps routed
        stamps) — 0 = every patch always picks the same stamp (total collapse),
        log(n_routed_stamps) = perfectly uniform load. Rising over training = healthy
        (stamps differentiating and each still getting used); falling toward 0 = router
        collapse (a couple of stamps absorbing everything, see docs/adr/0007).

        stamp_gate_entropy: entropy of the WITHIN-patch selection strengths (not across
        patches). Unlike the retired FilterRouter's softmax gate, `h_routed_dense` is raw
        and unbounded — not a probability distribution — so it's renormalized per-row
        (`/ sum`) here purely for this diagnostic, never touching the actual reconstruction
        path. 0 = one selected stamp dominates that patch's strength (confident/peaked
        selection), log(top_k) = the k selected stamps split strength near-uniformly.

        stamp_router_load_std: std of the load distribution across routed stamps —
        companion to stamp_router_entropy in raw (non-normalized) units; rising = load
        spreading out unevenly (some stamps starved), reacts faster than the log-scaled
        entropy number.
        """
        selected = (h_routed_dense.detach() > 0).float()
        load = selected.mean(dim=0)
        load_p = load / (load.sum() + 1e-8)
        stamp_router_entropy = -(load_p * torch.log(load_p + 1e-10)).sum()

        gm = h_routed_dense.detach().float().clamp(min=0)
        gm = gm / (gm.sum(dim=-1, keepdim=True) + 1e-8)  # diagnostic-only renormalization
        stamp_gate_entropy = -(gm * torch.log(gm + 1e-10)).sum(dim=-1).mean()

        _ema_update(self.ema_stamp_router_load_std, load.std())
        _ema_update(self.ema_stamp_router_entropy, stamp_router_entropy)
        _ema_update(self.ema_stamp_gate_entropy, stamp_gate_entropy)

    @torch.no_grad()
    def update_ffn_router_metrics(self, ffn_router_entropy, ffn_router_load_std, ffn_gate_entropy):
        """Same EMA smoothing as update_head_metrics, for the FFN MoE routers instead of the
        SAE Filter router — see TSAEncoder.forward (MeSAE_modules.py) for where these three
        already-averaged-across-blocks values come from. Called from
        MeSAETrainer.update_diagnostics with out.ffn_router_entropy/out.ffn_router_load_std/
        out.ffn_gate_entropy, same call site as update_head_metrics."""
        _ema_update(self.ema_ffn_router_load_std, ffn_router_load_std)
        _ema_update(self.ema_ffn_router_entropy, ffn_router_entropy)
        _ema_update(self.ema_ffn_gate_entropy, ffn_gate_entropy)

    def stage_features(self, x, coords, time_idx=None, bool_masked_pos=None):
        """Returns (z [B, C, N, D], ffn_lb_loss scalar) — ffn_lb_loss is the summed
        load-balance loss of every TSABlock's MoEFFN (see MeSAE_modules.TSAEncoder),
        distinct from the router (SAE Filter) load-balance loss produced in forward()."""
        z = self.embed(x, coords=coords, time_idx=time_idx)  # [B, C, N, D]
        if bool_masked_pos is not None:
            mask = bool_masked_pos.unsqueeze(-1).type_as(z)  # [B, C, N, 1]
            z = z * (1.0 - mask) + self.mask_token * mask
        return self.encoder(z)  # [B, C, N, D], ffn_lb_loss

    def _pool_channels(self, z, valid_channels=None):
        """z: [B, C, N, D] -> z_bnc [M, C, D] (M=B*N), valid_mask [M, C] or None."""
        B, C, N, D = z.shape
        z_bnc = z.permute(0, 2, 1, 3).reshape(B * N, C, D)
        valid_mask = None
        if valid_channels is not None:
            valid_mask = valid_channels.unsqueeze(1).expand(B, N, C).reshape(B * N, C)
        return z_bnc, valid_mask

    @staticmethod
    def _flatten_patches(x):
        """x: [B, C, N, L] -> x_mcl [M, C, L] (M=B*N) — same M ordering as _pool_channels,
        needed as StampBank's aux-rescue reconstruction target (see StampBank.forward)."""
        B, C, N, L = x.shape
        return x.permute(0, 2, 1, 3).reshape(B * N, C, L)

    def encode_post_stamp_expert(self, x, coords, time_idx=None, valid_channels=None, return_chan_attn=False):
        """
        Per-stamp code BEFORE generation — channels are already collapsed into each
        selected stamp's own channel-attention View by StampBank's spatial pool, so there
        is no channel dim here. Use this when a downstream head only needs
        stamp-differentiated signal and would otherwise have to re-pool the channel dim the
        backbone already collapsed. Mirrors MeFSQPretrain.encode_post_vq_expert.
        valid_channels: [B, C] bool, True = real (not zero-padded) channel (optional).
        return_chan_attn: also return each selected stamp's own channel-attention weights
        (the real per-channel importance, for diagnostics — e.g. attn-topo panels).
        Returns z_h [B, N, Q, D] (Q = top_k+n_shared, 36 by default), or (z_h, chan_attn
        [B, N, Q, C]) if return_chan_attn. Unselected routed stamps never appear here at
        all (hard top-k, not a zeroed-out dense slot) — the finetune head only ever sees
        the stamps actually selected for a given patch.
        """
        B, C, N, L = x.shape

        z, _ = self.stage_features(x, coords, time_idx=time_idx, bool_masked_pos=None)  # [B, C, N, D]
        z_bnc, valid_mask = self._pool_channels(z, valid_channels)  # [M, C, D]

        out = self.stamps(z_bnc, valid_mask=valid_mask)
        z_h = out.z_h.reshape(B, N, -1, self.head_dim)  # [B, N, Q, D]
        if not return_chan_attn:
            return z_h

        # gather the selected stamps' channel-attention rows out of the dense [M,n_stamps,C]
        # table StampBank.forward already computed, same idx it used to build z_h.
        chan_attn = out.attn.gather(1, out.idx.unsqueeze(-1).expand(-1, -1, C))  # [M, Q, C]
        return z_h, chan_attn.reshape(B, N, -1, C)

    def used_stamp_ids(self, out, max_stamps=100):
        """Global stamp ids actually selected SOMEWHERE across this batch (a batch built
        from one trial's patches, in practice — see check_pretrain/check_finetune), capped
        at max_stamps, ranked by accumulated selection strength. `out` needs `dense_routed`
        (from this model's own `forward`/`stamps(...)` output — any SimpleNamespace with
        that field works). Shared stamps always included first (constant weight, always
        selected every patch, so cheap to guarantee) — remaining budget filled by the
        highest-usage routed stamps, dropping ones that never fired at all this batch. This
        exists because hard top-k selection means a per-patch Q axis (see
        encode_post_stamp_expert) has NO stable cross-patch identity — a trial-wide view
        needs a fixed, shared set of global ids instead.
        """
        device = out.dense_routed.device
        shared_ids = torch.arange(self.n_routed_stamps, self.n_stamps, device=device)
        routed_usage = out.dense_routed.detach().sum(dim=0)  # [n_routed_stamps]
        routed_ids = torch.nonzero(routed_usage > 0, as_tuple=True)[0]
        order = torch.argsort(routed_usage[routed_ids], descending=True)
        routed_ids = routed_ids[order]
        budget = max(0, max_stamps - shared_ids.numel())
        return torch.cat([shared_ids, routed_ids[:budget]])

    def encode_used_stamps(self, x, coords, time_idx=None, valid_channels=None, max_stamps=100):
        """Trial-wide, stable-identity counterpart to encode_post_stamp_expert — instead of
        each patch's own (patch-locally-indexed) top-k picks, returns a single FIXED set of
        <=max_stamps global stamp ids (see used_stamp_ids) and every patch's real strength
        for exactly those ids (0 where that patch's own top-k didn't select a given id —
        reconstructed from StampBank's dense per-patch usage, not re-selected). Used by
        MeSAEChecker.render_finetune_attn for a trial-wide attn-topo panel; NOT used by
        MeSAEFinetune.forward itself, which still needs the real per-patch top-k path
        (encode_post_stamp_expert) since that's what the model was actually trained on.
        Returns (z_h [B,N,Qu,D], chan_attn [B,N,Qu,C], used_ids [Qu]).
        """
        B, C, N, L = x.shape

        z, _ = self.stage_features(x, coords, time_idx=time_idx, bool_masked_pos=None)  # [B, C, N, D]
        z_bnc, valid_mask = self._pool_channels(z, valid_channels)  # [M, C, D]

        out = self.stamps(z_bnc, valid_mask=valid_mask)
        used_ids = self.used_stamp_ids(out, max_stamps=max_stamps)  # [Qu]

        shared = out.dense_routed.new_full((out.dense_routed.shape[0], self.n_shared_stamps), self.shared_weight)
        full_dense = torch.cat([out.dense_routed, shared], dim=-1)  # [M, n_stamps]

        z_h = out.pooled[:, used_ids, :] * full_dense[:, used_ids].unsqueeze(-1)  # [M, Qu, D]
        chan_attn = out.attn[:, used_ids, :]                                       # [M, Qu, C]
        return z_h.reshape(B, N, -1, self.head_dim), chan_attn.reshape(B, N, -1, C), used_ids

    def forward(self, x, coords, time_idx=None, bool_masked_pos=None, valid_channels=None):
        """
        x: [B, C, N, L], coords: [B, C, 3]
        bool_masked_pos: [B, C, N] bool — None during the Tokenizer stage (no masking);
        pass real masks only in the Masked stage, once temporal/spatial mixing are enabled
        and the stamps are frozen (see enable_temporal/enable_spatial/freeze_stamps).
        valid_channels: [B, C] bool, True = real (not zero-padded) channel (optional).
        returns SimpleNamespace(recon [B,C,N,L], attn [B,N,n_stamps,C] (per-stamp
        topography), h [M,Q] selection strengths, dense_routed [M,n_routed_stamps]
        (diagnostic selection-frequency source), aux_loss scalar, stamp_lb_loss scalar
        (StampBank's own routed-pool load-balance term, see StampBank.forward), ffn_lb_loss
        scalar (TSABlock MoEFFN routers, summed across blocks — a separate MoE/loss).
        """
        B, C, N, L = x.shape

        z, ffn_lb_loss = self.stage_features(x, coords, time_idx=time_idx, bool_masked_pos=bool_masked_pos)  # [B, C, N, D]
        z_bnc, valid_mask = self._pool_channels(z, valid_channels)  # [M, C, D]
        x_target = self._flatten_patches(x)  # [M, C, L] — aux-rescue target, see StampBank.forward

        out = self.stamps(z_bnc, x_target=x_target, valid_mask=valid_mask)

        recon = out.recon.reshape(B, N, C, L).permute(0, 2, 1, 3)  # [B, C, N, L]

        return SimpleNamespace(
            recon=recon,
            attn=out.attn.reshape(B, N, self.n_stamps, C),
            h=out.h,
            dense_routed=out.dense_routed,
            aux_loss=out.aux_loss,
            stamp_lb_loss=out.stamp_lb_loss,
            ffn_lb_loss=ffn_lb_loss,
            ffn_router_entropy=self.encoder.last_ffn_router_entropy,
            ffn_router_load_std=self.encoder.last_ffn_router_load_std,
            ffn_gate_entropy=self.encoder.last_ffn_gate_entropy,
            decorr_loss=out.decorr_loss,
        )

    @staticmethod
    def _patch_pyramid_levels(recon, x):
        """Pool the patch axis N by successive halving, keeping the within-patch axis L
        intact: level 0 = 1 group of N patches averaged together (-> [B,C,1,L], the
        trial's average patch shape), ..., last level = win=1 (every patch its own group,
        L untouched) which is numerically identical to the raw (recon, x) pair. Returns
        list of (recon_level, x_level) coarsest-first, finest ([recon, x] themselves) last.
        """
        B, C, N, L = x.shape
        # Merge B,C (adjacent, safe) — the one unavoidable copy, since recon arrives
        # already permuted upstream (non-contiguous). Everything below then only ever
        # SPLITS the N axis in place (never merges/transposes non-adjacent axes), so each
        # pyramid level is a free reshape + a cheap mean — no per-level permute/avg_pool1d.
        # Must stay .reshape, not .view — recon arrives non-contiguous (permuted upstream).
        # Swapping to .view for a "perf win" hard-crashes here (good); pre-calling
        # .contiguous() on the wrong intermediate shape would silently scramble instead.
        r = recon.reshape(B * C, N, L).float()
        t = x.reshape(B * C, N, L).float()
        levels = []
        win = N
        while True:
            n_groups = N // win
            n_keep = n_groups * win  # avg_pool1d-equivalent: drop a non-dividing remainder
            rp = r[:, :n_keep, :].reshape(B * C, n_groups, win, L).mean(dim=2)
            tp = t[:, :n_keep, :].reshape(B * C, n_groups, win, L).mean(dim=2)
            levels.append((rp.reshape(B, C, n_groups, L), tp.reshape(B, C, n_groups, L)))
            if win == 1:
                break
            win = max(1, win // 2)
        return levels

    def _hierarchical_recon_loss(self, recon, x, bool_masked_pos):
        """Multi-scale MSE pyramid over the patch axis (see _patch_pyramid_levels): every
        level — coarsest trial-average-shape down to the finest per-patch/per-element
        level — is plain unweighted MSE, then summed across levels. masked/unmasked are
        NOT weighted into the loss (no masked_mse_weight/unmasked_mse_weight anymore); the
        finest level's masked-vs-unmasked split is still computed and returned, purely as
        a diagnostic (see MeSAETrainer/logging) — it plays no part in `total`.
        """
        levels = self._patch_pyramid_levels(recon, x)
        losses = [F.mse_loss(r, t) for r, t in levels]

        l_masked, l_unmasked = 1.0, losses[-1]
        if bool_masked_pos is not None:
            r, t = levels[-1]
            mask4    = bool_masked_pos.unsqueeze(-1).expand_as(t)
            unmasked = ~mask4
            l_masked   = F.mse_loss(r[mask4].float(),    t[mask4].float())    if mask4.any()    else r.new_zeros(1).squeeze()
            l_unmasked = F.mse_loss(r[unmasked].float(), t[unmasked].float()) if unmasked.any() else r.new_zeros(1).squeeze()

        # coarsest -> finest, for the plotter (see MeSAETrainer.epoch_metrics)
        self._last_pyramid_levels = [lv.detach().item() for lv in losses]
        return torch.stack(losses).sum(), l_masked, l_unmasked

    def get_loss(self, x, recon, aux_loss, bool_masked_pos=None, aux_weight=0.03, hierarchical_mse_weight=1.0,
                 decorr_loss=None, decorr_weight=0.01,
                 stamp_lb_loss=None, stamp_lb_weight=0.01,
                 ffn_lb_loss=None, ffn_lb_weight=0.01):
        """
        Returns (total, l_masked, l_unmasked).

        Reconstruction term is the hierarchical patch-pyramid MSE (see
        _hierarchical_recon_loss), scaled by hierarchical_mse_weight.

        Tokenizer stage (bool_masked_pos=None): plain full reconstruction, l_masked=1.0
        placeholder (nothing masked yet), aux_loss included so StampBank's dead-atom
        rescue can still train.

        Masked stage (bool_masked_pos given, self.stamps_frozen True by then): aux_loss is
        dropped from the total regardless of the aux_weight argument once the stamps are
        frozen — rescuing a frozen dictionary's dead atoms can't do anything, see
        freeze_stamps().

        decorr_loss (StampBank's spatial-pattern decorrelation) and stamp_lb_loss
        (StampBank's routed-pool load-balance term, see StampBank.forward) are dropped the
        same way and for the same reason once frozen — freeze_stamps() locks the whole
        StampBank, so neither can act on the gradient anymore (see docs/adr/0007,
        docs/adr/0009).

        ffn_lb_loss (MoEFFN routers' load-balance loss, summed across TSABlocks, see
        docs/adr/0008-moe-ffn-for-mesae.md) is added unconditionally, both stages: it comes
        from the encoder, which keeps training through the Masked stage (freeze_stamps()
        never locks the encoder).
        """
        recon_loss, l_masked, l_unmasked = self._hierarchical_recon_loss(recon, x, bool_masked_pos)
        total = hierarchical_mse_weight * recon_loss

        if bool_masked_pos is None or not self.stamps_frozen:
            total = total + aux_weight * aux_loss
            if decorr_loss is not None:
                total = total + decorr_weight * decorr_loss
            if stamp_lb_loss is not None:
                total = total + stamp_lb_weight * stamp_lb_loss
        if ffn_lb_loss is not None:
            total = total + ffn_lb_weight * ffn_lb_loss
        return total, l_masked, l_unmasked

    def get_metrics(self, dense_routed=None):
        # dense_routed's h_i=softmax(topk router logits) sums to exactly 1 within each
        # patch's top_k picks, so a batch-wide mean-of-selected is mathematically pinned
        # at 1/top_k regardless of training progress — not a real diagnostic (the old
        # h=exp(-self-recon-mse) had no such normalization constraint, so its mean/std
        # genuinely varied; this doesn't survive the switch to a plain linear scorer, see
        # StampBank.forward). Selection-sharpness is still covered properly by
        # stamp_gate_entropy below (entropy of the distribution shape, not its mean).
        metrics = {}
        metrics['dead_feature_rate'] = (self.stamps.fire_ema < self.stamps.dead_threshold).float().mean().item()

        # U-Net skip gate(s) on the encoder's residual-add path: sigmoid(g) in [0,1],
        # 0 = drop skip, 1 = plain add (same convention as MeFSQ.get_metrics).
        if self.encoder.skip_gates is not None:
            for i, g in enumerate(self.encoder.skip_gates):
                metrics[f'skip_gate_{i}'] = torch.sigmoid(g).item()

        # Per-block contribution norm — direct measure of how much each encoder block
        # actually changes its input (not the skip-gate proxy above, which conflates
        # "shallow skip re-injected" with "deep processing did nothing"). Only populated
        # after an eval-mode forward pass (validate_one_epoch), same convention as the
        # other diagnostics that gate on `not self.training`.
        block_norms = getattr(self.encoder, 'last_block_norms', None)
        if block_norms:
            for i, v in enumerate(block_norms):
                metrics[f'block_norm_{i}'] = v

        block_input_norms = getattr(self.encoder, 'last_block_input_norms', None)
        if block_input_norms:
            for i, v in enumerate(block_input_norms):
                metrics[f'block_input_norm_{i}'] = v
            # Relative change (block_norm / incoming x norm) — raw block_norm conflates a
            # block's own contribution with norm_out's per-block learned gain difference
            # from whatever scale x drifted to upstream (see TSABlock.norm_out's
            # docstring), so a block can show a huge raw delta while its LayerScale params
            # stay near-zero. The ratio isolates "how much this block reshaped its input
            # relative to that input's own scale", not a renorm-gain artifact.
            # NOT prefixed block_norm_* -- indexed_series('block_norm_', ...) would swallow
            # these into the raw-delta panel above via startswith prefix matching.
            for i, (bn, bin_) in enumerate(zip(block_norms, block_input_norms)):
                metrics[f'block_relnorm_{i}'] = bn / (bin_ + 1e-8)

        # Stamp router health — see update_stamp_router_metrics above for what each number
        # means.
        metrics['stamp_router_entropy']  = self.ema_stamp_router_entropy.item()
        metrics['stamp_router_load_std'] = self.ema_stamp_router_load_std.item()
        metrics['stamp_gate_entropy']    = self.ema_stamp_gate_entropy.item()

        # FFN router health — see update_ffn_router_metrics above; same 3 metrics, distinct
        # MoE (per-TSABlock MoEFFN routers, averaged across blocks, not the StampBank router).
        metrics['ffn_router_entropy']  = self.ema_ffn_router_entropy.item()
        metrics['ffn_router_load_std'] = self.ema_ffn_router_load_std.item()
        metrics['ffn_gate_entropy']    = self.ema_ffn_gate_entropy.item()

        return metrics


class MeSAEFinetune(nn.Module):
    """
    Wraps a pretrained MeSAEPretrain backbone (unmodified) with a temporal+stamp
    attention classification head (PerChannelHeadAttn) — same shape/rationale as
    MeFSQFinetune (model/MeFSQ/MeFSQ.py). Reads backbone.encode_post_stamp_expert: the
    pre-generator D-dim view for each selected stamp (`h_i * pooled_i`, BEFORE that
    stamp's own generator MLP ever runs — see docs/adr/0009). The channel dim is already
    collapsed by the backbone's own per-stamp channel-attention pool, so the head only
    pools over patches and stamps, not channels.
    """
    def __init__(self, backbone: MeSAEPretrain, num_channels, num_classes, hidden=128, freeze_backbone=False,
                 dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.head = PerChannelHeadAttn(backbone.head_dim, num_classes, dropout=dropout)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x, coords, time_idx=None, valid_channels=None, pad_mask=None):
        """
        x: [B, C, N, L]
        coords: [B, C, 3]
        valid_channels: [B, C] bool, True = real (not zero-padded) channel (optional)
        pad_mask: [B, N] bool, True = valid patch (optional, for padded trailing time)
        returns: (logits [B, num_classes], attn_h [B, Q], attn_n [B, Q, N])
        """
        z_per_head = self.backbone.encode_post_stamp_expert(
            x, coords, time_idx=time_idx, valid_channels=valid_channels)  # [B, N, Q, D]
        logits, attn_h, attn_n = self.head(z_per_head, pad_mask=pad_mask)
        return logits, attn_h, attn_n
