import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MeSAE.MeSAE_modules import (SpatialTemporalEmbeddings, TSAEncoder, StampBank,
                                         overlap_add_patches,
                                         spatial_mix, FlatTimePool, LearnedTimePool,
                                         EvokedBranch, phase_advance, StampExtractor, FeatureHead,
                                         resolve_head_config, make_head_checkpoint,
                                         needs_stamp, needs_raw, _normalize_features)


def _ema_update(buf, val, decay=0.99):
    """In-place EMA update, skipped if val is NaN/Inf. A plain `.mul_(decay).add_(val,
    alpha=1-decay)` permanently poisons buf the moment val is ever NaN even once (a single
    bad batch, e.g. a transient fp16-autocast overflow early in training): NaN propagates
    through every future update (decay*NaN + (1-decay)*anything = NaN), so a diagnostic
    stuck NaN for an entire run can trace back to one early outlier batch long since
    recovered from. Skipping non-finite updates lets the EMA keep tracking real values."""
    if torch.isfinite(val):
        buf.mul_(decay).add_(val, alpha=1 - decay)


def _restore_phase(module, incompatible_keys):
    """load_state_dict post-hook: re-apply the checkpoint's phase flags (plain
    attributes, not state). A checkpoint without masked_phase predates the fused run —
    treat it as fully enabled, the old loaders' behavior. Does not freeze anything."""
    legacy = [k for k in incompatible_keys.missing_keys if k.endswith('masked_phase')]
    for k in legacy:
        incompatible_keys.missing_keys.remove(k)
    if legacy or bool(module.masked_phase):
        module.enter_masked_phase(freeze_stamps=False)
    else:
        module.enter_tokenizer_phase()


class MeSAEPretrain(nn.Module):
    """
    Spatiotemporal stamp-dictionary EEG tokenizer — parallel to MeFSQPretrain, not a
    variant of it. Goal is explainable, per-patch embeddings (not a discrete vocabulary):
    channel-count invariant (cross-dataset unification still matters) but NOT
    length-invariant (each patch keeps its own embedding, for temporal localization of
    events within a trial).

    Pipeline: encoder -> StampBank (dictionary of fixed per-atom waveform templates, each
    presented at a per-channel, per-atom amplitude/phase read off that atom's own
    bottleneck) -> reconstruction, summed directly in patch space (no separate decoder
    stage). See docs/adr/0009-spatiotemporal-stamp-dictionary-for-mesae.md for the full
    derivation and docs/adr/0007-routed-filter-gating-for-mesae.md for the routed/shared
    split.

    Trains in two phases of one run (train_pretrain.py; docs/adr/0013, CONTEXT.md:
    Tokenizer stage / Masked stage):
    - enter_tokenizer_phase(): only the pool_after_blocks blocks run, temporal mixing
      only, no masking (bool_masked_pos=None) — encoder + StampBank train jointly on
      single-channel features, so the stamp dictionary isn't built from
      cross-channel-mixed input.
    - enter_masked_phase(freeze_stamps): every block runs, spatial attention + coord
      embedding on, bool_masked_pos set. StampBank optionally frozen.
    The phase is a buffer, so any load_state_dict restores it (_restore_phase).
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
        n_routed_stamps=796,
        n_shared_stamps=4,
        stamp_top_k=32,
        stamp_hidden_width=8,
        stamp_shared_hidden_width=16,
        dead_threshold_frac=0.1,
        stamp_ema_decay=0.999,
        stamp_amp_levels=None,
        stamp_phase_levels=None,
        stamp_amp_log2_range=(-6.0, 3.0),
        stamp_selection_mode='topk',
        stamp_aux_k_cap_frac=None,
        n_routed_ffn_experts=4,
        n_shared_ffn_experts=1,
        ffn_top_k=2,
        patch_stride=None,
    ):
        super().__init__()
        self.patch_len = patch_len
        # patch_stride drives overlap-add stitching in _recon_loss's trial term.
        # Falls back to patch_len (non-overlapping) matching PretrainDataset.
        self.patch_stride = patch_stride or patch_len
        self.head_dim = embed_dim
        self.num_channels = num_channels

        self.embed   = SpatialTemporalEmbeddings(patch_len, embed_dim)
        self.encoder = TSAEncoder(embed_dim, depth=enc_depth, num_heads=spatial_heads, mlp_ratio=mlp_ratio,
                                   dropout=dropout, pool_after_blocks=pool_after_blocks,
                                   n_routed_ffn_experts=n_routed_ffn_experts, n_shared_ffn_experts=n_shared_ffn_experts,
                                   ffn_top_k=ffn_top_k)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.stamps = StampBank(
            embed_dim, patch_len,
            n_routed_stamps=n_routed_stamps, n_shared_stamps=n_shared_stamps, top_k=stamp_top_k,
            hidden_width=stamp_hidden_width, shared_hidden_width=stamp_shared_hidden_width,
            dead_threshold_frac=dead_threshold_frac, ema_decay=stamp_ema_decay,
            amp_levels=stamp_amp_levels, phase_levels=stamp_phase_levels,
            amp_log2_range=stamp_amp_log2_range,
            selection_mode=stamp_selection_mode,
            aux_k_cap_frac=stamp_aux_k_cap_frac,
        )
        # convenience aliases — viz/checker code reads these off the model directly
        # (e.g. base_checker.py compute_unit_colors).
        self.n_stamps = self.stamps.n_stamps
        self.n_routed_stamps = self.stamps.n_routed
        self.n_shared_stamps = self.stamps.n_shared
        self.stamps_frozen = False
        self.register_buffer('masked_phase', torch.tensor(False))
        self.register_load_state_dict_post_hook(_restore_phase)

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

    def enable_coord_embed(self):
        """Coordinate embedding ONLY — each channel's token learns WHERE it is, with no
        cross-channel content mixing (that is enable_spatial's MHA half, below). Kept as
        a standalone manual/experimental toggle (no longer auto-called anywhere in the
        training path: it measurably did nothing during the Tokenizer stage, no matter how the embedding's
        own architecture was fixed, because per-channel content already differentiates
        channels enough for that stage's loss without it).

        Was originally meant to let amp_i(z_c) become position-aware even while stamps
        stay single-channel (StampBank can't otherwise tell "alpha at Oz" from "alpha at
        Fz" when the raw content happens to coincide) — a real idea, just not one the
        Tokenizer stage's own reconstruction objective rewards learning. enable_spatial
        (below) now enables this alongside cross-channel attention in the Pretrain
        stage instead, where position could plausibly matter for attending across
        channels."""
        self.embed.enable_spatial()

    def enable_spatial(self):
        self.embed.enable_spatial()
        self.encoder.enable_spatial()

    def enable_temporal(self):
        self.encoder.enable_temporal()

    def enter_tokenizer_phase(self):
        self.masked_phase.fill_(False)
        self.encoder.active_blocks = set(self.encoder.pool_after_blocks) or None
        self.enable_temporal()

    def enter_masked_phase(self, freeze_stamps=True):
        """Freeze before enable_spatial: a frozen dictionary never sees mixed z. With
        freeze_stamps=False it trains on mixed z, so per-stamp amp is no longer a
        source topomap (aux_loss/mp_loss stay on — get_loss gates them on stamps_frozen)."""
        self.masked_phase.fill_(True)
        if freeze_stamps:
            self.freeze_stamps()
        self.encoder.active_blocks = None
        self.enable_temporal()
        self.enable_spatial()

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
        MeFSQ.update_head_metrics, adapted for StampBank's raw selection strengths:
        post-rms amp magnitude, not a softmax (see StampBank.forward). h_routed_dense:
        [M, n_routed_stamps] (StampBank's `dense_routed`, zeros at unselected).

        stamp_router_entropy: entropy of the routed pool's LOAD distribution (how evenly,
        across this batch's patches, selection is spread over the n_routed_stamps routed
        stamps) — 0 = every patch always picks the same stamp (total collapse),
        log(n_routed_stamps) = perfectly uniform load. Rising over training = healthy
        (stamps differentiating and each still getting used); falling toward 0 = router
        collapse (a couple of stamps absorbing everything, see docs/adr/0007).

        stamp_gate_entropy: entropy of the WITHIN-patch selection strengths (not across
        patches). `h_routed_dense` is raw and unbounded — not a probability distribution
        — so it's renormalized per-row (`/ sum`) here purely for this diagnostic, never
        touching the actual reconstruction path. 0 = one selected stamp dominates that
        patch's strength (confident/peaked selection), log(top_k) = the k selected
        stamps split strength near-uniformly.

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

    # -- Finetune-only entry points, NOT used by the Tokenizer/Pretrain forward() path
    # below.

    def encode_post_stamp_expert(self, x, coords, time_idx=None, valid_channels=None, return_chan_attn=False):
        """Per-stamp channel View for the finetune StampExtractor: unlike MeFSQ's Experts (already
        channel-free via ExpertChannelPool before quantization), a StampBank stamp's
        response is inherently per-channel — its amp IS a topomap. Collapse channels
        here the same way, but for free: pool z's C channels for stamp i weighted by
        that stamp's OWN per-channel amp magnitude (softmax over C), instead of a
        learned query. Zero new params, and the weight is the physically meaningful
        quantity already (a stamp's mixing/topomap column) rather than something a
        classifier head would have to learn from scratch and risk overfitting on (see
        docs/agents/ / CONTEXT.md finetune val-chance bug).

        Uses dense_amp (every atom, no top-k) rather than the reconstruction path's
        selected top_k+n_shared: reconstruction sparsity optimizes what's needed to
        rebuild the signal, not what's discriminative for classification, and a dense
        axis gives every stamp a stable identity across patches for free (no
        zero-dilution bookkeeping needed).

        Returns z_per_head [B, N, n_stamps, D] (the per-stamp view a finetune head consumes),
        plus chan_attn [B, N, n_stamps, C] (the pooling weights, i.e. each stamp's
        per-patch topomap) if return_chan_attn=True.
        """
        z, _ = self.stage_features(x, coords, time_idx=time_idx)  # [B, C, N, D]
        B, C, N, D = z.shape
        G = B * N
        z_g = z.permute(0, 2, 1, 3).reshape(G, C, D)   # [G, C, D]
        x_g = x.permute(0, 2, 1, 3).reshape(G, C, -1)  # [G, C, L]
        rms = x_g.pow(2).mean(dim=-1, keepdim=True).sqrt()  # [G, C, 1]

        amp = self.stamps.dense_amp(z_g, rms=rms)      # [G, C, n_stamps, 2]
        mag = amp.pow(2).sum(dim=-1).sqrt()            # [G, C, n_stamps]

        if valid_channels is not None:
            vc_g = valid_channels.unsqueeze(1).expand(B, N, C).reshape(G, C)
            mag = mag.masked_fill(~vc_g.unsqueeze(-1), float('-inf'))

        chan_attn = torch.softmax(mag, dim=1)          # [G, C, n_stamps] — softmax over C, per stamp
        chan_attn = torch.nan_to_num(chan_attn)        # guards an all-padded channel set, shouldn't occur in practice
        z_per_head = torch.einsum('gcn,gcd->gnd', chan_attn, z_g)  # [G, n_stamps, D]
        z_per_head = z_per_head.view(B, N, self.n_stamps, D)

        if return_chan_attn:
            # chan_attn is [G, C, n_stamps] — permute to [G, n_stamps, C] before
            # splitting G, or view() silently swaps C and n_stamps instead of
            # transposing them (docs/agents/reshape-pitfalls.md).
            chan_attn = chan_attn.permute(0, 2, 1).reshape(B, N, self.n_stamps, C)
            return z_per_head, chan_attn
        return z_per_head

    def used_stamp_ids(self, out, max_stamps=100):
        """Global stamp ids actually selected SOMEWHERE across this batch (a batch built
        from one trial's patches, in practice — see check_pretrain/check_finetune), capped
        at max_stamps, ranked by accumulated selection strength. `out` needs `dense_routed`
        (from this model's own `forward`/`stamps(...)` output — any SimpleNamespace with
        that field works). Shared stamps always included first (constant weight, always
        selected every patch, so cheap to guarantee) — remaining budget filled by the
        highest-usage routed stamps, dropping ones that never fired at all this batch. This
        exists because hard top-k selection means `forward()`'s per-patch idx/dense_routed
        axis has NO stable cross-patch identity (patch A's slot 0 and patch B's slot 0 can
        be different physical stamps) — a trial-wide view needs a fixed, shared set of
        global ids instead. (Finetune's encode_used_stamps solves the same display-size
        problem a different way — see its docstring — since it has no top-k axis to
        begin with.)
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
        """Viz-only convenience over encode_post_stamp_expert: same z_per_head/chan_attn,
        capped to the max_stamps stamps with the largest trial-summed View magnitude —
        at n_stamps up to a few hundred, rendering every one regardless of relevance
        would swamp the panels (a viz-only concern). Ranked by real magnitude
        here, not selection frequency: unlike the Tokenizer/Pretrain path, nothing here
        goes through top-k, so there's no "selected" notion to rank by in the first
        place. Returns (z_per_head [B, N, Qu, D], chan_attn [B, N, Qu, C],
        used_ids [Qu])."""
        z_per_head, chan_attn = self.encode_post_stamp_expert(
            x, coords, time_idx=time_idx, valid_channels=valid_channels, return_chan_attn=True)
        importance = z_per_head.norm(dim=-1).sum(dim=(0, 1))  # [n_stamps] — ranking only
        used_ids = torch.argsort(importance, descending=True)[:max_stamps]
        return z_per_head[:, :, used_ids, :], chan_attn[:, :, used_ids, :], used_ids

    def forward(self, x, coords, time_idx=None, bool_masked_pos=None, valid_channels=None):
        """
        x: [B, C, N, L], coords: [B, C, 3]
        bool_masked_pos: [B, C, N] bool — None during the Tokenizer stage (no masking);
        pass real masks only in the Masked stage, once temporal/spatial mixing are enabled
        and the stamps are frozen (see enable_temporal/enable_spatial/freeze_stamps).
        valid_channels: [B, C] bool, True=real (not zero-padded) channel, or None. Used
        two ways now: (1) passed into StampBank so a padded channel's encoder-bias amp
        noise doesn't vote in the per-patch group selection score (see
        StampBank.forward), and (2) carried through on the returned SimpleNamespace so
        get_loss/_recon_loss can exclude padded channels from the loss (see get_loss) —
        a zero-padded channel's "reconstruction" is meaningless signal, not a real
        target. Padded channels still decode/reconstruct like any other.
        returns SimpleNamespace(recon [B,C,N,L], h [G,Q] selection confidences,
        dense_routed [G,n_routed_stamps] (diagnostic selection-frequency source, G =
        B*N patch positions — group-level selection, see StampBank), aux_loss scalar,
        ffn_lb_loss scalar (TSABlock MoEFFN routers,
        summed across blocks — StampBank has no load-balance loss of its own, see
        StampBank.forward). No `attn` anymore — there is no cross-channel pool left to
        produce a channel-attention map from.
        """
        B, C, N, L = x.shape

        z, ffn_lb_loss = self.stage_features(x, coords, time_idx=time_idx, bool_masked_pos=bool_masked_pos)  # [B, C, N, D]
        # Channel-grouped layout for StampBank: [B, C, N, *] -> permute to [B, N, C, *]
        # then merge (B, N) — adjacent after the permute, so the merge is a safe reshape
        # (docs/agents/reshape-pitfalls.md; permute forces a copy, contiguity handled by
        # reshape itself). One group = one patch position with all its channels — the
        # unit StampBank selects stamps for (see its class docstring).
        G = B * N
        z_g = z.permute(0, 2, 1, 3).reshape(G, C, -1)          # [G, C, D]
        x_g = x.permute(0, 2, 1, 3).reshape(G, C, L)           # [G, C, L] — aux-rescue target

        # Per-channel raw-input RMS — the amplitude signal the LayerNorm stack erased
        # from z (embed.norm -> per-block norm_out -> stamps.input_norm), multiplied back
        # into every amp inside StampBank. Masked positions get 1.0: their true patch is
        # hidden from the encoder, so feeding its RMS would leak the target's amplitude
        # into masked reconstruction — the encoder must predict a masked patch's loudness
        # through z, same as it always did.
        rms = x_g.pow(2).mean(dim=-1, keepdim=True).sqrt()     # [G, C, 1]

        vc_g = None
        if valid_channels is not None:
            # [B, C] -> broadcast over N -> [G, C]; group score should only count real
            # channels' amp energy (see StampBank.forward's valid_channels docstring).
            vc_g = valid_channels.unsqueeze(1).expand(B, N, C).reshape(G, C)

        if bool_masked_pos is not None:
            mask_g = bool_masked_pos.permute(0, 2, 1).reshape(G, C, 1)
            # Masking is generated channel-agnostic (IO/masking.py never sees
            # valid_channels), so a padded channel's patch can land inside the mask —
            # gate the override on valid-AND-masked, not masked alone, or a padded
            # channel's rms gets force-set to 1.0 here, fabricating a nonzero amp/recon
            # for a channel that's always exactly 0. get_loss already excludes padded
            # channels from the main loss, but the dead-atom aux rescue
            # (StampBank.forward's aux_loss) has no valid_channels masking at all, so
            # that fabricated signal would otherwise leak straight into a revived
            # atom's decoder weights — shared across every channel, real ones included.
            if vc_g is not None:
                mask_g = mask_g & vc_g.unsqueeze(-1)
            rms = torch.where(mask_g, torch.ones_like(rms), rms)

        # target_visible: which channels' x_target selection_mode='gain' may read. An
        # UNMASKED channel's content is the model's own input (no leak); a MASKED one is
        # the answer. Per (position, channel), not all-or-nothing -- see StampBank.forward.
        target_visible = None
        if bool_masked_pos is not None:
            target_visible = (~bool_masked_pos).permute(0, 2, 1).reshape(G, C)
        out = self.stamps(z_g, x_target=x_g, rms=rms, valid_channels=vc_g,
                           target_visible=target_visible)

        recon = out.recon.reshape(B, N, C, L).permute(0, 2, 1, 3)  # back to [B, C, N, L]

        return SimpleNamespace(
            recon=recon,
            h=out.h,
            dense_routed=out.dense_routed,
            aux_loss=out.aux_loss,
            mp_loss=out.mp_loss,
            # [B, C, N], same layout as bool_masked_pos (G = B*N rows were b*N + n)
            mp_map=None if out.mp_map is None else out.mp_map.reshape(B, N, C).permute(0, 2, 1),
            ffn_lb_loss=ffn_lb_loss,
            ffn_router_entropy=self.encoder.last_ffn_router_entropy,
            ffn_router_load_std=self.encoder.last_ffn_router_load_std,
            ffn_gate_entropy=self.encoder.last_ffn_gate_entropy,
            k_eff=out.k_eff,
            valid_channels=valid_channels,
            # None unless stamp quantization is configured. Carried up from StampBank
            # rather than left at its boundary: quant_clip_frac/quant_off_frac are the
            # only visibility into a mis-set amp_log2_range (neither shows up in the
            # loss), so the trainer's epoch metrics and the checkers have to be able to
            # read them. levels [G, C, K, 2] is the discrete code itself.
            levels=out.levels,
            quant_clip_frac=out.quant_clip_frac,
            quant_off_frac=out.quant_off_frac,
        )

    @staticmethod
    def _position_weights(x, bool_masked_pos, valid_channels, unmasked_weight):
        """-> (valid, w), each [B, C, N, 1]. valid: 1 on real channels. w: the training
        weight -- valid, times (1 on masked, unmasked_weight on visible) in the masked
        phase. Shared by the recon MSE and mp_loss so both weight positions the same."""
        B, C, N = x.shape[:3]
        valid = x.new_ones(B, C, 1, 1) if valid_channels is None \
            else valid_channels.view(B, C, 1, 1).to(x.dtype)
        valid = valid.expand(B, C, N, 1)
        if bool_masked_pos is None:
            return valid, valid
        m = bool_masked_pos.unsqueeze(-1).to(x.dtype)
        return valid, valid * (m + unmasked_weight * (1.0 - m))

    def _recon_loss(self, recon, x, bool_masked_pos, valid_channels=None,
                    mse_patch_weight=1.0, mse_trial_weight=1.0, unmasked_weight=1.0):
        """Two-term recon loss, both plain time-domain MSE:
        - patch: every raw patch against its own reconstruction.
        - trial: MSE on the REAL continuous trial, patches overlap-added back together
          (overlap_add_patches), so gradient reaches every patch through its real
          position in the trial. patch was once spectrally whitened, and a "window"
          level existed between them; both dropped, see docs/adr/0011.

        Position weights (both terms): padded channels 0; in the masked phase masked
        positions 1 and visible positions `unmasked_weight` (0 = standard MAE, loss on
        masked patches only; 1 = every position counts equally). The trial term gets
        the same weights overlap-added to samples. With bool_masked_pos=None (tokenizer
        phase) every valid position is a target and unmasked_weight is unused.

        Logged mse_patch/mse_trial stay the plain all-valid-position MSE, comparable
        across phases whatever the weights. masked/unmasked are a diagnostic split.
        Returns (weighted total, l_masked, l_unmasked).
        """
        B, C, N, L = x.shape
        stride = self.patch_stride
        recon, x = recon.float(), x.float()
        valid, w = self._position_weights(x, bool_masked_pos, valid_channels, unmasked_weight)

        def wmean(err, wt):
            wt = wt.expand_as(err)
            s = wt.sum()
            return (wt * err).sum() / s if s > 0 else err.new_zeros(())

        def to_trial(wt):  # per-patch weight -> per-sample weight, same crossfade as recon
            return overlap_add_patches(wt.expand(B, C, N, L), stride)

        patch_err = (recon - x).pow(2)
        trial_err = (overlap_add_patches(recon, stride) - overlap_add_patches(x, stride)).pow(2)  # [B, C, T]
        total = mse_patch_weight * wmean(patch_err, w) + mse_trial_weight * wmean(trial_err, to_trial(w))

        with torch.no_grad():
            plain_patch = wmean(patch_err, valid)
            plain_trial = wmean(trial_err, to_trial(valid))
            l_masked, l_unmasked = 1.0, plain_patch
            if bool_masked_pos is not None:
                m = bool_masked_pos.unsqueeze(-1).float()
                l_masked = wmean(patch_err, valid * m)
                l_unmasked = wmean(patch_err, valid * (1.0 - m))
        # Named so the log/dashboard keys read mse_patch/mse_trial (train_pretrain.py
        # reads _last_pyramid_levels; plugin.py matches the 'mse_' prefix).
        self._last_pyramid_levels = {'patch': plain_patch.item(), 'trial': plain_trial.item()}
        return total, l_masked, l_unmasked

    def get_loss(self, x, recon, aux_loss, bool_masked_pos=None, aux_weight=0.03,
                 mse_patch_weight=1.0, mse_trial_weight=1.0, unmasked_weight=1.0,
                 ffn_lb_loss=None, ffn_lb_weight=0.01, valid_channels=None,
                 mp_loss=None, mp_weight=0.0, mp_map=None):
        """
        Returns (total, l_masked, l_unmasked).

        mp_loss/mp_weight: optional Matching-Pursuit-style residual loss (see
        StampBank.forward's mp_loss section) — always computed there (cheap relative
        to the rest of the forward pass) but only added to total when mp_weight != 0,
        same "off by default, zero cost when off" convention as everything else here.

        Reconstruction term is _recon_loss's weighted patch + trial MSE (see its
        docstring for mse_patch_weight / mse_trial_weight / unmasked_weight).

        Tokenizer stage (bool_masked_pos=None): plain full reconstruction, l_masked=1.0
        placeholder (nothing masked yet), aux_loss included so StampBank's dead-atom
        rescue can still train.

        Masked stage (bool_masked_pos given, self.stamps_frozen True by then): aux_loss is
        dropped from the total regardless of the aux_weight argument once the stamps are
        frozen — rescuing a frozen dictionary's dead atoms can't do anything, see
        freeze_stamps().

        Dictionary shaping is mp_loss's job (see StampBank.forward): top_k is the
        sparsity budget, aux_loss the anti-collapse mechanism, and mp_loss the
        residual-ordered term that stops atoms being rewarded for re-explaining what a
        higher-ranked atom already covered. Earlier attempts at this — spectral
        whitening, then activation decorrelation/negentropy — were measured and
        dropped; see docs/adr/0011. StampBank has no load-balance loss of its own.

        ffn_lb_loss (MoEFFN routers' load-balance loss, summed across TSABlocks, see
        docs/adr/0008-moe-ffn-for-mesae.md) is added unconditionally, both stages: it comes
        from the encoder, which keeps training through the Masked stage (freeze_stamps()
        never locks the encoder).
        """
        total, l_masked, l_unmasked = self._recon_loss(
            recon, x, bool_masked_pos, valid_channels=valid_channels,
            mse_patch_weight=mse_patch_weight, mse_trial_weight=mse_trial_weight,
            unmasked_weight=unmasked_weight)

        if bool_masked_pos is None or not self.stamps_frozen:
            total = total + aux_weight * aux_loss
        if ffn_lb_loss is not None:
            total = total + ffn_lb_weight * ffn_lb_loss

        # Gated on stamps_frozen exactly like aux_loss above, same reasoning: mp_loss
        # exists to shape WHICH atom owns which content (see StampBank.forward's
        # mp_loss section), and a frozen dictionary's atoms can't be reshaped. Gradient
        # would still reach the (never-frozen) encoder through amp=f(z), but pushing the
        # encoder to make greedy residual decomposition easier is not the Masked stage's
        # job — that stage optimizes masked reconstruction, and leaving this on would
        # quietly add a second, unrelated objective to it.
        if mp_loss is not None and mp_weight and (bool_masked_pos is None or not self.stamps_frozen):
            if mp_map is not None:
                # Same position weights as the recon MSE: without this, mp_loss counts
                # visible patches at full weight and undoes unmasked_weight.
                _, w = self._position_weights(x, bool_masked_pos, valid_channels, unmasked_weight)
                w = w[..., 0].to(mp_map.dtype)
                s = w.sum()
                mp_train = (w * mp_map).sum() / s if s > 0 else mp_map.new_zeros(())
            else:
                mp_train = mp_loss
            total = total + mp_weight * mp_train
            # logged value stays the plain valid-channel mean, comparable across phases
            self._last_pyramid_levels['mp'] = mp_loss.detach().item()
        return total, l_masked, l_unmasked

    def get_metrics(self, dense_routed=None):
        # dense_routed param is currently unused here (kept for call-site symmetry with
        # MeFSQ's get_metrics) — its old rationale doesn't apply anymore either: it used
        # to be h_i=softmax(topk router logits), pinned to sum to 1 within each patch's
        # top_k picks (so a batch-wide mean-of-selected was mathematically stuck at
        # 1/top_k, not a real diagnostic). h is now raw post-rms amp magnitude (see
        # StampBank.forward) with no such constraint, so a real mean/std would be
        # meaningful again if this ever gets wired up. Selection-sharpness is covered
        # properly by stamp_gate_entropy below regardless (entropy of the distribution
        # shape, not its mean).
        metrics = {}
        metrics['dead_feature_rate'] = (self.stamps.fire_ema < self.stamps.dead_threshold).float().mean().item()

        # U-Net skip gate(s) on the encoder's residual-add path: sigmoid(g) in [0,1],
        # 0 = drop skip, 1 = plain add (same convention as MeFSQ.get_metrics).
        if self.encoder.skip_gates is not None:
            for i, g in enumerate(self.encoder.skip_gates):
                metrics[f'skip_gate_{i}'] = torch.sigmoid(g).item()

        # Per-block contribution norm — direct measure of how much each encoder block
        # actually changes its input (not the skip-gate proxy above, which conflates
        # "shallow skip re-injected" with "deep processing did nothing"). For a block
        # right before a pool point, this already has that block's own gate folded in
        # (see TSAEncoder.forward) — its real surviving contribution, not just its raw
        # pre-gate delta. Only populated after an eval-mode forward pass
        # (validate_one_epoch), same convention as the other diagnostics that gate on
        # `not self.training`.
        block_norms = getattr(self.encoder, 'last_block_norms', None)
        if block_norms:
            for i, v in enumerate(block_norms):
                metrics[f'block_norm_{i}'] = v

        # Largest branch output magnitude anywhere in the encoder, measured BEFORE
        # LayerScale shrinks it (see TSABlock._watch). The norms bound each branch's input
        # and each block's output; nothing bounds the middle, and LayerScale hides it from
        # block_norm. Early warning: this climbs for epochs before a float ceiling is
        # actually crossed, while every other diagnostic still looks healthy — v13 died
        # that way. Judge it against the training dtype's ceiling (65504 for fp16); order
        # 1-10 is normal, hundreds means the next run dies whatever the loss curve says.
        # No separate headroom fraction: it is this number over a constant, and it rounds
        # to 0.0000 in the log across the entire healthy range.
        branch_max = getattr(self.encoder, 'last_branch_max', None)
        if branch_max is not None:
            metrics['branch_max'] = branch_max.item()

        # Stamp router health — see update_stamp_router_metrics above for what each number
        # means.
        metrics['stamp_router_entropy']  = self.ema_stamp_router_entropy.item()
        # Same number as a FRACTION OF ITS OWN MAXIMUM, log(n_routed_stamps). The raw
        # entropy above is in nats and its ceiling moves with the pool size (4.79 at 120
        # routed vs 5.70 at 298), so raw values are not comparable across runs — 4.53 and
        # 3.32 look far apart but are 0.80 and 0.81 of max, i.e. equally healthy.
        # Measured reference band from real runs: 0.73-0.81 is healthy (v4/v5/v6, alive
        # 0.54-0.97), while v8's pool collapse read 0.53 (alive 0.19). A hinged entropy
        # FLOOR at ~0.70 is the documented safe shape if prevention is ever needed
        # (see StampBank's note on why plain load-balancing is not: since |amp|
        # self-selection the score IS the reconstruction coefficient, so pushing the
        # load uniform pushes reconstruction amplitudes uniform).
        metrics['stamp_router_entropy_frac'] = (
            self.ema_stamp_router_entropy.item() / math.log(max(self.n_routed_stamps, 2)))
        metrics['stamp_router_load_std'] = self.ema_stamp_router_load_std.item()
        metrics['stamp_gate_entropy']    = self.ema_stamp_gate_entropy.item()

        # FFN router health — see update_ffn_router_metrics above; same 3 metrics, distinct
        # MoE (per-TSABlock MoEFFN routers, averaged across blocks, not the StampBank router).
        metrics['ffn_router_entropy']  = self.ema_ffn_router_entropy.item()
        metrics['ffn_router_load_std'] = self.ema_ffn_router_load_std.item()
        metrics['ffn_gate_entropy']    = self.ema_ffn_gate_entropy.item()

        return metrics


class FinetuneModel(nn.Module):
    """Frozen MeSAE backbone + one FeatureHead (MeSAE_modules finetune section). Call signature
    matches the old finetune classes so train_finetune.py is unchanged: forward -> (logits, None, None)."""
    def __init__(self, backbone, head_cfg, channel_idx):
        super().__init__()
        self.backbone = backbone
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.register_buffer('channel_idx', torch.as_tensor(channel_idx, dtype=torch.long))
        _normalize_features(head_cfg)   # old checkpoints' baked-in head_config: 'feature' -> 'features'
        self.head_cfg = head_cfg
        stamp = needs_stamp(head_cfg)
        self.extractor = StampExtractor(backbone, channel_idx) if stamp else None
        if stamp:
            assert head_cfg['num_stamps'] == len(self.extractor.keep), "num_stamps must equal the alive stamp count"
        self.head = FeatureHead(head_cfg)
        if 'stamp_band' in head_cfg['features']:
            E_D, E_H = self.extractor.band_tables(head_cfg['sample_freq'])
            self.head.entries['stamp_band'].E_D.copy_(E_D)
            self.head.entries['stamp_band'].E_H.copy_(E_H)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()   # frozen: no dropout noise, stable top-k
        return self

    def forward(self, x, coords, time_idx=None, valid_channels=None, pad_mask=None):
        B, C = x.shape[:2]
        vm = valid_channels if valid_channels is not None else x.new_ones(B, C, dtype=torch.bool)
        inp = {}
        with torch.no_grad():
            if self.extractor is not None:
                inp['stamp'] = self.extractor(x, coords, time_idx, vm)
            if needs_raw(self.head_cfg):
                inp['raw'] = x[:, self.channel_idx] * vm[:, self.channel_idx].float()[:, :, None, None]
        return self.head(inp), None, None

    def head_checkpoint(self, backbone_checkpoint):
        return make_head_checkpoint(self.head, self.head_cfg, self.channel_idx.tolist(),
                                    self.extractor.keep.tolist() if self.extractor is not None else None,
                                    backbone_checkpoint)

    @classmethod
    def from_checkpoint(cls, backbone, ckpt):
        cfg = dict(ckpt['head_config'])
        channel_idx, keep = cfg.pop('channel_idx'), cfg.pop('keep')
        model = cls(backbone, cfg, channel_idx)
        if keep is not None and model.extractor.keep.tolist() != keep:
            raise ValueError("backbone's alive stamps differ from the checkpoint's head_config['keep']")
        model.head.load_state_dict(ckpt['model_state_dict'])
        return model


def build_finetune(backbone, num_channels, num_classes, channel_idx=None, num_patches=None,
                   sample_freq=200, **ft_params):
    """finetune_cls entry: builds the frozen backbone + FeatureHead from the numeric head config."""
    channel_idx = list(range(num_channels)) if channel_idx is None else list(channel_idx)
    norm = dict(ft_params)
    _normalize_features(norm)
    num_stamps = 0
    if needs_stamp(norm):
        st = backbone.stamps
        num_stamps = int((st.fire_ema >= st.dead_threshold).sum()) + (st.n_stamps - st.n_routed)
    cfg = resolve_head_config(ft_params, num_classes=num_classes, num_patches=num_patches,
                              num_channels=len(channel_idx), num_stamps=num_stamps,
                              patch_len=backbone.patch_len, patch_stride=backbone.patch_stride,
                              sample_freq=float(sample_freq))
    return FinetuneModel(backbone, cfg, channel_idx)
