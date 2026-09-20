import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MeSAE.MeSAE_modules import (SpatialTemporalEmbeddings, TSAEncoder, StampBank,
                                         PerChannelHeadAttn, overlap_add_patches)


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
        """Per-stamp channel View for MeSAEFinetune: unlike MeFSQ's Experts (already
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

        Returns z_per_head [B, N, n_stamps, D] (feed straight into PerChannelHeadAttn),
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
        would swamp the panels (see render_finetune_attn). Ranked by real magnitude
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


class MeSAEFinetune(nn.Module):
    """
    Wraps a pretrained MeSAEPretrain backbone (unmodified) with a temporal+stamp
    attention classification head (PerChannelHeadAttn) — same shape/rationale as
    the removed MeFSQFinetune. Reads backbone.encode_post_stamp_expert: a
    D-dim View per stamp per patch, every stamp densely (no top-k), channels collapsed
    by pooling z with that stamp's own per-channel amp magnitude as the weight — see
    encode_post_stamp_expert's docstring. The channel dim is already gone by the time
    the head sees it, so the head only pools over patches and stamps.
    """
    def __init__(self, backbone: MeSAEPretrain, num_channels, num_classes, hidden=128, freeze_backbone=False,
                 dropout=0.1, use_topo_feature=False):
        super().__init__()
        self.backbone = backbone
        # use_topo_feature: also hand the head each stamp's per-patch TOPOGRAPHY
        # (chan_attn, the softmax-over-channels weight encode_post_stamp_expert already
        # computes) as a FEATURE, instead of only consuming it as a pooling weight.
        #
        # An earlier comment here claimed the topography beats pooled magnitude on
        # BCICIV2a (0.373 vs 0.255). That came from trial-wise CV and was mostly
        # subject-identity leakage (ADR 0012 §4a); do not rely on it. What survives: a
        # softmax-weighted mean over channels cannot represent a C3-C4 contrast.
        #
        # Off by default -- it widens the head input to 2*head_dim, so it is an
        # architecture change, not a free fix. NOT universal either: on EEGMMIdb the
        # collapsed features win (pool_mag 0.502 vs topography 0.465), so this is
        # expected to help lateralized MI and may not help elsewhere.
        self.use_topo_feature = use_topo_feature
        head_dim = backbone.head_dim
        if use_topo_feature:
            # C -> head_dim so the topography arrives in the same space/width as z, and
            # the two can be concatenated per (patch, stamp) without one dominating the
            # LayerNorm inside the head.
            self.topo_proj = nn.Linear(num_channels, head_dim)
            head_dim = head_dim * 2
        self.head = PerChannelHeadAttn(head_dim, num_classes, dropout=dropout)

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
        if self.use_topo_feature:
            z_per_head, chan_attn = self.backbone.encode_post_stamp_expert(
                x, coords, time_idx=time_idx, valid_channels=valid_channels,
                return_chan_attn=True)                                    # [B,N,Q,D], [B,N,Q,C]
            # Concatenated, not added: the head's input_norm sees one vector per
            # (patch, stamp), and adding would let z's scale bury the topography.
            z_per_head = torch.cat([z_per_head, self.topo_proj(chan_attn)], dim=-1)
        else:
            z_per_head = self.backbone.encode_post_stamp_expert(
                x, coords, time_idx=time_idx, valid_channels=valid_channels)  # [B, N, Q, D]
        logits, attn_h, attn_n = self.head(z_per_head, pad_mask=pad_mask)
        return logits, attn_h, attn_n


class MeSAEFeatureHead(nn.Module):
    """ADR 0014 experiment B: one feature x one pooling choice per axis x a linear readout,
    so each factor can be changed alone. Frozen backbone only.

    input:        raw            the patched signal overlap-added back (no backbone)
                  recon          the backbone's unmasked reconstruction, overlap-added
                  stamp_bandpow  per-patch dense stamp amps (alive routed + shared),
                                 band energy via each template's spectrum (exact per stamp,
                                 see ADR 0014), no decoding
                  stamp_induced  ADR 0014 experiment C, step C0: per-(channel-filter, stamp)
                                 log power, flat (uniform) time weights -- same amp tensor as
                                 stamp_bandpow, kept per-stamp instead of band-collapsed. The
                                 pre-log per-stamp powers span the same information
                                 stamp_bandpow's band-summed powers do, but the linear readout
                                 here runs on log-power, so it is NOT a strict superset of
                                 stamp_bandpow's readout -- log doesn't distribute over the
                                 band sum (log(sum w*p) != sum w*log(p))
                  z_chan         encoder z, shared Linear(D, z_proj), time-mean per channel
    pool_channel: concat | spatial:K   (signed Linear(C, K), no softmax. For stamp_bandpow
                  it mixes each stamp's (a, b) across channels before the power, the
                  code-space analogue of a CSP filter.)
    pool_time:    trial | window:lo-hi (seconds from trial start; patches fully inside) |
                  learned:R (ADR 0014 C1, stamp_induced only -- low-rank softmax-weighted
                  time pooling, R = rank; requires num_patches)
    include_advance: bool, stamp_induced only (ADR 0014 C3) -- concatenates the phase-
                  advance branch (2c) to the induced-power features, tripling feature
                  width (K*S -> K*S*3). No size control beyond dropout is implemented;
                  this is the exact K*S*3 regime ADR 0014's "Size control is mandatory"
                  paragraph warns about, by design -- see the ADR for the reasoning.
    evoked_rank:  int, stamp_induced only, 0 = off (ADR 0014 C4) -- adds the evoked branch
                  (2b): a signed rank-R time filter T[s,n] applied linearly to the complex
                  code (a, b), giving re/im features (+2*K*S width, K*S -> K*S*3 alone).
                  Needs num_patches and the full patch axis (no window: pool_time). Like
                  learned:R and include_advance, depends on fixed-length trials.
    task:         mi (mu/beta log power). erp/ssvep not built yet.

    Padded channels are zeroed before any pooling. pad_mask is ignored: every trial in the
    intra-subject BCICIV2a runs has the same length. `pool_time="learned:R"` DEPENDS on
    that invariant rather than merely tolerating it -- a flat mean degrades gracefully
    over a padded (zeroed) patch, but nothing stops the learned softmax weights from
    concentrating onto a padded position if ever trained on data where that occurs; a
    future variable-length dataset needs real pad-masking added to this branch first.
    `include_advance` depends on the same fixed-length invariant a second, independent
    way: `z_re`/`z_im` are an unnormalized sum over N'-1 patches (unlike the induced-power
    arm, a normalized mean/softmax-weighted mean), so its feature scale grows with trial
    length -- harmless here since every BCICIV2a trial has N'=39, but another reason a
    variable-length dataset needs work before reusing this branch.
    Every trainable module lives under self.head, because train_finetune.py only
    optimizes model.head.
    Returns (logits, None, None) to match MeSAEFinetune's call signature.
    """
    BANDS = ((8.0, 13.0), (13.0, 30.0))

    def __init__(self, backbone: MeSAEPretrain, num_channels, num_classes, input='recon',
                 task='mi', pool_channel='concat', pool_time='trial', z_proj=8,
                 dropout=0.1, sample_freq=200, freeze_backbone=True, num_patches=None,
                 include_advance=False, evoked_rank=0):
        super().__init__()
        if task != 'mi':
            raise NotImplementedError(f"task={task!r}: only 'mi' is built (ADR 0014 build order)")
        if not freeze_backbone:
            raise NotImplementedError("MeSAEFeatureHead assumes a frozen backbone (ADR 0012)")
        if input not in ('raw', 'recon', 'stamp_bandpow', 'stamp_induced', 'z_chan'):
            raise ValueError(f"unknown input {input!r}")
        if include_advance and input != 'stamp_induced':
            raise NotImplementedError(
                f"include_advance is only implemented for input='stamp_induced' "
                f"(ADR 0014 experiment C3), got input={input!r}")
        self.include_advance = include_advance
        if evoked_rank and input != 'stamp_induced':
            raise NotImplementedError(
                f"evoked_rank is only implemented for input='stamp_induced' "
                f"(ADR 0014 experiment C4), got input={input!r}")
        if evoked_rank and num_patches is None:
            raise ValueError("evoked_rank requires num_patches (pass it through "
                              "build_finetune_from_config -- see model/factory.py)")
        self.evoked_rank = int(evoked_rank)
        self.backbone, self.input, self.fs = backbone, input, float(sample_freq)
        for p in backbone.parameters():
            p.requires_grad_(False)

        C = num_channels
        if input in ('stamp_bandpow', 'stamp_induced'):
            st = backbone.stamps
            alive = (st.fire_ema >= st.dead_threshold).nonzero().flatten()
            keep = torch.cat([alive, torch.arange(st.n_routed, st.n_stamps, device=alive.device)])
            self.register_buffer('keep', keep)
        K = C if pool_channel == 'concat' else int(pool_channel.split(':')[1])
        self.time_rank = None
        if pool_time == 'trial':
            self.window = None
        elif pool_time.startswith('learned:'):
            if input != 'stamp_induced':
                raise NotImplementedError(
                    f"pool_time='learned:R' is only implemented for input='stamp_induced' "
                    f"(ADR 0014 experiment C1), got input={input!r}")
            if num_patches is None:
                raise ValueError("pool_time='learned:R' requires num_patches (pass it through "
                                  "build_finetune_from_config -- see model/factory.py)")
            self.window = None
            self.time_rank = int(pool_time.split(':')[1])
        else:
            self.window = tuple(float(v) for v in pool_time.split(':')[1].split('-'))
        if evoked_rank and self.window is not None:
            raise NotImplementedError(
                "evoked_rank needs the full patch axis (N' == num_patches); it cannot be "
                "combined with a window: pool_time, which slices patches")
        head = {}
        if pool_channel != 'concat':
            head['spatial'] = nn.Linear(C, K, bias=False)
        if self.time_rank is not None:
            R = self.time_rank
            S = len(keep)
            # Small random init keeps pre-softmax logits near zero, so the learned weighting
            # starts equivalent to C0's flat mean (uniform softmax) and only diverges from it
            # as training proceeds -- makes "does learned beat flat" a clean ablation instead
            # of a different starting point (ADR 0014 build-order step 8).
            # nn.ModuleDict only accepts nn.Module values (not raw nn.Parameter), so the pair
            # is registered as a nested nn.ParameterDict (itself an nn.Module) under one key --
            # still lands under self.head, so model.head.parameters() still sees them.
            head['time'] = nn.ParameterDict({
                'p': nn.Parameter(torch.randn(R, S) * 0.02),
                'q': nn.Parameter(torch.randn(R, num_patches) * 0.02),
            })
        if self.evoked_rank:
            # Signed low-rank time filter for the evoked branch (2b), T[s,n] = 1/N' +
            # sum_r p_r[s] q_r[n]. Small random p,q => starts as the plain trial-mean of (a,b)
            # (the time-locked average) and learns a deviation. nn.ModuleDict rejects raw
            # nn.Parameter values (same constraint as head['time']), hence the ParameterDict.
            head['evoked'] = nn.ParameterDict({
                'p': nn.Parameter(torch.randn(self.evoked_rank, len(keep)) * 0.02),
                'q': nn.Parameter(torch.randn(self.evoked_rank, num_patches) * 0.02),
            })
        if input == 'z_chan':
            head['z_proj'] = nn.Linear(backbone.head_dim, z_proj)
            n_feat = K * z_proj
        elif input == 'stamp_induced':
            n_feat = K * len(keep) * (1 + 2 * bool(include_advance) + 2 * bool(self.evoked_rank))
        else:
            n_feat = K * len(self.BANDS)
        # BatchNorm stands in for the probe's StandardScaler: log-powers are far from unit scale.
        head['cls'] = nn.Sequential(nn.BatchNorm1d(n_feat), nn.Dropout(dropout), nn.Linear(n_feat, num_classes))
        self.head = nn.ModuleDict(head)

        if input == 'stamp_bandpow':
            with torch.no_grad():
                D_tab, H_tab = (t[keep].float() for t in st._template_tables())
                fr = torch.fft.rfftfreq(D_tab.shape[-1], 1.0 / self.fs)
                sel = [((fr >= lo) & (fr < hi)).to(D_tab.device) for lo, hi in self.BANDS]
                spec = lambda T: torch.stack([torch.fft.rfft(T, dim=-1).abs().pow(2)[:, m].sum(-1) for m in sel], -1)
                self.register_buffer('E_D', spec(D_tab))                   # [S, bands]
                self.register_buffer('E_H', spec(H_tab))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()   # frozen: no dropout noise, stable top-k
        return self

    def _band_logpow(self, s):
        """s: [B, K, T] -> [B, K, bands], same statistic as probe_v10.py."""
        sp = torch.fft.rfft(s.float(), dim=-1).abs().pow(2)
        fr = torch.fft.rfftfreq(s.shape[-1], 1.0 / self.fs).to(s.device)
        return torch.stack([torch.log(sp[..., (fr >= lo) & (fr < hi)].sum(-1) + 1e-12)
                            for lo, hi in self.BANDS], -1)

    def _patch_keep(self, N, device):
        if self.window is None:
            return None
        bb = self.backbone
        starts = torch.arange(N, device=device) * bb.patch_stride
        lo, hi = (w * self.fs for w in self.window)
        keep = (starts >= lo) & (starts + bb.patch_len <= hi)
        assert keep.any(), f"window {self.window} s keeps no patch"
        return keep

    def _mix(self, t, dim):
        """Signed spatial filter over channel axis `dim`, or identity for concat."""
        if 'spatial' not in self.head:
            return t
        return torch.movedim(self.head['spatial'](torch.movedim(t, dim, -1).float()), -1, dim)

    def forward(self, x, coords, time_idx=None, valid_channels=None, pad_mask=None):
        B, C, N, L = x.shape
        bb = self.backbone
        vmask = (valid_channels if valid_channels is not None
                 else x.new_ones(B, C, dtype=torch.bool)).float()
        with torch.no_grad():
            if self.input in ('raw', 'recon'):
                src = x if self.input == 'raw' else \
                    bb(x, coords, time_idx=time_idx, bool_masked_pos=None, valid_channels=valid_channels).recon
                sig = overlap_add_patches(src.float(), bb.patch_stride) * vmask[..., None]   # [B, C, T]
                if self.window is not None:
                    lo, hi = (int(w * self.fs) for w in self.window)
                    sig = sig[..., lo:hi]
            else:
                z, _ = bb.stage_features(x, coords, time_idx=time_idx)                # [B, C, N, D]
                pk = self._patch_keep(N, x.device)
                if self.input in ('stamp_bandpow', 'stamp_induced'):
                    zg = z.permute(0, 2, 1, 3).reshape(B * N, C, -1)
                    xg = x.permute(0, 2, 1, 3).reshape(B * N, C, L)
                    amp = bb.stamps.dense_amp(zg, rms=xg.float().pow(2).mean(-1, keepdim=True).sqrt())
                    amp = amp[:, :, self.keep].reshape(B, N, C, -1, 2).float() * vmask[:, None, :, None, None]
                    if pk is not None:
                        amp = amp[:, pk]
                else:
                    z = z.float() * vmask[..., None, None]
                    if pk is not None:
                        z = z[:, :, pk]

        # Feature math in fp32: under the training loop's autocast, einsum/log run in fp16,
        # where squared amplitudes overflow and the 1e-12 epsilon rounds to 0 (NaN loss).
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.input in ('raw', 'recon'):
                feat = self._band_logpow(self._mix(sig, 1))                                  # [B, K, bands]
            elif self.input == 'stamp_bandpow':
                a, b = self._mix(amp[..., 0], 2), self._mix(amp[..., 1], 2)                  # [B, N', K, S]
                pw = torch.einsum('bnks,sq->bkq', a.pow(2), self.E_D) \
                    + torch.einsum('bnks,sq->bkq', b.pow(2), self.E_H)
                feat = torch.log(pw / a.shape[1] + 1e-12)                                    # [B, K, bands]
            elif self.input == 'stamp_induced':
                # C0 (ADR 0014 experiment C): spatial filter (step 1) + induced branch with
                # flat time weights (step 2a, w[s,n] = 1/N) -- log mean power per (filter,
                # stamp), no band collapse. The pre-log per-stamp powers span the same
                # information stamp_bandpow's band-summed powers do (via E_D/E_H), but this
                # readout is on log-power, which is NOT a strict superset of stamp_bandpow's
                # readout -- log doesn't distribute over the band sum, so this is not
                # guaranteed to reproduce stamp_bandpow spatial:8's 0.522 exactly (ADR 0014).
                a, b = self._mix(amp[..., 0], 2), self._mix(amp[..., 1], 2)                  # [B, N', K, S]
                power = a.pow(2) + b.pow(2)                                                   # [B, N', K, S]
                if self.time_rank is not None:
                    # C1: learned low-rank time weights (step 2a, ADR 0014 build-order 8),
                    # softmax-normalized over the patch axis per stamp -- w[s,n] sums to 1
                    # over n for each stamp, so this is a weighted mean, directly comparable
                    # to C0's uniform mean (w[s,n] = 1/N) rather than an unbounded rescaling.
                    logits_time = torch.einsum('rs,rn->sn', self.head['time']['p'], self.head['time']['q'])  # [S, N']
                    w = torch.softmax(logits_time, dim=-1)                                    # [S, N']
                    pooled = torch.einsum('sn,bnks->bks', w, power)                            # [B, K, S]
                else:
                    pooled = power.mean(1)                                                    # [B, K, S]
                feat = torch.log(pooled + 1e-12)                                              # [B, K, S]
                if self.include_advance:
                    # C3 (ADR 0014 build-order step 10): phase-advance branch (2c),
                    # z[k,s] = sum_n u[n+1,k,s] * conj(u[n,k,s]), u = a + i*b -- the same
                    # a, b this branch already computed, no new backbone call. Real/imag
                    # expansion (no complex dtype): z_re captures rhythm steadiness
                    # (magnitude-like), z_im captures sub-bin frequency (phase-like);
                    # feeding both raw (no log -- they can be negative) lets the linear
                    # classifier learn any function of magnitude+angle without an explicit
                    # atan2. Not log-power-scaled like `feat`, but the shared BatchNorm1d
                    # ahead of the classifier normalizes per-feature scale regardless.
                    a_next, a_prev = a[:, 1:], a[:, :-1]                                       # [B, N'-1, K, S]
                    b_next, b_prev = b[:, 1:], b[:, :-1]
                    z_re = (a_next * a_prev + b_next * b_prev).sum(1)                          # [B, K, S]
                    z_im = (b_next * a_prev - a_next * b_prev).sum(1)                          # [B, K, S]
                    feat = torch.cat([feat, z_re, z_im], dim=-1)                                # [B, K, 3*S]
                if self.evoked_rank:
                    # C4 (ADR 0014 build-order step 12): evoked branch (2b),
                    # sum_n T[s,n] * u[n,k,s], u = a + i*b. T is signed and applied to a and b
                    # LINEARLY (not to power) -- a linear functional preserves phase-locked
                    # content, power destroys it. Same a, b as above, no new backbone call.
                    # Fed raw (can be negative); the shared BatchNorm1d normalizes scale.
                    T = 1.0 / a.shape[1] + torch.einsum(
                        'rs,rn->sn', self.head['evoked']['p'], self.head['evoked']['q'])   # [S, N']
                    ev_re = torch.einsum('sn,bnks->bks', T, a)                             # [B, K, S]
                    ev_im = torch.einsum('sn,bnks->bks', T, b)                             # [B, K, S]
                    feat = torch.cat([feat, ev_re, ev_im], dim=-1)
            else:
                zp = self.head['z_proj'](z).mean(2)                                          # [B, C, P]
                feat = self._mix(zp, 1)                                                      # [B, K, P]
            if 'spatial' not in self.head:
                # concat keeps padded channels as columns: zero them after the log, as the probe
                # does. Left at log(eps) = -27.6 they swamp BatchNorm until its running stats adapt.
                feat = feat * vmask[:, :, None]
        return self.head['cls'](feat.flatten(1)), None, None


def build_finetune(backbone, num_channels, num_classes, input='head_z', **kw):
    """finetune_cls entry: input='head_z' is the original MeSAEFinetune (the bundled ADR 0014
    reference); every other input is an experiment-B MeSAEFeatureHead."""
    if input == 'head_z':
        allowed = ('hidden', 'freeze_backbone', 'dropout', 'use_topo_feature')
        return MeSAEFinetune(backbone, num_channels, num_classes, **{k: v for k, v in kw.items() if k in allowed})
    allowed = ('task', 'pool_channel', 'pool_time', 'z_proj', 'dropout', 'sample_freq',
               'freeze_backbone', 'num_patches', 'include_advance', 'evoked_rank')
    return MeSAEFeatureHead(backbone, num_channels, num_classes, input=input,
                            **{k: v for k, v in kw.items() if k in allowed})
