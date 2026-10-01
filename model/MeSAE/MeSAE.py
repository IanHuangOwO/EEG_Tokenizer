import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MeSAE.MeSAE_modules import (SpatialTemporalEmbeddings, TSAEncoder, StampBank, RelativeSpatialBias,
                                         fold_sum,
                                         spatial_mix, FlatTimePool, LearnedTimePool,
                                         EvokedBranch, phase_advance, StampExtractor, FeatureHead,
                                         resolve_head_config, make_head_checkpoint,
                                         needs_stamp, needs_raw, needs_latent, feature_names)


def _ema_update(buf, val, decay=0.99):
    """In-place EMA update, skipped if val is NaN/Inf. A plain `.mul_(decay).add_(val,
    alpha=1-decay)` permanently poisons buf the moment val is ever NaN even once (a single
    bad batch, e.g. a transient fp16-autocast overflow early in training): NaN propagates
    through every future update (decay*NaN + (1-decay)*anything = NaN), so a diagnostic
    stuck NaN for an entire run can trace back to one early outlier batch long since
    recovered from. Skipping non-finite updates lets the EMA keep tracking real values."""
    if torch.isfinite(val):
        buf.mul_(decay).add_(val, alpha=1 - decay)


_ROUTED_STATE = ('stamps.W_down_routed', 'stamps.b_down_routed', 'stamps.w_amp_routed', 'stamps.b_amp_routed',
                 'stamps.D_routed', 'stamps.fire_ema', 'ema_stamp_router_entropy', 'ema_stamp_router_load_std',
                 'ema_stamp_gate_entropy')
_RENAMED_STATE = ('W_down', 'b_down', 'w_amp', 'b_amp', 'D')   # were stamps.<name>_shared


def _legacy_state(state_dict, prefix, *args):
    """load_state_dict pre-hook for checkpoints trained before routed stamps were removed: drop the empty routed
    tensors and routing EMAs of a static checkpoint, and rename stamps.<name>_shared to stamps.<name>.
    A checkpoint with routed stamps cannot be rebuilt here: it needs the `routed-stamps` branch."""
    for k in _ROUTED_STATE:
        v = state_dict.pop(prefix + k, None)
        if v is not None and k.startswith('stamps.') and v.numel() > 0:
            raise ValueError("checkpoint has routed stamps, which were removed: "
                             "load it from the `routed-stamps` branch")
    for k in _RENAMED_STATE:
        if prefix + f'stamps.{k}_shared' in state_dict:
            state_dict[prefix + f'stamps.{k}'] = state_dict.pop(prefix + f'stamps.{k}_shared')


def _restore_phase(module, incompatible_keys):
    """load_state_dict post-hook: re-apply the checkpoint's phase flags (plain
    attributes, not state). Does not freeze anything."""
    if bool(module.masked_phase):
        module.enter_masked_phase(freeze_stamps=False)
    else:
        module.enter_tokenizer_phase()


class MeSAEPretrain(nn.Module):
    """
    Spatiotemporal stamp-dictionary EEG tokenizer. Goal is explainable, per-patch embeddings (not a discrete vocabulary):
    channel-count invariant (cross-dataset unification still matters) but NOT
    length-invariant (each patch keeps its own embedding, for temporal localization of
    events within a trial).

    Pipeline: encoder -> StampBank (dictionary of fixed per-atom waveform templates, each
    presented at a per-channel, per-atom amplitude/phase read off that atom's own
    bottleneck) -> reconstruction, summed directly in patch space (no separate decoder
    stage); the dictionary is static (every stamp active everywhere).

    Trains in two phases of one run (train_pretrain.py, CONTEXT.md:
    Tokenizer stage / Masked stage):
    - enter_tokenizer_phase(): every block runs with temporal mixing only (spatial
      attention and the coordinate embedding off), no masking (bool_masked_pos=None) —
      encoder + StampBank train jointly on single-channel features, so the stamp dictionary
      isn't built from cross-channel-mixed input.
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
        blocks_per_stage=2,
        num_channels=1,
        spatial_embedding=True,
        n_stamps=16,
        stamp_hidden_width=16,
        stamp_spatial_rank=0,
        n_routed_ffn_experts=4,
        n_shared_ffn_experts=1,
        ffn_top_k=2,
        patch_stride=None,
        skip_mode='gated',
        decoder_blocks=0,
        skip_drop=0.0,
        temporal_bias=False,
    ):
        super().__init__()
        self.patch_len = patch_len
        # patch_stride: which samples a masked run truly hides (_position_weights, 50% overlap).
        # Falls back to patch_len (non-overlapping) matching PretrainDataset.
        self.patch_stride = patch_stride or patch_len
        self.head_dim = embed_dim
        self.num_channels = num_channels

        self.embed   = SpatialTemporalEmbeddings(patch_len, embed_dim, spatial=spatial_embedding)
        self.encoder = TSAEncoder(embed_dim, depth=enc_depth, num_heads=spatial_heads, mlp_ratio=mlp_ratio,
                                   dropout=dropout, blocks_per_stage=blocks_per_stage,
                                   n_routed_ffn_experts=n_routed_ffn_experts, n_shared_ffn_experts=n_shared_ffn_experts,
                                   ffn_top_k=ffn_top_k, skip_mode=skip_mode, decoder_blocks=decoder_blocks,
                                   skip_drop=skip_drop, temporal_bias=temporal_bias)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        # Masked tokens swap their CONTENT for mask_token before the time/coord embeddings
        # are added (MAE convention), so a masked token still knows where and when it is.
        # Zero-padded (missing) channels are left out of spatial attention as keys (TSABlock).
        # spatial_embedding: Fourier coordinate embedding + a directional relative-position bias in
        # every block's spatial attention (RelativeSpatialBias), on or off together (the ablation).
        self.spatial_bias = RelativeSpatialBias(enc_depth, spatial_heads) if spatial_embedding else None

        self.stamps = StampBank(embed_dim, patch_len, n_stamps=n_stamps, hidden_width=stamp_hidden_width,
                                spatial_rank=stamp_spatial_rank)
        # convenience alias — viz/checker code reads it off the model directly
        self.n_stamps = self.stamps.n_stamps
        self.stamps_frozen = False
        self.register_buffer('masked_phase', torch.tensor(False))
        self._register_load_state_dict_pre_hook(_legacy_state)
        self.register_load_state_dict_post_hook(_restore_phase)

        # EMA health of the FFN MoE routers (MoEFFN/FFNRouter, one per TSABlock, averaged
        # across blocks by TSAEncoder.forward), see update_ffn_router_metrics below.
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
        self.enable_temporal()

    def enter_masked_phase(self, freeze_stamps=True):
        """Freeze before enable_spatial: a frozen dictionary never sees mixed z. With
        freeze_stamps=False it trains on mixed z, so per-stamp amp is no longer a
        source topomap (mp_loss stays on — get_loss gates it on stamps_frozen)."""
        self.masked_phase.fill_(True)
        if freeze_stamps:
            self.freeze_stamps()
        self.enable_temporal()
        self.enable_spatial()

    def freeze_stamps(self):
        """
        End of Tokenizer stage: lock StampBank so the Masked stage's frozen reconstruction
        target stops moving (mp_loss is dropped from the Masked-stage loss once this is
        called, see get_loss): two-stage/sequential rather than joint-warmup-then-freeze.
        """
        for p in self.stamps.parameters():
            p.requires_grad_(False)
        self.stamps_frozen = True

    @torch.no_grad()
    def update_ffn_router_metrics(self, ffn_router_entropy, ffn_router_load_std, ffn_gate_entropy):
        """EMA smoothing of the FFN MoE router health — see TSAEncoder.forward (MeSAE_modules.py) for where these three
        already-averaged-across-blocks values come from. Called from
        MeSAETrainer.update_diagnostics with out.ffn_router_entropy/out.ffn_router_load_std/
        out.ffn_gate_entropy, same call site as update_head_metrics."""
        _ema_update(self.ema_ffn_router_load_std, ffn_router_load_std)
        _ema_update(self.ema_ffn_router_entropy, ffn_router_entropy)
        _ema_update(self.ema_ffn_gate_entropy, ffn_gate_entropy)

    def stage_features(self, x, coords, time_idx=None, bool_masked_pos=None, valid_channels=None):
        """Returns (z [B, C, N, D], ffn_lb_loss scalar) — ffn_lb_loss is the summed
        load-balance loss of every TSABlock's MoEFFN (see MeSAE_modules.TSAEncoder)."""
        z = self.embed(x, coords=coords, time_idx=time_idx, bool_masked_pos=bool_masked_pos,
                       mask_token=self.mask_token)  # [B, C, N, D]
        bias = self.spatial_bias(coords) if self.spatial_bias is not None and coords is not None else None
        # a window's zero-padded tail patches (zero on every channel) are left out of temporal attention
        valid_patches = x.abs().amax(dim=(1, 3)) > 0                                    # [B, N]
        return self.encoder(z, valid_channels, bias, valid_patches)  # [B, C, N, D], ffn_lb_loss

    @torch.no_grad()
    def encode_stamps(self, x, coords, time_idx=None, valid_channels=None):
        """Unmasked stamp code for analysis/viz: StampBank output (recon [G, C, L], amp
        [G, C, n_stamps, 2] with rms, h [G, n_stamps]; G = B*N positions, b*N + n), no mp_loss.
        Same per-channel rms as forward() -- without it amp lacks its raw-amplitude factor."""
        B, C, N, L = x.shape
        z, _ = self.stage_features(x, coords, time_idx=time_idx, valid_channels=valid_channels)
        z_g = z.permute(0, 2, 1, 3).reshape(B * N, C, -1)
        rms = x.permute(0, 2, 1, 3).reshape(B * N, C, L).pow(2).mean(dim=-1, keepdim=True).sqrt()
        vc_g = None if valid_channels is None else valid_channels.unsqueeze(1).expand(B, N, C).reshape(B * N, C)
        c_g = None if coords is None else coords.unsqueeze(1).expand(B, N, C, 3).reshape(B * N, C, 3)
        return self.stamps(z_g, rms=rms, valid_channels=vc_g, coords=c_g)

    def forward(self, x, coords, time_idx=None, bool_masked_pos=None, valid_channels=None):
        """
        x: [B, C, N, L], coords: [B, C, 3]
        bool_masked_pos: [B, C, N] bool — None during the Tokenizer stage (no masking);
        pass real masks only in the Masked stage, once temporal/spatial mixing are enabled
        and the stamps are frozen (see enable_temporal/enable_spatial/freeze_stamps).
        valid_channels: [B, C] bool, True=real (not zero-padded) channel, or None. Passed into
        StampBank (padded channels stay out of h and mp_loss) and carried through on the returned
        SimpleNamespace so get_loss/_recon_loss can exclude padded channels from the loss — a
        zero-padded channel's "reconstruction" is meaningless signal, not a real target. Padded
        channels still decode/reconstruct like any other.
        returns SimpleNamespace(recon [B,C,N,L], h [G, n_stamps] stamp strengths and amp
        [G, C, n_stamps, 2] (G = B*N patch positions, b*N + n; StampBank.decode re-expands amp),
        mp_loss/mp_map, ffn_lb_loss scalar (TSABlock MoEFFN routers, summed across blocks), FFN
        router health.
        """
        B, C, N, L = x.shape

        z, ffn_lb_loss = self.stage_features(x, coords, time_idx=time_idx, bool_masked_pos=bool_masked_pos,
                                              valid_channels=valid_channels)  # [B, C, N, D]
        # Channel-grouped layout for StampBank: [B, C, N, *] -> permute to [B, N, C, *]
        # then merge (B, N) — adjacent after the permute, so the merge is a safe reshape
        # (docs/agents/reshape-pitfalls.md; permute forces a copy, contiguity handled by
        # reshape itself). One group = one patch position with all its channels.
        G = B * N
        z_g = z.permute(0, 2, 1, 3).reshape(G, C, -1)          # [G, C, D]
        x_g = x.permute(0, 2, 1, 3).reshape(G, C, L)           # [G, C, L] — mp_loss target

        # Per-channel raw-input RMS — the amplitude signal the LayerNorm stack erased
        # from z (embed.norm -> per-block norm_out -> stamps.input_norm), multiplied back
        # into every amp inside StampBank. Masked positions get 1.0: their true patch is
        # hidden from the encoder, so feeding its RMS would leak the target's amplitude
        # into masked reconstruction — the encoder must predict a masked patch's loudness
        # through z, same as it always did.
        rms = x_g.pow(2).mean(dim=-1, keepdim=True).sqrt()     # [G, C, 1]

        vc_g = None
        if valid_channels is not None:
            # [B, C] -> broadcast over N -> [G, C]: real channels only in h / mp_loss.
            vc_g = valid_channels.unsqueeze(1).expand(B, N, C).reshape(G, C)

        if bool_masked_pos is not None:
            mask_g = bool_masked_pos.permute(0, 2, 1).reshape(G, C, 1)
            # Masking is generated channel-agnostic (IO/masking.py never sees
            # valid_channels), so a padded channel's patch can land inside the mask —
            # gate the override on valid-AND-masked, not masked alone, or a padded
            # channel's rms gets force-set to 1.0 here, fabricating a nonzero amp/recon
            # for a channel that's always exactly 0.
            if vc_g is not None:
                mask_g = mask_g & vc_g.unsqueeze(-1)
            rms = torch.where(mask_g, torch.ones_like(rms), rms)

        c_g = None if coords is None else coords.unsqueeze(1).expand(B, N, C, 3).reshape(G, C, 3)   # [G, C, 3]
        out = self.stamps(z_g, x_target=x_g, rms=rms, valid_channels=vc_g, coords=c_g)

        recon = out.recon.reshape(B, N, C, L).permute(0, 2, 1, 3)  # back to [B, C, N, L]

        return SimpleNamespace(
            recon=recon,
            h=out.h,
            amp=out.amp,   # [G, C, n_stamps, 2] (G = B*N), for StampBank.decode
            src=out.src,   # [G, n_stamps, K, 2] source activations, or None (spatial_rank 0)
            mp_loss=out.mp_loss,
            # [B, C, N], same layout as bool_masked_pos (G = B*N rows were b*N + n)
            mp_map=None if out.mp_map is None else out.mp_map.reshape(B, N, C).permute(0, 2, 1),
            ffn_lb_loss=ffn_lb_loss,
            ffn_router_entropy=self.encoder.last_ffn_router_entropy,
            ffn_router_load_std=self.encoder.last_ffn_router_load_std,
            ffn_gate_entropy=self.encoder.last_ffn_gate_entropy,
            valid_channels=valid_channels,
        )

    def _position_weights(self, x, bool_masked_pos, valid_channels, unmasked_weight):
        """-> (valid [B, C, N, 1], w [B, C, N, L], hidden [B, C, N, L] or None). valid: 1 on real
        channels AND real patches. A patch that is exactly zero on every channel is time padding
        (a window's zero tail, see IO/preprocessing.py's window_continuous_signal) -- z-scored
        real EEG is never all-zero across a whole patch. hidden: a SAMPLE counts as masked only
        if every patch covering it is masked -- with 50% patch overlap, the outer half of a run's
        first/last masked patch is also inside a visible neighbour, so it is visible content,
        not a reconstruction target. w: valid, times (1 on hidden samples, unmasked_weight on
        visible ones) in the masked phase. Shared by the recon MSE and mp_loss."""
        B, C, N, L = x.shape
        valid = x.new_ones(B, C, 1, 1) if valid_channels is None \
            else valid_channels.view(B, C, 1, 1).to(x.dtype)
        real_patch = (x.abs().amax(dim=(1, 3)) > 0).to(x.dtype).view(B, 1, N, 1)
        valid = valid * real_patch
        if bool_masked_pos is None:
            return valid, valid.expand(B, C, N, L), None
        stride = self.patch_stride
        m = bool_masked_pos.to(x.dtype).unsqueeze(-1).expand(B, C, N, L)
        n_cover = fold_sum(torch.ones(N, L, device=x.device, dtype=x.dtype), stride)       # [T]
        hidden = (fold_sum(m, stride) > n_cover - 0.5).to(x.dtype).unfold(-1, L, stride)  # [B, C, N, L]
        return valid, valid * (hidden + unmasked_weight * (1.0 - hidden)), hidden

    def _recon_loss(self, recon, x, bool_masked_pos, valid_channels=None,
                    mse_patch_weight=1.0, unmasked_weight=1.0):
        """Recon loss: plain time-domain MSE of every raw patch against its own reconstruction.

        Position weights: padded channels 0; in the masked phase masked positions 1 and visible
        positions `unmasked_weight` (0 = standard MAE, loss on masked patches only; 1 = every
        position counts equally). With bool_masked_pos=None (tokenizer phase) every valid position
        is a target and unmasked_weight is unused.

        Removed terms, with the evidence: the overlap-added trial MSE (it let
        neighbouring patches disagree on their shared samples as long as the crossfade averaged
        out; without it seam disagreement fell 40%), a masked STFT loss (0019), a nested
        reconstruction loss (0018); earlier spectral whitening and a window level (0011).

        Logged mse_patch stays the plain all-valid-position MSE, comparable across phases
        whatever the weights. masked/unmasked are a diagnostic split.
        Returns (weighted total, l_masked, l_unmasked).
        """
        recon, x = recon.float(), x.float()
        valid, w, hidden = self._position_weights(x, bool_masked_pos, valid_channels, unmasked_weight)

        def wmean(err, wt):
            wt = wt.expand_as(err)
            s = wt.sum()
            return (wt * err).sum() / s if s > 0 else err.new_zeros(())

        patch_err = (recon - x).pow(2)
        total = mse_patch_weight * wmean(patch_err, w)

        with torch.no_grad():
            plain_patch = wmean(patch_err, valid)
            l_masked, l_unmasked = 1.0, plain_patch
            if bool_masked_pos is not None:
                l_masked = wmean(patch_err, valid * hidden)
                l_unmasked = wmean(patch_err, valid * (1.0 - hidden))
        # Logged as mse_<name> (train_pretrain.py reads _last_loss_terms; plugin.py matches 'mse_').
        self._last_loss_terms = {'patch': plain_patch.item()}
        return total, l_masked, l_unmasked

    def get_loss(self, x, recon, bool_masked_pos=None,
                 mse_patch_weight=1.0, unmasked_weight=1.0,
                 ffn_lb_loss=None, ffn_lb_weight=0.01, valid_channels=None,
                 mp_loss=None, mp_weight=0.0, mp_map=None):
        """
        Returns (total, l_masked, l_unmasked).

        mp_map/mp_weight: the Matching-Pursuit-style residual loss per position (StampBank.forward),
        trained under the same position weights as the recon MSE; mp_loss (its plain valid-channel
        mean) is only logged. Added only when mp_weight != 0.

        Reconstruction term is _recon_loss's weighted patch MSE (see its docstring for
        mse_patch_weight / unmasked_weight).

        Tokenizer stage (bool_masked_pos=None): plain full reconstruction, l_masked=1.0
        placeholder (nothing masked yet).

        Dictionary shaping is mp_loss's job (see StampBank.forward): the residual-ordered term
        that stops stamps being rewarded for re-explaining what a higher-ranked stamp already
        covered. Earlier attempts at this — spectral whitening, then activation
        decorrelation/negentropy — were measured and dropped.

        ffn_lb_loss (MoEFFN routers' load-balance loss, summed across TSABlocks) is added unconditionally, both stages: it comes
        from the encoder, which keeps training through the Masked stage (freeze_stamps()
        never locks the encoder).
        """
        total, l_masked, l_unmasked = self._recon_loss(
            recon, x, bool_masked_pos, valid_channels=valid_channels,
            mse_patch_weight=mse_patch_weight, unmasked_weight=unmasked_weight)

        if ffn_lb_loss is not None:
            total = total + ffn_lb_weight * ffn_lb_loss

        # Gated on stamps_frozen: mp_loss exists to shape WHICH atom owns which content (see StampBank.forward's
        # mp_loss section), and a frozen dictionary's atoms can't be reshaped. Gradient
        # would still reach the (never-frozen) encoder through amp=f(z), but pushing the
        # encoder to make greedy residual decomposition easier is not the Masked stage's
        # job — that stage optimizes masked reconstruction, and leaving this on would
        # quietly add a second, unrelated objective to it.
        if mp_map is not None and mp_weight and (bool_masked_pos is None or not self.stamps_frozen):
            # Same position weights as the recon MSE: without them mp_loss counts visible patches at
            # full weight and undoes unmasked_weight.
            _, w, _ = self._position_weights(x, bool_masked_pos, valid_channels, unmasked_weight)
            w = w.mean(-1).to(mp_map.dtype)                                          # per patch
            s = w.sum()
            total = total + mp_weight * ((w * mp_map).sum() / s if s > 0 else mp_map.new_zeros(()))
            # logged value stays the plain valid-channel mean, comparable across phases
            self._last_loss_terms['mp'] = mp_loss.detach().item()
        return total, l_masked, l_unmasked

    def get_metrics(self):
        metrics = {}

        # U-Net skip gate(s) on the encoder's residual-add path: sigmoid(g) in [0,1],
        # 0 = drop skip, 1 = plain add.
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

        # FFN router health — see update_ffn_router_metrics above (per-TSABlock MoEFFN routers,
        # averaged across blocks).
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
        self.head_cfg = head_cfg
        if needs_latent(head_cfg):
            raise NotImplementedError("latent_* head entries run from the feature cache (train_finetune.py) only")
        stamp = needs_stamp(head_cfg)
        self.extractor = StampExtractor(backbone, channel_idx) if stamp else None
        if stamp:
            assert head_cfg['num_stamps'] == backbone.n_stamps, "num_stamps must equal the stamp count"
        self.head = FeatureHead(head_cfg)
        if 'stamp_band' in feature_names(head_cfg):
            E_D, E_H = self.extractor.band_tables(head_cfg['sample_freq'])
            self.head.entries['stamp_band'].E_D.copy_(E_D)
            self.head.entries['stamp_band'].E_H.copy_(E_H)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()   # frozen: no dropout noise
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
        return make_head_checkpoint(self.head, self.head_cfg, self.channel_idx.tolist(), backbone_checkpoint)

    @classmethod
    def from_checkpoint(cls, backbone, ckpt):
        cfg = dict(ckpt['head_config'])
        channel_idx = cfg.pop('channel_idx')
        cfg.pop('keep', None)   # heads saved before routed stamps were removed list the alive stamps: now all of them
        model = cls(backbone, cfg, channel_idx)
        model.head.load_state_dict(ckpt['model_state_dict'])
        return model


def build_finetune(backbone, num_channels, num_classes, channel_idx=None, num_patches=None,
                   sample_freq=200, **ft_params):
    """finetune_cls entry: builds the frozen backbone + FeatureHead from the numeric head config."""
    channel_idx = list(range(num_channels)) if channel_idx is None else list(channel_idx)
    num_stamps = 0
    if needs_stamp(ft_params):
        num_stamps = backbone.stamps.n_stamps
    cfg = resolve_head_config(ft_params, num_classes=num_classes, num_patches=num_patches,
                              num_channels=len(channel_idx), num_stamps=num_stamps,
                              patch_len=backbone.patch_len, patch_stride=backbone.patch_stride,
                              sample_freq=float(sample_freq))
    return FinetuneModel(backbone, cfg, channel_idx)
