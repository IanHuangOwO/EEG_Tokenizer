import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MeSAE.MeSAE_modules import SpatialTemporalEmbeddings, TSAEncoder, ExpertChannelPool, MultiHeadDecoder, PerChannelHeadAttn


class TopKSAE(nn.Module):
    """
    Sparse Autoencoder with hard top-K sparsity (OpenAI-style: keep the top-K encoder
    pre-activations per example, zero the rest, no L1 penalty needed since sparsity is
    structural). Weights are shared across whatever axis calls this (see
    MeSAEPretrain, which applies one TopKSAE per spatial filter) — that's what gives
    the sparse feature dictionary a single reusable vocabulary across filters instead of
    a per-filter-unique one.

    Dead-feature handling mirrors the codebook-collapse problem MeFSQ already fights,
    different mechanism: an EMA firing-frequency buffer flags features that haven't
    fired recently, and an aux-k loss lets those dead features have a second shot at
    explaining the residual reconstruction error, so they get gradient instead of
    staying permanently dead.
    """
    def __init__(self, dim, n_features, k, aux_k=None, dead_threshold=1e-3, ema_decay=0.99):
        super().__init__()
        self.dim = dim
        self.n_features = n_features
        self.k = k
        self.aux_k = aux_k or min(n_features, k * 4)
        self.dead_threshold = dead_threshold
        self.ema_decay = ema_decay

        self.b_dec = nn.Parameter(torch.zeros(dim))
        self.enc = nn.Linear(dim, n_features)
        self.dec = nn.Parameter(torch.empty(n_features, dim))
        nn.init.kaiming_uniform_(self.dec, a=math.sqrt(5))

        self.register_buffer('fire_ema', torch.zeros(n_features))

    def _decode(self, h):
        dec_n = F.normalize(self.dec, dim=-1)
        return h @ dec_n + self.b_dec

    def forward(self, x):
        """x: [..., dim] -> x_hat [..., dim], h [..., n_features], aux_loss scalar"""
        shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.dim)
        pre = self.enc(x_flat - self.b_dec)  # [B, F]

        topk_val, topk_idx = pre.topk(self.k, dim=-1)
        h = torch.zeros_like(pre).scatter_(-1, topk_idx, F.relu(topk_val))
        x_hat = self._decode(h)

        aux_loss = x_hat.new_zeros(())
        if self.training:
            with torch.no_grad():
                fired = torch.zeros_like(pre).scatter_(-1, topk_idx, 1.0).mean(dim=0)
                self.fire_ema.mul_(self.ema_decay).add_(fired, alpha=1 - self.ema_decay)
                dead_mask = self.fire_ema < self.dead_threshold  # [F]

            if dead_mask.any():
                residual = (x_flat - x_hat).detach()
                dead_pre = pre.masked_fill(~dead_mask.unsqueeze(0), float('-inf'))
                aux_k = min(self.aux_k, int(dead_mask.sum().item()))
                aux_val, aux_idx = dead_pre.topk(aux_k, dim=-1)
                h_aux = torch.zeros_like(pre).scatter_(-1, aux_idx, F.relu(aux_val))
                residual_hat = self._decode(h_aux)
                aux_loss = F.mse_loss(residual_hat, residual)

        return x_hat.reshape(*shape, self.dim), h.reshape(*shape, self.n_features), aux_loss


class MeSAEPretrain(nn.Module):
    """
    Per-filter Sparse Autoencoder EEG tokenizer — parallel to MeFSQPretrain, not a
    variant of it. Goal is explainable, per-patch embeddings (not a discrete vocabulary):
    channel-count invariant (cross-dataset unification still matters) but NOT
    length-invariant (each patch keeps its own embedding, for temporal localization of
    events within a trial). No Router, no Expert pools, no discrete VQ.

    Pipeline: encoder -> ExpertChannelPool (reused as a small fixed bank of "spatial
    filters", channel-count invariant) -> per-filter TopKSAE (shared weights across
    filters) -> MultiHeadDecoder (reused, "sum after decode not before") -> reconstruction.
    Every filter is isolable end-to-end (pool -> SAE -> decode -> contribution, summed),
    matching an ICA-style linear-sum-of-independent-components model — not a discrete-token
    vocabulary (still deliberately deferred).

    Trains in two sequential stages (see docs/adr/0003-mesae-two-stage-masked-training.md
    and CONTEXT.md: Tokenizer stage / Masked stage):
    - Tokenizer stage: temporal/spatial mixing OFF (enable_temporal/enable_spatial not yet
      called), no masking (bool_masked_pos=None) — encoder + SAE train jointly on
      patch-local features only, so the SAE's dictionary can't be built from
      already-context-leaked input.
    - Masked stage: call enable_temporal(), enable_spatial(), freeze_sae() (in that order),
      then train with bool_masked_pos set — only embed/encoder/mask_token keep learning,
      predicting masked patches through the now-frozen SAE/decoder.
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
        upsample_residual_add=True,
        num_channels=1,
        n_filters=8,
        pool_hidden=32,
        pool_temperature=1.0,
        sae_expansion=8,
        sae_k=32,
        decoder_hidden=None,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.head_dim = embed_dim
        self.num_channels = num_channels
        self.n_filters = n_filters

        self.embed   = SpatialTemporalEmbeddings(patch_len, embed_dim)
        self.encoder = TSAEncoder(embed_dim, depth=enc_depth, num_heads=spatial_heads, mlp_ratio=mlp_ratio,
                                   dropout=dropout, pool_after_blocks=pool_after_blocks,
                                   upsample_residual_add=upsample_residual_add)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.filter_pool = ExpertChannelPool(embed_dim, n_filters, hidden=pool_hidden, temperature=pool_temperature)
        self.sae         = TopKSAE(embed_dim, sae_expansion * embed_dim, sae_k)
        self.decoder     = MultiHeadDecoder(n_filters, embed_dim, num_channels * patch_len, hidden=decoder_hidden)
        self.sae_frozen  = False

        # EMA buffers for the decoder-output fingerprint (data-dependent diversity of
        # each filter's actual decoded contribution) — same diagnostic value as
        # MeFSQPretrain._update_fingerprint_stats, adapted to filters instead of Experts.
        self.register_buffer('ema_fingerprint_sim',     torch.tensor(0.0))
        self.register_buffer('ema_fingerprint_sim_std', torch.tensor(0.0))

    def enable_spatial(self):
        self.embed.enable_spatial()
        self.encoder.enable_spatial()

    def enable_temporal(self):
        self.encoder.enable_temporal()

    def freeze_sae(self):
        """
        End of Tokenizer stage: lock filter_pool + SAE + decoder so the Masked stage's
        frozen reconstruction target stops moving. aux_loss (dead-feature rescue) must be
        dropped from the Masked-stage loss entirely once this is called — rescuing a frozen
        SAE's dead features is meaningless, see get_loss. Mirrors MeFSQ's
        freeze_vq_and_decoder(), but two-stage/sequential rather than joint-warmup-then-
        freeze (see docs/adr/0003-mesae-two-stage-masked-training.md).
        """
        for p in self.filter_pool.parameters():
            p.requires_grad_(False)
        for p in self.sae.parameters():
            p.requires_grad_(False)
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.sae_frozen = True

    def stage_features(self, x, coords, time_idx=None, bool_masked_pos=None):
        z = self.embed(x, coords=coords, time_idx=time_idx)  # [B, C, N, D]
        if bool_masked_pos is not None:
            mask = bool_masked_pos.unsqueeze(-1).type_as(z)  # [B, C, N, 1]
            z = z * (1.0 - mask) + self.mask_token * mask
        return self.encoder(z)  # [B, C, N, D]

    def _pool_channels(self, z, valid_channels=None):
        """z: [B, C, N, D] -> z_bnc [M, C, D] (M=B*N), valid_mask [M, C] or None."""
        B, C, N, D = z.shape
        z_bnc = z.permute(0, 2, 1, 3).reshape(B * N, C, D)
        valid_mask = None
        if valid_channels is not None:
            valid_mask = valid_channels.unsqueeze(1).expand(B, N, C).reshape(B * N, C)
        return z_bnc, valid_mask

    def encode_post_sae(self, x, coords, time_idx=None, valid_channels=None):
        """
        Real per-filter, per-channel signal AFTER the SAE's sparse-code round-trip: each
        filter's decoder output (C*patch_len, jointly reconstructing all channels from that
        filter's own sparse code) reshaped per channel, before the cross-filter sum.
        Genuinely filter-differentiated (each filter reads a different pooled View through
        the shared TopKSAE dictionary and has its own decoder weights) — mirrors
        MeFSQPretrain.encode_post_vq; this is what MeSAEFinetune reads.
        valid_channels: [B, C] bool, True = real (not zero-padded) channel (optional).
        Returns z_h [B, C, N, Q, patch_len].
        """
        B, C, N, L = x.shape
        P = self.patch_len

        z = self.stage_features(x, coords, time_idx=time_idx, bool_masked_pos=None)  # [B, C, N, D]
        z_bnc, valid_mask = self._pool_channels(z, valid_channels)  # [M, C, D]

        pooled, _ = self.filter_pool(z_bnc, valid_mask)  # [M, Q, D]
        sae_out, _, _ = self.sae(pooled)                 # [M, Q, D]

        recon_per_filter = self.decoder(sae_out)  # [M, Q, C*P]
        recon_per_filter = recon_per_filter.reshape(B, N, self.n_filters, C, P)
        return recon_per_filter.permute(0, 3, 1, 2, 4)  # [B, C, N, Q, P]

    @torch.no_grad()
    def _update_fingerprint_stats(self, recon_per_filter, decay=0.99):
        """Pairwise cosine similarity between filters' own decoded contributions — low
        mean/std means filters are decoding near-identical signals regardless of patch."""
        M, Q, K = recon_per_filter.shape
        if Q < 2:
            return
        normed = F.normalize(recon_per_filter.float(), dim=-1)
        sim = torch.einsum('mqk,mpk->mqp', normed, normed)  # [M, Q, Q]
        off_diag = ~torch.eye(Q, dtype=torch.bool, device=sim.device)
        off = sim[:, off_diag].reshape(M, Q * (Q - 1))
        self.ema_fingerprint_sim.mul_(decay).add_(off.mean(), alpha=1 - decay)
        self.ema_fingerprint_sim_std.mul_(decay).add_(off.std(), alpha=1 - decay)

    def forward(self, x, coords, time_idx=None, bool_masked_pos=None, valid_channels=None):
        """
        x: [B, C, N, L], coords: [B, C, 3]
        bool_masked_pos: [B, C, N] bool — None during the Tokenizer stage (no masking);
        pass real masks only in the Masked stage, once temporal/spatial mixing are enabled
        and the SAE is frozen (see enable_temporal/enable_spatial/freeze_sae).
        valid_channels: [B, C] bool, True = real (not zero-padded) channel (optional).
        returns SimpleNamespace(recon [B,C,N,L], attn [B,N,Q,C] (per-filter topography),
        sae_hidden [M,Q,F], aux_loss scalar).
        """
        B, C, N, L = x.shape
        D = self.head_dim

        z = self.stage_features(x, coords, time_idx=time_idx, bool_masked_pos=bool_masked_pos)  # [B, C, N, D]
        z_bnc, valid_mask = self._pool_channels(z, valid_channels)  # [M, C, D]

        pooled, attn = self.filter_pool(z_bnc, valid_mask)   # [M, Q, D], [M, Q, C]
        sae_out, sae_hidden, aux_loss = self.sae(pooled)     # [M, Q, D], [M, Q, F]

        recon_per_filter = self.decoder(sae_out)             # [M, Q, C*patch_len]
        recon = recon_per_filter.sum(dim=1).reshape(B, N, C, L).permute(0, 2, 1, 3)

        if not self.training:
            self._update_fingerprint_stats(recon_per_filter)

        return SimpleNamespace(
            recon=recon,
            attn=attn.reshape(B, N, self.n_filters, C),
            sae_hidden=sae_hidden,
            aux_loss=aux_loss,
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
        r = recon.reshape(B * C, L, N).float()
        t = x.reshape(B * C, L, N).float()
        levels = []
        win = N
        while True:
            rp = F.avg_pool1d(r, kernel_size=win, stride=win)
            tp = F.avg_pool1d(t, kernel_size=win, stride=win)
            levels.append((rp.reshape(B, C, -1, L), tp.reshape(B, C, -1, L)))
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

    def get_loss(self, x, recon, aux_loss, bool_masked_pos=None, aux_weight=0.03, hierarchical_mse_weight=1.0):
        """
        Returns (total, l_masked, l_unmasked).

        Reconstruction term is the hierarchical patch-pyramid MSE (see
        _hierarchical_recon_loss), scaled by hierarchical_mse_weight.

        Tokenizer stage (bool_masked_pos=None): plain full reconstruction, l_masked=1.0
        placeholder (nothing masked yet), aux_loss included so the SAE's dead-feature
        rescue can still train.

        Masked stage (bool_masked_pos given, self.sae_frozen True by then): aux_loss is
        dropped from the total regardless of the aux_weight argument once the SAE is
        frozen — rescuing a frozen SAE's dead features can't do anything, see freeze_sae().
        """
        recon_loss, l_masked, l_unmasked = self._hierarchical_recon_loss(recon, x, bool_masked_pos)
        total = hierarchical_mse_weight * recon_loss

        if bool_masked_pos is None or not self.sae_frozen:
            total = total + aux_weight * aux_loss
        return total, l_masked, l_unmasked

    def get_metrics(self, sae_hidden=None):
        metrics = {}
        if sae_hidden is not None:
            # sae_hidden: [M, Q, F] — keep the filter axis (dim=1) before the final mean
            # so std-across-filters is available alongside the overall mean, not just a
            # single number that's already averaged every filter's spread away.
            l0_per_filter = (sae_hidden.detach() > 0).float().sum(dim=-1).mean(dim=0)  # [Q]
            metrics['l0_sparsity']     = l0_per_filter.mean().item()
            metrics['l0_sparsity_std'] = l0_per_filter.std().item()
        metrics['dead_feature_rate']          = (self.sae.fire_ema < self.sae.dead_threshold).float().mean().item()
        metrics['decoder_fingerprint_sim']     = self.ema_fingerprint_sim.item()
        metrics['decoder_fingerprint_sim_std'] = self.ema_fingerprint_sim_std.item()

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

        return metrics


class MeSAEFinetune(nn.Module):
    """
    Wraps a pretrained MeSAEPretrain backbone (unmodified) with a per-channel head-attention
    classification head (PerChannelHeadAttn) — same shape/rationale as MeFSQFinetune
    (model/MeFSQ/MeFSQ.py), reading backbone.encode_post_sae instead of encode_post_vq:
    filter_pool broadcasts nothing (unlike a pre-quantization vector), the SAE's sparse
    code genuinely differentiates each filter, so encode_post_sae carries real per-filter
    signal for the classifier's attention pooling to work with.
    """
    def __init__(self, backbone: MeSAEPretrain, num_channels, num_classes, hidden=128, freeze_backbone=False,
                 dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.head = PerChannelHeadAttn(backbone.patch_len, num_channels, num_classes, hidden=hidden, dropout=dropout)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x, coords, time_idx=None, pad_mask=None):
        """
        x: [B, C, N, L]
        coords: [B, C, 3]
        pad_mask: [B, C, N] bool, True = valid (optional, for zero-padded channels)
        returns: (logits [B, num_classes], attn_h [B, C, Q], attn_n [B, C, Q, N], attn_c [B, C])
        """
        z_per_head = self.backbone.encode_post_sae(x, coords, time_idx=time_idx)  # [B, C, N, Q, patch_len]
        logits, attn_h, attn_n, attn_c = self.head(z_per_head, pad_mask=pad_mask)
        return logits, attn_h, attn_n, attn_c
