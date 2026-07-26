import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MeFSQ.MeFSQ_modules import SpatialTemporalEmbeddings, TSAEncoder, MeFSQ, Router, PerChannelHeadAttn, MultiHeadDecoder


class MeFSQPretrain(nn.Module):
    """
    DeepSeekMoE-style shared+routed expert pools, each pool an independent group of
    FSQ experts (down-proj -> quantize -> up-proj -> per-expert decode, summed after
    decode). Shared experts are always active (summed uncapped, DeepSeek-style); routed
    experts are top-k gated per patch by `Router`, weighted-summed over the k selected.

    Routing here buys representation specialization, not FLOPs savings — every routed
    expert is still densely computed and the unselected ones masked to zero before the
    sum (no sparse dispatch/gather). Fine at this scale (patch-level EEG tokenizer, not a
    giant LLM FFN); if `n_routed_experts` grows large enough for compute to matter, real
    sparse dispatch is the upgrade path.
    """
    def __init__(
        self,
        embed_dim=128,
        enc_depth=8,
        mlp_ratio=4.0,
        patch_len=100,
        spatial_heads=8,
        dropout=0.0,
        pool_after_blocks=(),
        upsample_residual_add=True,
        n_routed_experts=64,
        top_k=4,
        n_shared_experts=2,
        routed_r=10,
        routed_num_discrete=3,
        routed_decoder_hidden=None,
        shared_r=16,
        shared_num_discrete=5,
        shared_decoder_hidden=None,
        num_channels=1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.head_dim  = embed_dim
        self.num_channels = num_channels
        self.shared_weight = 0.2  # shared pool's contribution to recon (down-weighted vs routed)
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_total_experts  = n_routed_experts + n_shared_experts

        self.embed      = SpatialTemporalEmbeddings(patch_len, embed_dim)
        self.encoder    = TSAEncoder(embed_dim, depth=enc_depth, num_heads=spatial_heads, mlp_ratio=mlp_ratio,
                                      dropout=dropout, pool_after_blocks=pool_after_blocks,
                                      upsample_residual_add=upsample_residual_add)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        # Every patch's C per-channel D-vectors are concatenated into one C*D token before
        # routing/quantization — one shared code now has to explain all channels jointly
        # (was: one code per channel, C independent codes per patch). Router/MeFSQ/stage_bn
        # below all read this concatenated width; vq_proj_*/decoder_* up-project back down
        # to plain embed_dim first, so their input side is unaffected by C.
        cat_dim = num_channels * embed_dim
        self.router = Router(cat_dim, n_routed_experts, top_k)

        self.stage_bn_routed = nn.BatchNorm1d(n_routed_experts * cat_dim)
        self.mefsq_routed    = MeFSQ(n_routed_experts, routed_r, cat_dim, routed_num_discrete)
        self.vq_proj_routed  = nn.Parameter(torch.empty(n_routed_experts, embed_dim, routed_r))
        # per-head init (not one kaiming_uniform_ call on the whole 3D tensor) — same fan-in
        # fix as MeFSQ.A: avoids r leaking into the fan-in calc and shrinking the init scale
        # as r grows. See model/MeFSQ/MeFSQ_modules.py MeFSQ.__init__ for the full rationale.
        for h in range(n_routed_experts):
            nn.init.kaiming_uniform_(self.vq_proj_routed[h], a=math.sqrt(5))
        self.decoder_routed  = MultiHeadDecoder(n_routed_experts, embed_dim, num_channels * patch_len, hidden=routed_decoder_hidden)

        self.stage_bn_shared = nn.BatchNorm1d(n_shared_experts * cat_dim)
        self.mefsq_shared    = MeFSQ(n_shared_experts, shared_r, cat_dim, shared_num_discrete)
        self.vq_proj_shared  = nn.Parameter(torch.empty(n_shared_experts, embed_dim, shared_r))
        for h in range(n_shared_experts):
            nn.init.kaiming_uniform_(self.vq_proj_shared[h], a=math.sqrt(5))
        self.decoder_shared  = MultiHeadDecoder(n_shared_experts, embed_dim, num_channels * patch_len, hidden=shared_decoder_hidden)

        # encode_pre_vq (read by finetune) broadcasts each channel's own D-dim vector to
        # every expert independently — decoupled from mefsq_routed/shared.norm above, which
        # now normalize the C*D concatenated quantity, not a lone per-channel vector.
        self.pre_vq_norm_routed = nn.LayerNorm(embed_dim)
        self.pre_vq_norm_shared = nn.LayerNorm(embed_dim)

        nn.init.normal_(self.mask_token, std=0.02)

        # EMA buffers for routing health metrics (routed pool only — shared pool doesn't route)
        self.register_buffer('ema_router_entropy',  torch.tensor(0.0))
        self.register_buffer('ema_router_load_std', torch.tensor(0.0))
        self.register_buffer('ema_gate_entropy',    torch.tensor(0.0))

    def enable_spatial(self):
        self.embed.enable_spatial()
        self.encoder.enable_spatial()

    def freeze_vq_and_decoder(self):
        """
        After VQ warmup, lock the entire VQ apparatus — both expert pools (quantizers,
        up-proj, decoders), the router, and the fusion params that feed them (stage_bn).
        Everything left trainable is the main transformer: embed, encoder, mask_token.
        Mirrors LaBraM's split (frozen tokenizer, encoder does the remaining learning),
        except here the tokenizer/decoder were trained jointly with the encoder up to this
        point rather than pretrained standalone first.
        """
        for p in self.mefsq_routed.parameters():
            p.requires_grad_(False)
        for p in self.mefsq_shared.parameters():
            p.requires_grad_(False)
        self.vq_proj_routed.requires_grad_(False)
        self.vq_proj_shared.requires_grad_(False)
        for p in self.decoder_routed.parameters():
            p.requires_grad_(False)
        for p in self.decoder_shared.parameters():
            p.requires_grad_(False)
        for p in self.stage_bn_routed.parameters():
            p.requires_grad_(False)
        for p in self.stage_bn_shared.parameters():
            p.requires_grad_(False)
        for p in self.router.parameters():
            p.requires_grad_(False)

    def stage_features(self, x, coords, time_idx=None, bool_masked_pos=None):
        """
        Runs the full encoder stack and returns only the last block's output.
        Returns z [B, C, N, D].
        """
        z = self.embed(x, coords=coords, time_idx=time_idx)  # [B, C, N, D]

        if bool_masked_pos is not None:
            mask = bool_masked_pos.unsqueeze(-1).type_as(z)  # [B, C, N, 1]
            z = z * (1.0 - mask) + self.mask_token * mask

        return self.encoder(z)  # [B, C, N, D]

    def encode_pre_vq(self, x, coords, time_idx=None):
        """
        The continuous, unquantized per-channel vector right before the pools' concat+`A`
        projection — broadcasts each channel's own D-dim vector to every expert
        independently (unlike forward(), which concatenates all channels into one C*D
        token per patch). Uses dedicated pre_vq_norm_* rather than mefsq_*.norm, since
        those now normalize the concatenated C*D quantity, not a lone per-channel vector.
        For finetune to read features that haven't been through the discrete VQ
        bottleneck. Returns z_h [B, C, N, H_total, D] — shared experts first (indices
        [0, n_shared_experts)), then routed — last-layer vector broadcast to every expert.
        """
        B, C, N, L = x.shape
        D = self.head_dim
        z = self.stage_features(x, coords, time_idx=time_idx, bool_masked_pos=None)  # [B, C, N, D]
        M = B * C * N
        z_flat = z.reshape(M, D)

        Hr = self.n_routed_experts
        z_h_r = z_flat.reshape(M, 1, D).expand(-1, Hr, -1)
        z_h_r = self.pre_vq_norm_routed(z_h_r)

        Hs = self.n_shared_experts
        z_h_s = z_flat.reshape(M, 1, D).expand(-1, Hs, -1)
        z_h_s = self.pre_vq_norm_shared(z_h_s)

        z_h = torch.cat([z_h_s, z_h_r], dim=1)  # [M, H_total, D] — shared first
        return z_h.reshape(B, C, N, self.n_total_experts, D)

    def encode_post_vq(self, x, coords, time_idx=None):
        """
        Real per-head, per-channel signal AFTER quantization: each head's decoder output
        (C*patch_len, jointly reconstructing all channels from that head's own quantized
        code) reshaped per channel, before the cross-head sum. Unlike encode_pre_vq (which
        broadcasts one identical vector to every head — no per-head signal exists yet at
        that point), this is genuinely head-differentiated: each head reads a different
        quantized code and has its own decoder weights, so channel c's slice of head h's
        decode is a real, distinct function of that head's contribution.
        Returns z_h [B, C, N, H_total, patch_len] — shared experts first, then routed
        (same head ordering as encode_pre_vq).
        """
        B, C, N, L = x.shape
        D = self.head_dim
        P = self.patch_len

        z = self.stage_features(x, coords, time_idx=time_idx, bool_masked_pos=None)  # [B, C, N, D]
        CD = self.num_channels * D
        z_cat = z.permute(0, 2, 1, 3).reshape(B, N, CD)  # [B, N, C*D]
        M = B * N
        z_flat = z_cat.reshape(M, CD)

        Hr = self.n_routed_experts
        z_h_r = z_flat.reshape(M, 1, CD).expand(-1, Hr, -1)
        z_h_r = self.stage_bn_routed(z_h_r.reshape(M, Hr * CD)).reshape(M, Hr, CD)
        gate_mask_routed, _ = self.router(z_flat)  # [M, Hr]
        v_q_routed, _, _ = self.mefsq_routed(z_h_r)
        v_q_routed_gated = v_q_routed * gate_mask_routed.unsqueeze(-1)
        z_per_head_r = torch.einsum('mhr,hdr->mhd', v_q_routed_gated, self.vq_proj_routed)
        recon_per_head_r = self.decoder_routed(z_per_head_r)  # [M, Hr, C*P]

        Hs = self.n_shared_experts
        z_h_s = z_flat.reshape(M, 1, CD).expand(-1, Hs, -1)
        z_h_s = self.stage_bn_shared(z_h_s.reshape(M, Hs * CD)).reshape(M, Hs, CD)
        v_q_shared, _, _ = self.mefsq_shared(z_h_s)
        z_per_head_s = torch.einsum('mhr,hdr->mhd', v_q_shared, self.vq_proj_shared)
        recon_per_head_s = self.decoder_shared(z_per_head_s)  # [M, Hs, C*P]

        recon_per_head = torch.cat([recon_per_head_s, recon_per_head_r], dim=1)  # [M, H_total, C*P] — shared first
        recon_per_head = recon_per_head.reshape(B, N, self.n_total_experts, C, P)
        return recon_per_head.permute(0, 3, 1, 2, 4)  # [B, C, N, H_total, P]

    def forward(self, x, coords, time_idx=None, bool_masked_pos=None):
        """
        x: [B, C, N, L]
        coords: [B, C, 3]
        bool_masked_pos: [B, C, N] bool
        returns SimpleNamespace(recon [B,C,N,L], indices_routed [B,N,Hr,r_r],
        indices_shared [B,N,Hs,r_s], v_q_routed [M,Hr,r_r] (gated), v_q_shared [M,Hs,r_s],
        gate_mask_routed [M,Hr], lb_loss scalar) — M=B*N: one shared code per (batch,
        patch), concatenated across all C channels, not one code per channel.
        """
        B, C, N, L = x.shape

        z = self.stage_features(x, coords, time_idx=time_idx, bool_masked_pos=bool_masked_pos)  # [B, C, N, D]

        # Concatenate all C channels' D-dim vectors into one token per (batch, patch) —
        # router/mefsq/decoder below run as a single flat batch of M=B*N tokens, each
        # CD-wide, instead of M=B*C*N tokens each D-wide.
        D = self.head_dim
        CD = self.num_channels * D
        z_cat = z.permute(0, 2, 1, 3).reshape(B, N, CD)  # [B, N, C*D]
        M = B * N
        z_flat = z_cat.reshape(M, CD)

        # ── Routed pool ──
        Hr = self.n_routed_experts
        z_h_r = z_flat.reshape(M, 1, CD).expand(-1, Hr, -1)
        z_h_r = self.stage_bn_routed(z_h_r.reshape(M, Hr * CD)).reshape(M, Hr, CD)

        gate_mask_routed, lb_loss = self.router(z_flat)  # [M, Hr], scalar

        v_q_routed, indices_routed, _ = self.mefsq_routed(z_h_r)          # [M, Hr, r_r]
        v_q_routed_gated = v_q_routed * gate_mask_routed.unsqueeze(-1)    # zero non-selected experts

        z_per_head_r    = torch.einsum('mhr,hdr->mhd', v_q_routed_gated, self.vq_proj_routed)  # [M, Hr, D]
        recon_per_head_r = self.decoder_routed(z_per_head_r)                                     # [M, Hr, C*patch_len]
        recon_routed = recon_per_head_r.sum(dim=1)  # [M, C*patch_len]

        # ── Shared pool: no gating, every expert always active (sum, uncapped) ──
        Hs = self.n_shared_experts
        z_h_s = z_flat.reshape(M, 1, CD).expand(-1, Hs, -1)
        z_h_s = self.stage_bn_shared(z_h_s.reshape(M, Hs * CD)).reshape(M, Hs, CD)

        v_q_shared, indices_shared, _ = self.mefsq_shared(z_h_s)          # [M, Hs, r_s]

        z_per_head_s    = torch.einsum('mhr,hdr->mhd', v_q_shared, self.vq_proj_shared)  # [M, Hs, D]
        recon_per_head_s = self.decoder_shared(z_per_head_s)                               # [M, Hs, C*patch_len]
        recon_shared = recon_per_head_s.sum(dim=1)  # [M, C*patch_len]

        recon = (recon_routed + self.shared_weight * recon_shared).reshape(B, N, C, L).permute(0, 2, 1, 3)  # [B, C, N, L]

        return SimpleNamespace(
            recon=recon,
            indices_routed=indices_routed.reshape(B, N, Hr, v_q_routed.shape[-1]),
            indices_shared=indices_shared.reshape(B, N, Hs, v_q_shared.shape[-1]),
            v_q_routed=v_q_routed_gated,
            v_q_routed_raw=v_q_routed,
            v_q_shared=v_q_shared,
            gate_mask_routed=gate_mask_routed,
            lb_loss=lb_loss,
        )

    def get_loss(self, x, recon, bool_masked_pos, mask_weight=1.0):
        """
        MSE split into masked/unmasked patches.
        Returns (total, l_masked, l_unmasked).
        """
        if bool_masked_pos is not None:
            mask4    = bool_masked_pos.unsqueeze(-1).expand_as(x)
            unmasked = ~mask4
            l_masked   = F.mse_loss(recon[mask4].float(),    x[mask4].float())    if mask4.any()    else recon.new_zeros(1).squeeze()
            l_unmasked = F.mse_loss(recon[unmasked].float(), x[unmasked].float()) if unmasked.any() else recon.new_zeros(1).squeeze()
        else:
            l_masked   = 1.0
            l_unmasked = F.mse_loss(recon.float(), x.float())

        return l_masked * mask_weight + l_unmasked, l_masked, l_unmasked

    @torch.no_grad()
    def update_head_metrics(self, gate_mask_routed):
        """Update EMA monitoring metrics: routing health (routed pool only)."""
        selected = (gate_mask_routed.detach() > 0).float()
        load = selected.mean(dim=0)
        load_p = load / (load.sum() + 1e-8)
        router_entropy = -(load_p * torch.log(load_p + 1e-10)).sum()

        # entropy of the softmax weight distribution AMONG the k selected experts, per
        # patch — low = one expert dominates (confident/peaked routing), high (up to
        # log(k)) = weight split near-uniformly across the k selected.
        gm = gate_mask_routed.detach().float().clamp(min=0)
        gate_entropy = -(gm * torch.log(gm + 1e-10)).sum(dim=-1).mean()

        self.ema_router_load_std.mul_(0.99).add_(load.std(),     alpha=0.01)
        self.ema_router_entropy.mul_(0.99).add_(router_entropy,  alpha=0.01)
        self.ema_gate_entropy.mul_(0.99).add_(gate_entropy,      alpha=0.01)

    @torch.no_grad()
    def get_metrics(self, v_q_routed=None, v_q_shared=None):
        metrics = {}

        def _codebook_health(mefsq, suffix):
            if mefsq.avg_probs.sum() > 0:
                avg_p   = mefsq.avg_probs  # [H, r, N_d]
                entropy = -(avg_p * torch.log(avg_p + 1e-10)).sum(dim=-1)
                metrics[f'codebook_perplexity_{suffix}'] = torch.exp(entropy).mean().item()
                metrics[f'codebook_ste_gap_{suffix}']    = mefsq.ema_ste_gap.item()

        _codebook_health(self.mefsq_routed, 'routed')
        _codebook_health(self.mefsq_shared, 'shared')

        def _head_cosine_sim(vq_proj, suffix):
            H  = vq_proj.shape[0]
            vp = vq_proj.detach().float().reshape(H, -1)
            vp_n = F.normalize(vp, dim=-1)
            sim_hh = vp_n @ vp_n.T
            off_hh = sim_hh * (1 - torch.eye(H, device=sim_hh.device))
            metrics[f'head_cosine_sim_{suffix}'] = off_hh.abs().mean().item()

        _head_cosine_sim(self.vq_proj_routed, 'routed')
        _head_cosine_sim(self.vq_proj_shared, 'shared')

        # Routing health (routed pool only — shared pool doesn't route)
        metrics['router_entropy']  = self.ema_router_entropy.item()
        metrics['router_load_std'] = self.ema_router_load_std.item()
        metrics['gate_entropy']    = self.ema_gate_entropy.item()

        # U-Net skip gate(s): sigmoid(g) in [0,1], 0 = drop skip, 1 = plain add
        if self.encoder.skip_gates is not None:
            for i, g in enumerate(self.encoder.skip_gates):
                metrics[f'skip_gate_{i}'] = torch.sigmoid(g).item()

        return metrics


class MeFSQFinetune(nn.Module):
    """
    Wraps a pretrained MeFSQPretrain backbone (unmodified) with a per-channel head-attention
    classification head (PerChannelHeadAttn).

    Reads backbone.encode_post_vq — each head's own decoded reconstruction, split per
    channel, from AFTER the discrete VQ round-trip (real quantized code, real per-head
    decoder weights). encode_pre_vq (broadcast, pre-quantization) was tried first but
    carries no per-head signal at all — every head sees the identical vector, so the
    classifier's per-head attention pooling had nothing to differentiate and collapsed to
    uniform (see finetune head-attention heatmap: flat color across heads). Post-VQ trades
    that for real per-head signal; if this regresses val accuracy vs the old bypass,
    that's the same "VQ destroys fine detail" tradeoff previously documented — worth
    re-checking against actual numbers here rather than assuming.
    """
    def __init__(self, backbone: MeFSQPretrain, num_channels, num_classes, hidden=128, freeze_backbone=False,
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
        returns: (logits [B, num_classes], attn_h [B, C, H], attn_n [B, C, H, N], attn_c [B, C])
        """
        z_per_head = self.backbone.encode_post_vq(x, coords, time_idx=time_idx)  # [B, C, N, H, patch_len]
        logits, attn_h, attn_n, attn_c = self.head(z_per_head, pad_mask=pad_mask)
        return logits, attn_h, attn_n, attn_c
