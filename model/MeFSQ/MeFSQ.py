import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MeFSQ.MeFSQ_modules import SpatialTemporalEmbeddings, TSAEncoder, MeFSQ, PerChannelHeadAttn, MultiHeadDecoder


class MeFSQPretrain(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        enc_depth=8,
        mlp_ratio=4.0,
        patch_len=100,
        vq_head_num=64,
        vq_head_vocab_size=16,
        vq_num_discrete=5,
        spatial_heads=8,
        stage_indices=None,
        k_active=16,
    ):
        super().__init__()
        self.patch_len  = patch_len
        self.k_active   = k_active
        self.vq_head_num = vq_head_num
        self._stage_indices = sorted(stage_indices or [enc_depth - 1])

        self.embed      = SpatialTemporalEmbeddings(patch_len, embed_dim)
        self.encoder    = TSAEncoder(embed_dim, depth=enc_depth, num_heads=spatial_heads, mlp_ratio=mlp_ratio)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        vq_in_dim  = embed_dim * len(self._stage_indices)
        self.mefsq = MeFSQ(vq_head_num, vq_head_vocab_size, vq_in_dim, vq_num_discrete)

        # Each head projects its own r-dim code to the FULL embed_dim (not a disjoint slice) —
        # heads are independent full-width sub-embeddings. head_dim == embed_dim; kept as its
        # own attribute since downstream code (finetune head, viz) refers to "the width of
        # one head's output".
        self.head_dim = embed_dim
        self.vq_proj  = nn.Parameter(torch.empty(vq_head_num, self.head_dim, vq_head_vocab_size))
        nn.init.kaiming_uniform_(self.vq_proj, a=math.sqrt(5))
        # each head decodes independently (own weights, own hidden-layer nonlinearity) and
        # heads combine by SUMMING their decoded reconstructions — see forward()
        self.decoder  = MultiHeadDecoder(vq_head_num, embed_dim, patch_len)

        # Router: scores each head per patch from pre-VQ features
        self.router = nn.Linear(vq_in_dim, vq_head_num, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)

        nn.init.normal_(self.mask_token, std=0.02)

        # EMA buffers for routing + fingerprint health metrics
        self.register_buffer('ema_router_entropy',  torch.tensor(0.0))
        self.register_buffer('ema_router_load_std', torch.tensor(0.0))
        self.register_buffer('ema_gate_entropy',    torch.tensor(0.0))
        self.register_buffer('ema_fp_sim_mean',     torch.tensor(0.0))
        self.register_buffer('ema_fp_sim_std',      torch.tensor(0.0))

    def enable_spatial(self):
        self.embed.enable_spatial()
        self.encoder.enable_spatial()

    def forward(self, x, coords, time_idx=None, bool_masked_pos=None, use_routing=True, k_active_override=None):
        """
        x: [B, C, N, L]
        coords: [B, C, 3]
        bool_masked_pos: [B, C, N] bool
        use_routing: if False, routing is skipped (VQ warmup)
        k_active_override: int, overrides self.k_active for this forward pass (ramp schedule)
        returns: (recon [B,C,N,L], indices [B,C,N,H,r], v_q [B*C,N,H,r], gate_mask [B*C,N,H], lb_loss scalar)
        """
        B, C, N, L = x.shape

        z = self.embed(x, coords=coords, time_idx=time_idx)  # [B, C, N, D]

        if bool_masked_pos is not None:
            mask = bool_masked_pos.unsqueeze(-1).type_as(z)  # [B, C, N, 1]
            z = z * (1.0 - mask) + self.mask_token * mask

        needed = set(self._stage_indices)
        stage_outs = {}
        for i, block in enumerate(self.encoder.blocks):
            z = block(z)
            if i in needed:
                stage_outs[i] = z

        z_cat = torch.cat([stage_outs[si] for si in self._stage_indices], dim=-1)  # [B, C, N, D*S]

        B_C = B * C
        z_flat = z_cat.reshape(B_C, N, -1)                   # [B*C, N, D*S]

        # Routing: top-K head selection per patch, weighted by softmax over the K selected
        # (not a hard 0/1 mask + STE) — weights always sum to 1 regardless of k, so no manual
        # energy-compensation scaling is needed, and gradients flow smoothly through the actual
        # forward values (no forward/backward mismatch from a straight-through estimator).
        H = self.vq_head_num
        k = k_active_override if k_active_override is not None else self.k_active
        if use_routing and k < H:
            gate_logits = self.router(z_flat)                              # [B*C, N, H]
            topk_val, topk_idx = gate_logits.topk(k, dim=-1)                # [B*C, N, K]
            # softmax is autocast-promoted to fp32 for stability — cast back to match
            # gate_logits' dtype (may be fp16 under AMP) before scatter
            topk_weight = torch.softmax(topk_val, dim=-1).to(gate_logits.dtype)  # normalized over the K selected
            gate_mask = torch.zeros_like(gate_logits).scatter_(-1, topk_idx, topk_weight)  # [B*C, N, H]

            # Load balance loss: Switch-Transformer style (= 1.0 at uniform routing).
            # f = selection frequency (detached, no grad); p = dense softmax over ALL heads
            # (not just the selected K) so non-selected heads still get gradient signal —
            # without this, a head that's never in the top-k would never get pushed back in.
            f = (gate_mask.detach() > 0).float().mean(dim=[0, 1])          # [H] routing freq
            p = torch.softmax(gate_logits, dim=-1).mean(dim=[0, 1])        # [H] dense prob (has grad)
            lb_loss = H * ((f / (f.sum() + 1e-8)) * (p / (p.sum() + 1e-8))).sum()
            self._last_gate_logits = gate_logits.detach()                  # stash for viz (router importance)
        else:
            gate_mask = torch.ones(B_C, N, H, device=z_flat.device, dtype=z_flat.dtype)
            lb_loss   = z_flat.new_zeros(1).squeeze()
            self._last_gate_logits = None

        v_q, indices, _ = self.mefsq(z_flat)                 # [B*C, N, H, r]
        v_q_gated = v_q * gate_mask.unsqueeze(-1)            # zero non-selected heads, softmax-weighted otherwise

        z_per_head    = torch.einsum('bnhr,hdr->bnhd', v_q_gated, self.vq_proj)  # [B_C, N, H, embed_dim]
        recon_per_head = self.decoder(z_per_head)                                # [B_C, N, H, patch_len] — each head decodes independently
        recon = recon_per_head.sum(dim=2).reshape(B, C, N, L)                    # combine AFTER decode

        return recon, indices.reshape(B, C, N, H, v_q.shape[3]), v_q_gated, gate_mask, lb_loss

    def _head_fingerprints(self, v_q_gated, gate_mask, B, C):
        """
        Compute gate-weighted mean+var fingerprint per head.
        v_q_gated: [B*C, N, H, r]
        gate_mask:  [B*C, N, H]  (hard 0/1 in forward)
        Returns fp [H, 2*C*F]
        """
        H, N = v_q_gated.shape[2], v_q_gated.shape[1]

        z_all     = torch.einsum('bnhr,hdr->bnhd', v_q_gated, self.vq_proj)   # [B*C, N, H, embed_dim]
        recon_all = self.decoder(z_all)                                      # [B*C, N, H, P] — decoder is already per-head & vectorized

        fft_c   = torch.fft.rfft(recon_all.float(), dim=-1)
        fft_all = (fft_c.real.pow(2) + fft_c.imag.pow(2)).clamp(min=1e-8).sqrt()  # [B*C, N, H, F]
        fft_all = fft_all.reshape(B, C, N, H, -1)                             # [B, C, N, H, F]

        # Gate-weighted mean per (C, H): normalize weights over (B, N) independently
        gm    = gate_mask.reshape(B, C, N, H).unsqueeze(-1).float()           # [B, C, N, H, 1]
        w_sum = gm.sum(dim=[0, 2], keepdim=True)                              # [1, C, 1, H, 1]
        w     = gm / (w_sum + 1e-8)                                           # [B, C, N, H, 1]

        fp_mean = (fft_all * w).sum(dim=[0, 2])                               # [C, H, F]
        fp_sq   = (fft_all.pow(2) * w).sum(dim=[0, 2])                        # [C, H, F]
        fp_var  = (fp_sq - fp_mean.pow(2)).clamp(min=0)                       # [C, H, F]

        fp_mean = fp_mean.permute(1, 0, 2).reshape(H, -1)                     # [H, C*F]
        fp_var  = fp_var.permute(1, 0, 2).reshape(H, -1)                      # [H, C*F]

        fp = torch.cat([F.normalize(fp_mean, dim=-1),
                        F.normalize(fp_var,  dim=-1)], dim=-1)                # [H, 2*C*F]
        return fp

    def get_diversity_loss(self, v_q_gated, gate_mask, B, C, std_weight=0.5):
        """
        Differentiable head-diversity loss: penalizes high average pairwise fingerprint
        cosine similarity (heads too similar) and REWARDS spread in that similarity
        (subtracted, weighted by std_weight) — directly opposes the collapse signature
        seen without this loss (sim mean -> ~0.9, std -> ~0.1, i.e. every pair of heads
        becomes equally, highly similar). Also updates the EMA monitoring buffers from
        this same computation so nothing is computed twice.
        """
        H = v_q_gated.shape[2]
        fp   = self._head_fingerprints(v_q_gated, gate_mask, B, C)   # [H, 2*C*F], WITH grad
        fp_n = F.normalize(fp, dim=-1)
        sim  = fp_n @ fp_n.T                                          # [H, H]
        eye  = torch.eye(H, device=sim.device)
        off_abs = (sim * (1 - eye)).abs()
        fp_sim_mean = off_abs.sum() / (H * (H - 1))
        fp_sim_std  = off_abs[~eye.bool()].std()

        with torch.no_grad():
            self.ema_fp_sim_mean.mul_(0.99).add_(fp_sim_mean, alpha=0.01)
            self.ema_fp_sim_std.mul_(0.99).add_(fp_sim_std,   alpha=0.01)

        return fp_sim_mean - std_weight * fp_sim_std

    @torch.no_grad()
    def update_head_metrics(self, gate_mask):
        """Update EMA monitoring metrics: routing health."""
        # gate_mask is now a softmax weight (typically << 0.5 per selected head, not a
        # hard 0/1) — "selected" means nonzero, not "above 0.5"
        selected = (gate_mask.detach() > 0).float()
        load = selected.mean(dim=[0, 1])
        load_p = load / (load.sum() + 1e-8)
        router_entropy = -(load_p * torch.log(load_p + 1e-10)).sum()

        # entropy of the softmax weight distribution AMONG the k selected heads, per patch —
        # low = one head dominates (confident/peaked routing), high (up to log(k)) = weight
        # split near-uniformly across the k selected. A new axis top-k softmax makes possible:
        # the old hard-mask scheme always gave every selected head weight exactly 1.
        # .float() first: gate_mask may be fp16 under autocast, where 1e-10 underflows to
        # exactly 0 — log(0)=-inf and 0*(-inf)=nan. fp32 makes the epsilon representable.
        gm = gate_mask.detach().float().clamp(min=0)
        gate_entropy = -(gm * torch.log(gm + 1e-10)).sum(dim=-1).mean()

        self.ema_router_load_std.mul_(0.99).add_(load.std(),     alpha=0.01)
        self.ema_router_entropy.mul_(0.99).add_(router_entropy,  alpha=0.01)
        self.ema_gate_entropy.mul_(0.99).add_(gate_entropy,      alpha=0.01)

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
    def get_metrics(self, v_q_gated):
        metrics = {}

        # Codebook health
        if self.mefsq.avg_probs.sum() > 0:
            avg_p   = self.mefsq.avg_probs  # [H, r, N_d]
            entropy = -(avg_p * torch.log(avg_p + 1e-10)).sum(dim=-1)
            metrics['codebook_perplexity'] = torch.exp(entropy).mean().item()
            metrics['codebook_ste_gap']    = self.mefsq.ema_ste_gap.item()

        # Head projection diversity (vq_proj cosine sim)
        H  = self.vq_proj.shape[0]
        vp = self.vq_proj.detach().float().reshape(H, -1)
        vp_n = F.normalize(vp, dim=-1)
        sim_hh = vp_n @ vp_n.T
        off_hh = sim_hh * (1 - torch.eye(H, device=sim_hh.device))
        metrics['head_cosine_sim'] = off_hh.abs().mean().item()

        # Fingerprint similarity (from EMA updated in get_separation_loss)
        metrics['fp_sim_mean']     = self.ema_fp_sim_mean.item()
        metrics['fp_sim_std']      = self.ema_fp_sim_std.item()

        # Routing health
        metrics['router_entropy']  = self.ema_router_entropy.item()
        metrics['router_load_std'] = self.ema_router_load_std.item()
        metrics['gate_entropy']    = self.ema_gate_entropy.item()

        return metrics


class MeFSQFinetune(nn.Module):
    """
    Wraps a pretrained MeFSQPretrain backbone (unmodified) with a per-channel head-attention
    classification head (PerChannelHeadAttn). Recomputes each head's projected embedding
    from the backbone's own v_q_gated/vq_proj rather than touching MeFSQPretrain.forward's
    return contract — classification bypasses the backbone's per-head decoder entirely.
    """
    def __init__(self, backbone: MeFSQPretrain, num_channels, num_classes, hidden=128, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.head = PerChannelHeadAttn(backbone.head_dim, num_channels, num_classes, hidden=hidden)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x, coords, time_idx=None, pad_mask=None, use_routing=True, k_active_override=None,
                return_head_stats=False):
        """
        x: [B, C, N, L]
        coords: [B, C, 3]
        pad_mask: [B, C, N] bool, True = valid (optional, for zero-padded channels)
        returns: (logits [B, num_classes], attn_h [B, C, H], attn_n [B, C, H, N], lb_loss scalar)
        if return_head_stats: also (v_q_gated, gate_mask, B, C) for head/fingerprint diversity monitoring
        """
        B, C, N, L = x.shape

        _, _, v_q_gated, gate_mask, lb_loss = self.backbone(
            x, coords, time_idx=time_idx, bool_masked_pos=None,
            use_routing=use_routing, k_active_override=k_active_override,
        )

        bb = self.backbone
        z_per_head = torch.einsum('bnhr,hdr->bnhd', v_q_gated, bb.vq_proj).view(B, C, N, bb.vq_head_num, bb.head_dim)
        logits, attn_h, attn_n = self.head(z_per_head, pad_mask=pad_mask)  # attn_h: [B, C, H], attn_n: [B, C, H, N]

        if return_head_stats:
            return logits, attn_h, attn_n, lb_loss, v_q_gated, gate_mask, B, C
        return logits, attn_h, attn_n, lb_loss
