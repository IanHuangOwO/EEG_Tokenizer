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
        vq_k_active=16,
        dropout=0.0,
    ):
        super().__init__()
        self.patch_len  = patch_len
        self.vq_k_active = vq_k_active
        self.vq_head_num = vq_head_num

        self.embed      = SpatialTemporalEmbeddings(patch_len, embed_dim)
        self.encoder    = TSAEncoder(embed_dim, depth=enc_depth, num_heads=spatial_heads, mlp_ratio=mlp_ratio,
                                      dropout=dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        # Every encoder block's output is a candidate stage — no curated subset. Each head
        # gets its OWN softmax-weighted combination over all enc_depth stages (all D-dim) —
        # H*S params (e.g. 50*8=400), not S — so different heads can mix depths/temporal
        # resolutions differently instead of every head reading the same fused vector.
        # Tradeoff (measured): this makes every downstream activation H-times wider (each
        # head now carries its own full D-dim vector into stage_bn/router/VQ instead of one
        # shared D-dim vector), which measurably slows training. Accepted deliberately for
        # the head specialization it buys. Init at 0 -> softmax gives uniform 1/S per head
        # at the start of training.
        self.num_stages = enc_depth
        self.stage_weight = nn.Parameter(torch.zeros(vq_head_num, self.num_stages))
        # BatchNorm per (head, dim) channel right after the weighted sum — with all
        # enc_depth stages now eligible (not a curated, magnitude-matched subset), fused
        # activations from early vs. late blocks can sit at very different scales; this
        # normalizes before router/VQ.
        self.stage_bn = nn.BatchNorm1d(vq_head_num * embed_dim)
        self.mefsq = MeFSQ(vq_head_num, vq_head_vocab_size, embed_dim, vq_num_discrete)

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

        # Router: scores each head per patch from that head's OWN fused input (per-head dot
        # product, consistent with stage_weight/A both being per-head).
        self.router_weight = nn.Parameter(torch.empty(vq_head_num, embed_dim))
        nn.init.normal_(self.router_weight, std=0.01)

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

    def freeze_vq_and_decoder(self):
        """
        After VQ warmup, lock the entire VQ apparatus — quantizer (mefsq: A + norm),
        reconstruction path (vq_proj, decoder), and the routing/fusion params that feed it
        (stage_weight, stage_bn, router_weight; the latter two both depend on stage_weight's
        output distribution, so leaving them trainable while stage_weight is frozen would
        just chase noise in an otherwise-fixed input). Everything left trainable is the main
        transformer: embed, encoder, mask_token. Mirrors LaBraM's split (frozen tokenizer,
        encoder does the remaining learning), except here the tokenizer/decoder were trained
        jointly with the encoder up to this point rather than pretrained standalone first.
        """
        for p in self.mefsq.parameters():
            p.requires_grad_(False)
        self.vq_proj.requires_grad_(False)
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.stage_weight.requires_grad_(False)
        for p in self.stage_bn.parameters():
            p.requires_grad_(False)
        self.router_weight.requires_grad_(False)

    def stage_features(self, x, coords, time_idx=None, bool_masked_pos=None):
        """
        Multi-stage feature extraction. Returns z_stack [B, C, N, S, D] — every encoder
        block's output stacked (not fused), before per-head stage_weight and the mefsq
        quantizer. S == enc_depth.
        """
        z = self.embed(x, coords=coords, time_idx=time_idx)  # [B, C, N, D]

        if bool_masked_pos is not None:
            mask = bool_masked_pos.unsqueeze(-1).type_as(z)  # [B, C, N, 1]
            z = z * (1.0 - mask) + self.mask_token * mask

        stage_outs = []
        for block in self.encoder.blocks:
            z = block(z)
            stage_outs.append(z)

        return torch.stack(stage_outs, dim=-2)  # [B, C, N, S, D]

    def encode_pre_vq(self, x, coords, time_idx=None):
        """
        The continuous, unquantized per-head vector right before MeFSQ's A projection —
        same per-head stage fusion + stage_bn as forward(), plus MeFSQ's own LayerNorm
        applied explicitly here (forward() lets self.mefsq apply it internally). For
        finetune to read features that haven't been through the discrete VQ bottleneck.
        Returns z_h [B, C, N, H, D] — real per-head vectors, not a shared/dummy axis.
        """
        B, C, N, L = x.shape
        H, D = self.vq_head_num, self.head_dim
        z_stack = self.stage_features(x, coords, time_idx=time_idx, bool_masked_pos=None)  # [B, C, N, S, D]
        M = B * C * N
        z_stack = z_stack.reshape(M, self.num_stages, -1)
        stage_w = torch.softmax(self.stage_weight, dim=-1)              # [H, S]
        z_h = torch.einsum('msd,hs->mhd', z_stack, stage_w)             # [M, H, D]
        z_h = self.stage_bn(z_h.reshape(M, H * D)).reshape(M, H, D)
        z_h = self.mefsq.norm(z_h)
        return z_h.reshape(B, C, N, H, D)

    def forward(self, x, coords, time_idx=None, bool_masked_pos=None, use_routing=True, vq_k_active_override=None):
        """
        x: [B, C, N, L]
        coords: [B, C, 3]
        bool_masked_pos: [B, C, N] bool
        use_routing: if False, routing is skipped (VQ warmup)
        vq_k_active_override: int, overrides self.vq_k_active for this forward pass (ramp schedule)
        returns: (recon [B,C,N,L], indices [B,C,N,H,r], v_q [B*C*N,H,r], gate_mask [B*C*N,H], lb_loss scalar)
        """
        B, C, N, L = x.shape

        z_stack = self.stage_features(x, coords, time_idx=time_idx, bool_masked_pos=bool_masked_pos)  # [B, C, N, S, D]

        # N has no cross-patch interaction from here on (router/mefsq/decoder are all
        # elementwise over the batch dim) — fold it into the batch dim alongside B*C so the
        # rest of the pipeline runs as a single flat batch of M=B*C*N independent patches.
        H, D = self.vq_head_num, self.head_dim
        M = B * C * N
        z_stack = z_stack.reshape(M, self.num_stages, -1)                  # [M, S, D]
        stage_w = torch.softmax(self.stage_weight, dim=-1)                 # [H, S]
        z_h = torch.einsum('msd,hs->mhd', z_stack, stage_w)                # [M, H, D] — per-head fused input

        z_h = self.stage_bn(z_h.reshape(M, H * D)).reshape(M, H, D)

        # Routing: top-K head selection per patch, weighted by softmax over the K selected
        # (not a hard 0/1 mask + STE) — weights always sum to 1 regardless of k, so no manual
        # energy-compensation scaling is needed, and gradients flow smoothly through the actual
        # forward values (no forward/backward mismatch from a straight-through estimator).
        k = vq_k_active_override if vq_k_active_override is not None else self.vq_k_active
        if use_routing and k < H:
            gate_logits = torch.einsum('mhd,hd->mh', z_h, self.router_weight)  # [M, H]
            topk_val, topk_idx = gate_logits.topk(k, dim=-1)                # [M, K]
            # softmax is autocast-promoted to fp32 for stability — cast back to match
            # gate_logits' dtype (may be fp16 under AMP) before scatter
            topk_weight = torch.softmax(topk_val, dim=-1).to(gate_logits.dtype)  # normalized over the K selected
            gate_mask = torch.zeros_like(gate_logits).scatter_(-1, topk_idx, topk_weight)  # [M, H]

            # Load balance loss: Switch-Transformer style (= 1.0 at uniform routing).
            # f = selection frequency (detached, no grad); p = dense softmax over ALL heads
            # (not just the selected K) so non-selected heads still get gradient signal —
            # without this, a head that's never in the top-k would never get pushed back in.
            f = (gate_mask.detach() > 0).float().mean(dim=0)               # [H] routing freq
            p = torch.softmax(gate_logits, dim=-1).mean(dim=0)             # [H] dense prob (has grad)
            lb_loss = H * ((f / (f.sum() + 1e-8)) * (p / (p.sum() + 1e-8))).sum()
            self._last_gate_logits = gate_logits.detach()                  # stash for viz (router importance)
        else:
            gate_mask = torch.ones(M, H, device=z_h.device, dtype=z_h.dtype)
            lb_loss   = z_h.new_zeros(1).squeeze()
            self._last_gate_logits = None

        v_q, indices, _ = self.mefsq(z_h)                     # [M, H, r]
        v_q_gated = v_q * gate_mask.unsqueeze(-1)            # zero non-selected heads, softmax-weighted otherwise

        z_per_head    = torch.einsum('mhr,hdr->mhd', v_q_gated, self.vq_proj)  # [M, H, embed_dim]
        recon_per_head = self.decoder(z_per_head)                                # [M, H, patch_len] — each head decodes independently
        recon = recon_per_head.sum(dim=1).reshape(B, C, N, L)                    # combine AFTER decode

        return recon, indices.reshape(B, C, N, H, v_q.shape[-1]), v_q_gated, gate_mask, lb_loss

    def _head_fingerprints(self, v_q_gated, gate_mask, B, C):
        """
        Compute gate-weighted mean+var fingerprint per head.
        v_q_gated: [M, H, r]  (M = B*C*N)
        gate_mask: [M, H]     (hard 0/1 in forward)
        Returns fp [H, 2*C*F]
        """
        M, H, r = v_q_gated.shape
        N = M // (B * C)

        z_all     = torch.einsum('mhr,hdr->mhd', v_q_gated, self.vq_proj)     # [M, H, embed_dim]
        recon_all = self.decoder(z_all)                                      # [M, H, P] — decoder is already per-head & vectorized

        fft_c   = torch.fft.rfft(recon_all.float(), dim=-1)
        fft_all = (fft_c.real.pow(2) + fft_c.imag.pow(2)).clamp(min=1e-8).sqrt()  # [M, H, F]
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
        H = v_q_gated.shape[1]
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
        load = selected.mean(dim=0)
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
    classification head (PerChannelHeadAttn).

    pre_vq=False (default): recomputes each head's projected embedding from the backbone's
    own v_q_gated/vq_proj — classification sees the discrete, quantized per-head codes.

    pre_vq=True: reads backbone.encode_pre_vq instead — the continuous, unquantized
    per-head vector right before MeFSQ's A projection (real H heads, each with its own
    stage-weighted fusion — not a shared/dummy axis). Fed to PerChannelHeadAttn exactly
    like the quantized path, just skipping the VQ round-trip. Use when the discrete VQ
    codebook is suspected of destroying task-relevant fine detail the classifier needs.
    """
    def __init__(self, backbone: MeFSQPretrain, num_channels, num_classes, hidden=128, freeze_backbone=False,
                 dropout=0.1, pre_vq=False):
        super().__init__()
        self.backbone = backbone
        self.pre_vq = pre_vq
        self.head = PerChannelHeadAttn(backbone.head_dim, num_channels, num_classes, hidden=hidden, dropout=dropout)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x, coords, time_idx=None, pad_mask=None, use_routing=True, vq_k_active_override=None,
                return_head_stats=False):
        """
        x: [B, C, N, L]
        coords: [B, C, 3]
        pad_mask: [B, C, N] bool, True = valid (optional, for zero-padded channels)
        returns: (logits [B, num_classes], attn_h [B, C, H], attn_n [B, C, H, N], lb_loss scalar)
        if return_head_stats: also (v_q_gated, gate_mask, B, C) for head/fingerprint diversity
        monitoring — both None in pre_vq mode, since there's no VQ routing to monitor.
        """
        B, C, N, L = x.shape
        bb = self.backbone

        if self.pre_vq:
            z_per_head = bb.encode_pre_vq(x, coords, time_idx=time_idx)  # [B, C, N, H, D]
            logits, attn_h, attn_n = self.head(z_per_head, pad_mask=pad_mask)
            lb_loss = x.new_zeros(())
            v_q_gated, gate_mask = None, None
        else:
            _, _, v_q_gated, gate_mask, lb_loss = bb(
                x, coords, time_idx=time_idx, bool_masked_pos=None,
                use_routing=use_routing, vq_k_active_override=vq_k_active_override,
            )
            z_per_head = torch.einsum('mhr,hdr->mhd', v_q_gated, bb.vq_proj).view(B, C, N, bb.vq_head_num, bb.head_dim)
            logits, attn_h, attn_n = self.head(z_per_head, pad_mask=pad_mask)  # attn_h: [B, C, H], attn_n: [B, C, H, N]

        if return_head_stats:
            return logits, attn_h, attn_n, lb_loss, v_q_gated, gate_mask, B, C
        return logits, attn_h, attn_n, lb_loss
