import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MeFSQ.MeFSQ_modules import SpatialTemporalEmbeddings, TSAEncoder, MeFSQ


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

        assert embed_dim % vq_head_num == 0, \
            f"embed_dim ({embed_dim}) must be divisible by vq_head_num ({vq_head_num})"
        self.head_dim = embed_dim // vq_head_num
        self.vq_proj  = nn.Parameter(torch.empty(vq_head_num, self.head_dim, vq_head_vocab_size))
        self.vq_bias  = nn.Parameter(torch.zeros(embed_dim))
        nn.init.kaiming_uniform_(self.vq_proj, a=math.sqrt(5))
        self.decoder  = nn.Linear(embed_dim, patch_len)

        # Router: scores each head per patch from pre-VQ features
        self.router = nn.Linear(vq_in_dim, vq_head_num, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)

        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder.weight, std=0.02)
        nn.init.zeros_(self.decoder.bias)

        # EMA buffers for routing + fingerprint health metrics
        self.register_buffer('ema_router_entropy',  torch.tensor(0.0))
        self.register_buffer('ema_router_load_std', torch.tensor(0.0))
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

        # Routing: top-K head selection per patch
        H = self.vq_head_num
        k = k_active_override if k_active_override is not None else self.k_active
        if use_routing and k < H:
            gate_logits = self.router(z_flat)                 # [B*C, N, H]
            topk_idx    = gate_logits.topk(k, dim=-1).indices               # [B*C, N, K]
            hard_mask   = torch.zeros_like(gate_logits).scatter_(-1, topk_idx, 1.0)
            gate_soft   = torch.sigmoid(gate_logits)
            # STE: hard forward, soft gradient
            gate_mask   = hard_mask + gate_soft - gate_soft.detach()       # [B*C, N, H]
            # Load balance loss: normalized Switch-Transformer style (= 1.0 at uniform routing)
            # ponytail: detach f so only the router's soft probs get gradient, not the hard selection
            f     = hard_mask.detach().mean(dim=[0, 1])                    # [H] routing freq
            p     = gate_soft.mean(dim=[0, 1])                             # [H] mean prob (has grad)
            lb_loss = H * ((f / (f.sum() + 1e-8)) * (p / (p.sum() + 1e-8))).sum()
            self._last_gate_logits = gate_logits.detach()                  # stash for viz (router importance)
        else:
            gate_mask = torch.ones(B_C, N, H, device=z_flat.device, dtype=z_flat.dtype)
            lb_loss   = z_flat.new_zeros(1).squeeze()
            self._last_gate_logits = None

        v_q, indices, _ = self.mefsq(z_flat)                 # [B*C, N, H, r]
        v_q_gated = v_q * gate_mask.unsqueeze(-1)            # zero non-selected heads
        # ponytail: inverted-dropout scaling — keeps z_q energy constant as k decreases
        if use_routing and k < H:
            v_q_gated = v_q_gated * (H / k)

        z_q = torch.einsum('bnhr,hdr->bnhd', v_q_gated, self.vq_proj).reshape(B_C, N, H * self.head_dim) + self.vq_bias
        recon = self.decoder(z_q).reshape(B, C, N, L)

        return recon, indices.reshape(B, C, N, H, v_q.shape[3]), v_q_gated, gate_mask, lb_loss

    def _head_fingerprints(self, v_q_gated, gate_mask, B, C):
        """
        Compute gate-weighted mean+var fingerprint per head.
        v_q_gated: [B*C, N, H, r]
        gate_mask:  [B*C, N, H]  (hard 0/1 in forward)
        Returns fp [H, 2*C*F]
        """
        H, N = v_q_gated.shape[2], v_q_gated.shape[1]
        d = self.head_dim

        z_all     = torch.einsum('bnhr,hdr->bnhd', v_q_gated, self.vq_proj)
        W_all     = self.decoder.weight.reshape(self.patch_len, H, d).permute(1, 0, 2)
        recon_all = torch.einsum('bnhd,hpd->bnhp', z_all, W_all)              # [B*C, N, H, P]

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

    @torch.no_grad()
    def update_head_metrics(self, v_q_gated, gate_mask, B, C):
        """Update EMA monitoring metrics: fingerprint similarity + routing health."""
        H = v_q_gated.shape[2]
        fp   = self._head_fingerprints(v_q_gated, gate_mask, B, C)   # [H, 2*C*F]
        fp_n = F.normalize(fp, dim=-1)
        sim  = fp_n @ fp_n.T                                          # [H, H]
        eye  = torch.eye(H, device=sim.device)
        off_abs = (sim * (1 - eye)).abs()
        fp_sim_mean = off_abs.sum() / (H * (H - 1))
        fp_sim_std  = off_abs[~eye.bool()].std()

        hard_mask = (gate_mask.detach() > 0.5).float()
        load = hard_mask.mean(dim=[0, 1])
        load_p = load / (load.sum() + 1e-8)
        router_entropy = -(load_p * torch.log(load_p + 1e-10)).sum()

        self.ema_fp_sim_mean.mul_(0.99).add_(fp_sim_mean,        alpha=0.01)
        self.ema_fp_sim_std.mul_(0.99).add_(fp_sim_std,          alpha=0.01)
        self.ema_router_load_std.mul_(0.99).add_(load.std(),     alpha=0.01)
        self.ema_router_entropy.mul_(0.99).add_(router_entropy,  alpha=0.01)

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

        return metrics
