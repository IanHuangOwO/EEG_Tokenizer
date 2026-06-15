import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MeFSQ.MeFSQ_modules import SpatialTemporalEmbeddings, TSAEncoder, MeFSQ


class MeFSQPretrain(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        enc_depth=8,
        enc_heads=8,
        mlp_ratio=4.0,
        patch_len=100,
        vq_head_num=64,
        vq_head_vocab_size=16,
        vq_num_discrete=5,
        spatial_heads=8,
        stage_indices=None,
    ):
        super().__init__()
        self.patch_len  = patch_len
        self._stage_indices = sorted(stage_indices or [enc_depth - 1])

        self.embed      = SpatialTemporalEmbeddings(patch_len, embed_dim)
        self.encoder    = TSAEncoder(embed_dim, depth=enc_depth, num_heads=spatial_heads, mlp_ratio=mlp_ratio)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        vq_in_dim  = embed_dim * len(self._stage_indices)
        self.mefsq = MeFSQ(vq_head_num, vq_head_vocab_size, vq_in_dim, vq_num_discrete)

        vq_out_dim   = vq_head_num * vq_head_vocab_size
        self.vq_proj = nn.Linear(vq_out_dim, embed_dim)
        self.decoder  = nn.Linear(embed_dim, patch_len)

        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder.weight, std=0.02)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x, coords, time_idx=None, bool_masked_pos=None):
        """
        x: [B, C, N, L]
        coords: [B, C, 3]
        bool_masked_pos: [B, C, N] bool
        returns: (recon [B,C,N,L], indices [B,C,N,H,r], v_q [B*C,N,H,r])
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
        z_flat = z_cat.reshape(B_C, N, -1)
        v_q, indices, _ = self.mefsq(z_flat)              # [B*C, N, H, r]
        H, r = v_q.shape[2], v_q.shape[3]

        z_q   = self.vq_proj(v_q.reshape(B_C, N, H * r))  # [B*C, N, D]
        recon = self.decoder(z_q).reshape(B, C, N, L)

        return recon, indices.reshape(B, C, N, H, r), v_q

    def get_loss(self, x, recon, bool_masked_pos, mask_weight=1.0):
        """
        MSE split into masked/unmasked patches.
        mask_weight scales l_masked (use 1/mask_ratio to compensate for fewer elements).
        Returns (total, l_masked, l_unmasked).
        """
        if bool_masked_pos is not None:
            mask4    = bool_masked_pos.unsqueeze(-1).expand_as(x)
            unmasked = ~mask4
            l_masked   = F.mse_loss(recon[mask4].float(),    x[mask4].float())    if mask4.any()    else recon.new_zeros(1).squeeze()
            l_unmasked = F.mse_loss(recon[unmasked].float(), x[unmasked].float()) if unmasked.any() else recon.new_zeros(1).squeeze()
        else:
            l_masked   = F.mse_loss(recon.float(), x.float())
            l_unmasked = l_masked

        return l_masked * mask_weight, l_masked, l_unmasked

    @torch.no_grad()
    def get_metrics(self, v_q):
        metrics = {}

        # EMA-based usage health
        if self.mefsq.avg_probs.sum() > 0:
            avg_p   = self.mefsq.avg_probs  # [H, r, N_d]
            entropy = -(avg_p * torch.log(avg_p + 1e-10)).sum(dim=-1)
            ppl     = torch.exp(entropy).mean()
            metrics['codebook_perplexity'] = ppl.item()
            metrics['codebook_ste_gap']    = self.mefsq.ema_ste_gap.item()

        # A matrix geometry: condition number, active rank, avg singular value, eigvec cosim
        A   = self.mefsq.A.float()          # [D, H*r]
        H   = self.mefsq.num_heads
        r   = self.mefsq.r
        A3  = A.view(-1, H, r)              # [D, H, r]

        svs_list, cosim_list = [], []
        for h in range(H):
            U, s, _ = torch.linalg.svd(A3[:, h, :], full_matrices=False)  # U:[D,k], s:[k]
            svs_list.append(s)
            U_n  = F.normalize(U, dim=0)   # [D, k]
            gram = U_n.T @ U_n             # [k, k]
            k    = gram.shape[0]
            off  = gram[~torch.eye(k, dtype=torch.bool, device=gram.device)]
            cosim_list.append(off.abs().mean())

        svs  = torch.stack(svs_list)       # [H, min(D,r)]
        cond = (svs[:, 0] / (svs[:, -1] + 1e-8)).mean()
        p    = svs / (svs.sum(dim=-1, keepdim=True) + 1e-10)
        eff_rank = torch.exp(-(p * torch.log(p + 1e-10)).sum(dim=-1)).mean()
        metrics['codebook_condition_number']   = cond.item()
        metrics['codebook_active_rank']        = eff_rank.item()
        metrics['codebook_avg_singular_value'] = svs.mean().item()
        metrics['codebook_eigvec_cosim']       = torch.stack(cosim_list).mean().item()

        return metrics
