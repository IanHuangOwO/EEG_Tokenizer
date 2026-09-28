"""
Where a trained linear probe (a `latent_signed` head entry on z) reads from: per virtual channel (the
head's spatial filter, spatial_k rows) and patch, how much the class decision depends on it.

A latent_signed head is linear in z: features f[k, m, r] = sum_n q[r, n] (P z)[n, k, m] with
k = spatial filter, m = PCA axis, r = time filter, then BatchNorm (a per-feature affine at eval) and a
linear readout. Mapping the readout back through the BatchNorm scale and the time filters gives an
effective weight E[c, k, n, m] = sum_r w[c, k, m, r] * gamma / sqrt(var + eps) * q[r, n]; it is
centred over classes (softmax ignores a common shift) and its norm over (class, m) is the
importance[k, n] of virtual channel k at patch n.

Several heads (folds, seeds) are summarised together: virtual channels come out in arbitrary order and
sign per head, so each head's channels are ranked by total importance, and each ranked spatial filter
is sign-aligned to the first head's before averaging.
"""
import torch


def head_maps(head_pth, entry='latent_signed'):
    """-> (importance [K, N], spatial [K, C]) for one head, virtual channels ranked by importance."""
    sd = torch.load(head_pth, map_location='cpu', weights_only=False)['model_state_dict']
    q = sd[f'entries.{entry}.q'].float()                                     # [R, N]
    S = sd[f'spatials.{entry}.weight'].float()                               # [K, C]
    M = sd[f'entries.{entry}.proj.weight'].shape[0]
    K, R = S.shape[0], q.shape[0]
    scale = sd['cls.0.weight'] / (sd['cls.0.running_var'] + 1e-5).sqrt()      # BatchNorm at eval
    w = (sd['cls.2.weight'] * scale).view(-1, K, M, R)                       # [classes, K, M, R]
    E = torch.einsum('ckmr,rn->cknm', w, q)
    E = E - E.mean(0, keepdim=True)                                          # class-centred
    imp = E.pow(2).sum((0, 3)).sqrt()                                        # [K, N]
    order = imp.sum(1).argsort(descending=True)
    return imp[order], S[order]


def summarise(head_paths, entry='latent_signed'):
    """-> (mean importance [K, N], mean sign-aligned spatial filter [K, C], number of heads)."""
    imps, sps = [], []
    for p in head_paths:
        imp, sp = head_maps(p, entry)
        if sps:
            sign = torch.sign((sp * sps[0]).sum(1, keepdim=True))
            sp = sp * torch.where(sign == 0, torch.ones_like(sign), sign)
        imps.append(imp)
        sps.append(sp)
    return torch.stack(imps).mean(0), torch.stack(sps).mean(0), len(imps)
