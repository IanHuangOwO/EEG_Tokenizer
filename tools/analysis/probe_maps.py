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

from model.MeSAE.MeSAE_modules import _entry_dim, feature_names


def readout(ckpt, entry):
    """Readout weight of one entry's feature block, BatchNorm scale folded in -> [classes, entry features]. The head
    concatenates its entries in features order (FeatureHead), so the block starts after the earlier entries' widths."""
    sd, hc = ckpt['model_state_dict'], ckpt['head_config']
    names = feature_names(hc)
    lo = sum(_entry_dim(hc, n) for n in names[:names.index(entry)])
    hi = lo + _entry_dim(hc, entry)
    scale = sd['cls.0.weight'] / (sd['cls.0.running_var'] + 1e-5).sqrt()      # BatchNorm at eval
    return (sd['cls.2.weight'] * scale)[:, lo:hi].float()


def head_maps(head_pth, entry='latent_signed'):
    """-> (importance [K, N], spatial [K, C]) for one head, virtual channels ranked by importance."""
    ckpt = torch.load(head_pth, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    q = sd[f'entries.{entry}.q'].float()                                     # [R, N]
    S = sd[f'spatials.{entry}.weight'].float()                               # [K, C]
    K, R = S.shape[0], q.shape[0]
    M = sd[f'entries.{entry}.proj.weight'].shape[0]
    w = readout(ckpt, entry).view(-1, K, M, R)                               # [classes, K, M, R]
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


def stamp_head_maps(head_pth, entry='stamp_power'):
    """A stamp_power entry: features f[k, s] = log(sum_n w[s, n] (S a)^2 + (S b)^2) with S the spatial filter [K, C] and
    w the learned per-atom softmax time weights. -> (time weights [S, N], importance [K, S] = class-centred readout norm,
    channel map [S, C] = sum_k importance[k, s] * S[k, c]^2 / sum_k importance[k, s]: which electrodes' power the
    decision on Q-atom s reads, invariant to the order and sign of the virtual channels)."""
    ckpt = torch.load(head_pth, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    p, q = sd[f'entries.{entry}.time.p'].float(), sd[f'entries.{entry}.time.q'].float()
    tw = torch.softmax(torch.einsum('rs,rn->sn', p, q), dim=-1)            # [S, N]
    S = sd[f'spatials.{entry}.weight'].float()                               # [K, C]
    w = readout(ckpt, entry).view(-1, S.shape[0], p.shape[1])                # [classes, K, S]
    imp = (w - w.mean(0, keepdim=True)).norm(dim=0)                          # [K, S]
    chan = torch.einsum('ks,kc->sc', imp, S.pow(2)) / imp.sum(0)[:, None].clamp(min=1e-12)
    return tw, imp, chan


def summarise_stamps(head_paths, entry='stamp_power'):
    """Mean over heads of one backbone (Q-atoms are that backbone's own dictionary): time weights [S, N], per-atom
    importance [S] (summed over virtual channels), channel map [S, C] (each head's rows scaled to max 1 first)."""
    tws, imps, chans = [], [], []
    for h in head_paths:
        tw, imp, chan = stamp_head_maps(h, entry)
        tws.append(tw); imps.append(imp.sum(0)); chans.append(chan / chan.amax(1, keepdim=True).clamp(min=1e-12))
    return torch.stack(tws).mean(0), torch.stack(imps).mean(0), torch.stack(chans).mean(0), len(tws)


def gain_scale(amp, spatial, n_trials=2000, seed=0):
    """Std of the spatially filtered Q-atom gains [2, K, S] (a, b) over trials and patches: the input scale each signed_ab
    readout weight multiplies. amp [T, N', C, S, 2] (the feature cache), spatial [K, C]; a random subset of trials."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(amp.shape[0], generator=g)[:n_trials]
    x = torch.einsum('kc,tncsj->jtnks', spatial, amp[idx].float())           # [2, T', N', K, S]
    return x.flatten(1, 2).std(1)                                             # [2, K, S]


def signed_stamp_head_maps(head_pth, amp, entry='signed_ab'):
    """A signed_ab entry in Q-atom space: features f[ab, k, m, r] = sum_n q[r, n] sum_s W[m, s] (S g_ab)[n, k, s], so the
    class-centred effective weight per Q-atom and patch is E[c, ab, k, n, s] = sum_{m, r} w[c, ab, k, m, r] q[r, n] W[m, s],
    times the Q-atom's filtered gain std (a weight on a quiet Q-atom moves the decision little).
    -> (importance [S, N] = norm over class, a / b and virtual channel; per-atom importance [K, S] over the rest;
    channel map [S, C] weighted like stamp_head_maps)."""
    ckpt = torch.load(head_pth, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    q, W = sd[f'entries.{entry}.q'].float(), sd[f'entries.{entry}.stamp.weight'].float()   # [R, N], [M, S]
    S = sd[f'spatials.{entry}.weight'].float()                                            # [K, C]
    K, (M, R) = S.shape[0], (W.shape[0], q.shape[0])
    w = readout(ckpt, entry).view(-1, 2, K, M, R)
    E = torch.einsum('cjkmr,rn,ms->cjkns', w, q, W) * gain_scale(amp, S)[None, :, :, None, :]
    E = E - E.mean(0, keepdim=True)
    tw = E.pow(2).sum((0, 1, 2)).sqrt().T                                                 # [S, N]
    imp = E.pow(2).sum((0, 1, 3)).sqrt()                                                  # [K, S]
    chan = torch.einsum('ks,kc->sc', imp, S.pow(2)) / imp.sum(0)[:, None].clamp(min=1e-12)
    return tw, imp, chan


def summarise_signed_stamps(head_paths, amp, entry='signed_ab'):
    """Mean over heads of one backbone: importance map [S, N] (each head's map scaled to sum 1 first), per-atom
    importance [S] (each head's scaled to sum 1), channel map [S, C] (rows scaled to max 1)."""
    tws, imps, chans = [], [], []
    for h in head_paths:
        tw, imp, chan = signed_stamp_head_maps(h, amp, entry)
        tws.append(tw / tw.sum()); i = imp.sum(0); imps.append(i / i.sum())
        chans.append(chan / chan.amax(1, keepdim=True).clamp(min=1e-12))
    return torch.stack(tws).mean(0), torch.stack(imps).mean(0), torch.stack(chans).mean(0), len(tws)
