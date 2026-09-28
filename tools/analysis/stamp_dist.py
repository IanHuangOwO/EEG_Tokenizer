"""Stamp dictionary statistics: templates, phase-invariant similarity, learned time-pool weights of
a finetune head, and per-stamp (a, b) sample accumulation for panel_stamp_distribution.py's violins.
Reuses tools/viz/extract.py's _used_flat_stamps StampBank call convention
(model.stage_features -> model.stamps) but keeps every raw per-(patch, channel, slot)
firing's (a, b) pair instead of averaging them into one trial-mean row -- the point here
is the spread, not a summary statistic."""
import numpy as np
import torch


@torch.no_grad()
def accumulate_stamp_ab(model, pretrain_dataset, trial_indices, device, max_stamps=30):
    """Runs the frozen tokenizer (no masking, same as tools/analysis/snapshot.py's
    build_pretrain_bundle) over pretrain_dataset[trial_indices] and buckets every
    selected (patch, channel, slot)'s real (a, b) pair -- StampBank's post-rms
    quadrature gain, see model/MeSAE/MeSAE_modules.py's StampBank class docstring -- by
    its GLOBAL stamp id (routed-then-shared layout, StampBank.forward's idx convention).

    Returns {stamp_id: (a [n], b [n])} np.ndarray pairs, for the max_stamps stamps with
    the most firings across trial_indices -- a dataset/subject can have hundreds of
    stamps, most barely used; capping keeps the violin plot readable the same way
    encode_used_stamps caps the gallery panels.
    """
    was_training = model.training
    model.eval()
    hit_count = torch.zeros(model.n_stamps, dtype=torch.long)
    a_by_stamp, b_by_stamp = {}, {}
    try:
        for trial_idx in trial_indices:
            x_patches, coords, _mask, time_indices, _y, valid_channels = pretrain_dataset[trial_idx]
            x = x_patches.unsqueeze(0).to(device)
            c = coords.unsqueeze(0).to(device)
            t = time_indices.unsqueeze(0).to(device)
            vc = valid_channels.unsqueeze(0).to(device)

            z, _ = model.stage_features(x, c, time_idx=t, valid_channels=vc)  # [1, C, N, D]
            B, C, N, D = z.shape
            z_g = z.permute(0, 2, 1, 3).reshape(B * N, C, D)
            x_g = x.permute(0, 2, 1, 3).reshape(B * N, C, -1)
            rms = x_g.pow(2).mean(dim=-1, keepdim=True).sqrt()
            vc_g = vc.unsqueeze(1).expand(B, N, C).reshape(B * N, C)

            out = model.stamps(z_g, x_target=None, rms=rms, valid_channels=vc_g)
            idx, amp = out.idx.cpu(), out.amp.cpu()  # idx [N, K], amp [N, C, K, 2]

            flat_idx = idx.unsqueeze(1).expand(-1, amp.shape[1], -1).reshape(-1)  # [N*C*K]
            a_flat = amp[..., 0].reshape(-1)
            b_flat = amp[..., 1].reshape(-1)
            hit_count.scatter_add_(0, flat_idx, torch.ones_like(flat_idx))

            for sid in flat_idx.unique().tolist():
                sel = flat_idx == sid
                a_by_stamp.setdefault(sid, []).append(a_flat[sel])
                b_by_stamp.setdefault(sid, []).append(b_flat[sel])
    finally:
        model.train(was_training)

    order = torch.argsort(hit_count, descending=True)
    used = [int(s) for s in order.tolist() if hit_count[s] > 0][:max_stamps]

    return {
        sid: (torch.cat(a_by_stamp[sid]).numpy(), torch.cat(b_by_stamp[sid]).numpy())
        for sid in used
    }


def stamp_templates(model):
    """-> D, H ([n_stamps, patch_len] numpy), ids (every stamp)."""
    D, H = (t.detach().cpu().numpy() for t in model.stamps._template_tables())
    return D, H, np.arange(D.shape[0])


def stamp_similarity(model):
    """Phase-invariant template similarity: a stamp presents a*D + b*H, so stamp j can show stamp i's
    waveform at any phase; sim(i, j) = sqrt(<D_i,D_j>^2 + <D_i,H_j>^2) in [0, 1] (1 = the same atom at
    some phase), symmetrised by max, diagonal NaN. -> (sim over the stamps, their labels)."""
    D, H, ids = stamp_templates(model)
    sim = np.sqrt((D[ids] @ D[ids].T) ** 2 + (D[ids] @ H[ids].T) ** 2)
    sim = np.maximum(sim, sim.T)
    np.fill_diagonal(sim, np.nan)
    return sim, [str(i) for i in ids]


def time_pool_weights(head_pth):
    """A finetune head's learned time-pool weights, w[s, n] = softmax_n(sum_r p[r, s] q[r, n]) per head
    entry with time_pool 'learned' -> {entry name: [S, N] numpy}."""
    sd = torch.load(head_pth, map_location='cpu', weights_only=False)['model_state_dict']
    return {k[:-len('.time.p')].removeprefix('entries.'):
            torch.softmax(torch.einsum('rs,rn->sn', sd[k], sd[k[:-1] + 'q']), -1).numpy()
            for k in sd if k.endswith('.time.p')}
