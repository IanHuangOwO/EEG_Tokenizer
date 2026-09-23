"""Per-stamp (a, b) sample accumulation for panel_stamp_distribution.py's violin plots.
Reuses tools/viz/extract.py's _used_flat_stamps StampBank call convention
(model.stage_features -> model.stamps) but keeps every raw per-(patch, channel, slot)
firing's (a, b) pair instead of averaging them into one trial-mean row -- the point here
is the spread, not a summary statistic."""
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
    used_stamp_ids/encode_used_stamps cap the gallery panels.
    """
    was_training = model.training
    model.eval()
    hit_count = torch.zeros(model.n_stamps, dtype=torch.long)
    a_by_stamp, b_by_stamp = {}, {}
    try:
        for trial_idx in trial_indices:
            x_patches, coords, _mask, time_indices, _y, _fft, valid_channels = pretrain_dataset[trial_idx]
            x = x_patches.unsqueeze(0).to(device)
            c = coords.unsqueeze(0).to(device)
            t = time_indices.unsqueeze(0).to(device)
            vc = valid_channels.unsqueeze(0).to(device)

            z, _ = model.stage_features(x, c, time_idx=t)  # [1, C, N, D]
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
