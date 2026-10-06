"""Q-atom dictionary statistics: templates, phase-invariant similarity, learned time-pool weights of
a finetune head, and per-atom (a, b) samples for panel_stamp_distribution.py's violins (every raw
per-(patch, channel) pair kept -- the point there is the spread, not a summary statistic)."""
import numpy as np
import torch


@torch.no_grad()
def accumulate_stamp_ab(model, pretrain_dataset, trial_indices, device, max_stamps=30):
    """Runs the frozen tokenizer (no masking) over pretrain_dataset[trial_indices] and collects every
    (patch, channel)'s post-rms (a, b) pair per Q-atom (StampBank's quadrature gain). Returns
    {stamp_id: (a [n], b [n])} np.ndarray pairs for the max_stamps Q-atoms with the largest mean
    amplitude, in that order."""
    was_training = model.training
    model.eval()
    amps = []
    try:
        for trial_idx in trial_indices:
            x_patches, coords, _mask, time_indices, _y, valid_channels = pretrain_dataset[trial_idx]
            out = model.encode_stamps(x_patches.unsqueeze(0).to(device), coords.unsqueeze(0).to(device),
                                      time_idx=time_indices.unsqueeze(0).to(device),
                                      valid_channels=valid_channels.unsqueeze(0).to(device))
            amps.append(out.amp[:, valid_channels.to(device).bool()].reshape(-1, model.n_stamps, 2).cpu())
    finally:
        model.train(was_training)
    ab = torch.cat(amps)                                                   # [n, S, 2]
    order = ab.norm(dim=-1).mean(0).argsort(descending=True)[:max_stamps].tolist()
    return {s: (ab[:, s, 0].numpy(), ab[:, s, 1].numpy()) for s in order}


def stamp_templates(model):
    """-> D, H ([n_stamps, patch_len] numpy), ids (every Q-atom)."""
    D, H = (t.detach().cpu().numpy() for t in model.stamps.templates())
    return D, H, np.arange(D.shape[0])


def stamp_similarity(model):
    """Phase-invariant template similarity: a Q-atom presents a*D + b*H, so Q-atom j can show Q-atom i's
    waveform at any phase; sim(i, j) = sqrt(<D_i,D_j>^2 + <D_i,H_j>^2) in [0, 1] (1 = the same atom at
    some phase), symmetrised by max, diagonal NaN. -> (sim over the Q-atoms, their labels)."""
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
