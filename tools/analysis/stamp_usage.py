"""
Stamp usage on held-out pretrain windows (unmasked, CPU): the direct measures of whether the stamp
dictionary's slots are distinct and all used -- what mp_loss (and any dedup change) is for.

Per stamp:
  first       share of patches where it is the strongest stamp (per-patch ranking by strength h)
  rank_ent    entropy of its rank across patches / log(#stamps): high = its rank moves with content
  energy      its share of the total reconstructed energy
  remove_cost rise in reconstruction MSE when only this stamp is dropped, / the model's MSE
              (~0 = redundant: another stamp covers it, e.g. a near-duplicate)
Summary: stamps ranked first in >= 5% of patches, redundant stamps (remove_cost < 1%).
Routing (routed stamps only): usage entropy / log(#routed), dead share (used in < 0.1% of patch
selections), and dataset dependence = mean Jensen-Shannon divergence between each dataset's routed
usage and the pooled usage (0 = the same stamps everywhere; higher = content-dependent selection).

Panel: `python analysis_pretrain.py --run <backbone> --panel stamp_usage` writes
output/<backbone>/pretrain/analysis/stamp_usage.json.
"""
import json

import numpy as np
import torch

from tools.analysis.backbone_eval import batches, eval_windows


@torch.no_grad()
def stamp_usage(model, config, out_path, max_windows=256):
    model = model.cpu().eval()
    st = model.stamps
    S = st.n_stamps
    ds, idx = eval_windows(config, max_windows)
    first, ranks = np.zeros(S), np.zeros((S, S))
    energy, cost, base = np.zeros(S), np.zeros(S), 0.0
    n_routed, top_k = st.n_routed, st.top_k
    routed_use = {}                                                          # dataset -> [n_routed] counts
    names = ds.base_dataset.dataset_names
    for wids, x, coords, t, valid in batches(ds, idx):
        B, C, N, L = x.shape
        out = model(x, coords, t, valid_channels=valid)
        G = B * N
        x_g = x.permute(0, 2, 1, 3).reshape(G, C, L)                         # G = b*N + n, as in forward
        v = valid[:, None, :].expand(B, N, C).reshape(G, C, 1).float() \
            * (x_g.abs().amax(-1, keepdim=True) > 0).float()                  # real channel AND real patch
        contrib = st.decode_selected(out.idx, out.amp).float()               # [G, C, K, L]
        recon = contrib.sum(2)
        err = (x_g - recon).pow(2)
        base += float((err * v).sum())
        real_g = v.amax(1).flatten() > 0                                     # [G] patches with real content
        h = out.h[real_g]                                                     # [G', K] strength per slot
        ids = out.idx[real_g]                                                 # [G', K] stamp id per slot
        order = h.argsort(-1, descending=True)
        ranked_ids = ids.gather(1, order)                                     # [G', K], rank 0 = strongest
        np.add.at(first, ranked_ids[:, 0].numpy(), 1)
        if top_k > 0:                                                         # routed selections per patch
            for g in torch.nonzero(real_g).flatten().tolist():
                name = names[wids[g // N]]
                cnt = routed_use.setdefault(name, np.zeros(n_routed))
                np.add.at(cnt, out.idx[g, :top_k].numpy(), 1)
        for r in range(ranked_ids.shape[1]):
            np.add.at(ranks[:, r], ranked_ids[:, r].numpy(), 1)
        for k in range(contrib.shape[2]):
            sid = out.idx[:, k]                                               # [G]
            e_k = (contrib[:, :, k].pow(2) * v).sum((1, 2))
            c_k = (((x_g - recon + contrib[:, :, k]).pow(2) - err) * v).sum((1, 2))
            np.add.at(energy, sid.numpy(), e_k.numpy())
            np.add.at(cost, sid.numpy(), c_k.numpy())
    n_patch = first.sum()
    p = ranks / np.maximum(ranks.sum(1, keepdims=True), 1)
    rank_ent = -(p * np.log(p + 1e-12)).sum(1) / np.log(ranks.shape[1])
    res = {'windows': len(idx), 'patches': int(n_patch),
           'per_stamp': [{'stamp': s, 'first': first[s] / n_patch, 'rank_entropy': float(rank_ent[s]),
                          'energy_share': energy[s] / energy.sum(), 'remove_cost': cost[s] / base}
                         for s in range(S)]}
    res['stamps_first_ge_5pct'] = int(sum(d['first'] >= 0.05 for d in res['per_stamp']))
    res['redundant_stamps'] = [d['stamp'] for d in res['per_stamp'] if d['remove_cost'] < 0.01]
    if routed_use:
        tot = sum(routed_use.values())
        pz = tot / tot.sum()
        ent = float(-(pz * np.log(pz + 1e-12)).sum() / np.log(n_routed))
        def js(p, q):
            mm = (p + q) / 2
            kl = lambda a, b: float((a * np.log((a + 1e-12) / (b + 1e-12))).sum())
            return (kl(p, mm) + kl(q, mm)) / 2
        res['routing'] = {'usage_entropy': ent, 'dead_share': float((pz < 1e-3).mean()),
                          'dataset_js': float(np.mean([js(c / c.sum(), pz) for c in routed_use.values()]))}
    json.dump(res, open(out_path, 'w'), indent=2)

    print(f'  {"stamp":>5} {"first%":>7} {"rank_ent":>8} {"energy%":>8} {"remove_cost":>11}')
    for d in sorted(res['per_stamp'], key=lambda d: -d['first']):
        print(f'  {d["stamp"]:5d} {d["first"]*100:7.1f} {d["rank_entropy"]:8.2f} {d["energy_share"]*100:8.1f} '
              f'{d["remove_cost"]*100:10.1f}%')
    print(f'  stamps ranked first in >= 5% of patches: {res["stamps_first_ge_5pct"]}/{S} | '
          f'redundant (remove_cost < 1%): {res["redundant_stamps"]}')
    if 'routing' in res:
        print(f'  routed usage entropy {res["routing"]["usage_entropy"]:.2f}, dead share {res["routing"]["dead_share"]:.2f},'
              f' dataset JS {res["routing"]["dataset_js"]:.3f}')
    return res
