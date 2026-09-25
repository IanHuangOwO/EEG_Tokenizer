"""
One-off: do a pretrained MeSAE backbone's spatial (coord) and temporal (pos_emb) embeddings
matter? CPU is enough (small model, a few hundred held-out windows).

1. Size: per-token L2 norm of the content projection, the temporal embedding and the coord
   embedding, before the embedding LayerNorm.
2. Structure: learned coord_scale; Spearman correlation between channel-pair cosine
   similarity of the coord embedding and electrode distance (all canonical channels);
   pos_emb drift from its sinusoidal init and adjacent-patch cosine similarity.
3. Function: masked / visible reconstruction MSE on the run's own validation subjects (same
   seed-42 subject split as train_pretrain.py) under ablations of coords and time_idx, with
   one fixed 50% token mask shared by every condition.

Usage: python -m tools.misc.embedding_ablation --run mesae_tiny_static16_s1 [...more runs]
       [--max-windows 512]
"""
import argparse
import copy
import json
import os
import random

import numpy as np
import torch
from scipy.stats import spearmanr

from IO.dataset import build_dataset_from_config
from IO.loader import get_standard_coords
from model.MeSAE.MeSAE_modules import get_sinusoidal_pos
from tools.analysis import load_model


def val_config(config):
    """Replicates train_pretrain.py's per-dataset seed-42 subject split, returns the val half."""
    random.seed(42)
    cfg = copy.deepcopy(config)
    for name, args in config['dataset_params']['pretrain'].items():
        meta = json.load(open(os.path.join(args['dataset_path'], 'metadata.json')))
        subs = sorted(meta['data_structure'].keys())
        req = args['subject_to_use']
        subs = subs if req in (['all'], 'all') else [s for s in subs if s in {str(r) for r in req}]
        random.shuffle(subs)
        n_train = int(len(subs) * config['training_params']['pretrain'].get('train_val_split', 0.9))
        if n_train == len(subs) and len(subs) > 1:
            n_train -= 1
        n_train = max(n_train, 1) if subs else 0
        cfg['dataset_params']['pretrain'][name]['subject_to_use'] = subs[n_train:]
    cfg['dataset_params']['pretrain'] = {k: v for k, v in cfg['dataset_params']['pretrain'].items()
                                         if v['subject_to_use']}
    return cfg


def batches(ds, n, bs=32):
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:n].tolist()
    for i in range(0, len(idx), bs):
        items = [ds[j] for j in idx[i:i + bs]]
        x, coords, t, valid = (torch.stack([torch.as_tensor(it[k]) for it in items]) for k in (0, 1, 3, 6))
        yield x.float(), coords.float(), t.long(), valid.bool()


def fixed_mask(valid, N, seed):
    g = torch.Generator().manual_seed(seed)
    tok = valid[:, :, None].expand(-1, -1, N)
    return (torch.rand(tok.shape, generator=g) < 0.5) & tok


def ablate(name, coords, t, valid):
    B, C, _ = coords.shape
    coords, t = coords.clone(), t.clone()
    if name == 'coords_zero':
        coords.zero_()
    elif name == 'coords_shuffle':
        for b in range(B):
            v = valid[b].nonzero().flatten()
            coords[b, v] = coords[b, v[torch.randperm(len(v))]]
    elif name == 'coords_mean':
        for b in range(B):
            v = valid[b].nonzero().flatten()
            coords[b, v] = coords[b, v].mean(0)
    elif name == 'time_shuffle':
        t = torch.stack([row[torch.randperm(len(row))] for row in t])
    elif name == 'time_const':
        t.zero_()
    return coords, t


CONDITIONS = ['baseline', 'coords_zero', 'coords_shuffle', 'coords_mean', 'time_shuffle', 'time_const']


@torch.no_grad()
def run(name, max_windows):
    ckpt = f'output/{name}/pretrain/checkpoint/last.pth'
    config = json.load(open(f'output/{name}/pretrain/artifacts/config.json'))
    model = load_model(config, ckpt, torch.device('cpu'), mode='pretrain').eval()
    emb = model.embed
    ds = build_dataset_from_config(val_config(config), mode='pretrain')

    # 1 + 3: sizes and ablations over the same batches
    sums = {c: [0.0, 0.0, 0] for c in CONDITIONS}
    norms = {'content': [], 'temporal': [], 'coord': []}
    torch.manual_seed(0)
    for bi, (x, coords, t, valid) in enumerate(batches(ds, max_windows)):
        B, C, N, L = x.shape
        mp = fixed_mask(valid, N, bi)
        content = emb.proj(x)                                         # [B, C, N, D]
        tv = valid[:, :, None].expand(-1, -1, N)
        norms['content'].append(content[tv].norm(dim=-1))
        norms['temporal'].append(emb.pos_emb[0][t].norm(dim=-1).flatten())
        norms['coord'].append(emb.coord_proj(coords[valid] * emb.coord_scale).norm(dim=-1))
        for c in CONDITIONS:
            cc, tt = ablate(c, coords, t, valid)
            torch.manual_seed(bi)                                     # identical routing noise, if any
            out = model(x, cc, tt, bool_masked_pos=mp, valid_channels=valid)
            _, lm, lu = model.get_loss(x, out.recon, out.aux_loss, bool_masked_pos=mp,
                                       valid_channels=out.valid_channels)
            sums[c][0] += float(lm); sums[c][1] += float(lu); sums[c][2] += 1

    # 2: structure
    names = [ch['label'] for ch in json.load(open('configs/montages.json'))['10-10']['channels']
             if get_standard_coords(ch['label']) is not None]
    pos = torch.tensor(np.stack([get_standard_coords(n) for n in names]), dtype=torch.float32)
    ce = emb.coord_proj(pos * emb.coord_scale)
    ce = ce / ce.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    iu = torch.triu_indices(len(names), len(names), 1)
    rho = spearmanr((ce @ ce.T)[iu[0], iu[1]].numpy(), -torch.cdist(pos, pos)[iu[0], iu[1]].numpy()).correlation
    N = 39
    pe = emb.pos_emb[0, :N]
    init = get_sinusoidal_pos(emb.pos_emb.shape[1], emb.pos_emb.shape[2], torch.device('cpu'))[0, :N]
    drift = ((pe - init).norm() / init.norm()).item()
    pen = pe / pe.norm(dim=-1, keepdim=True)
    adj = (pen[1:] * pen[:-1]).sum(-1).mean().item()

    base_m, base_u = sums['baseline'][0] / sums['baseline'][2], sums['baseline'][1] / sums['baseline'][2]
    print(f'\n=== {name}  ({sum(s[2] for s in sums.values()) // len(CONDITIONS)} batches)')
    print('  norms (median): ' + ', '.join(f'{k} {torch.cat(v).median():.2f}' for k, v in norms.items()))
    print(f'  coord_scale {emb.coord_scale.item():.2f} | coord-emb similarity vs closeness Spearman {rho:.3f} '
          f'({len(names)} channels) | pos_emb drift from sinusoid {drift:.1%}, adjacent-patch cos {adj:.3f}')
    print(f'  {"condition":16} {"masked MSE":>11} {"change":>8} {"visible MSE":>12} {"change":>8}')
    for c in CONDITIONS:
        m, u = sums[c][0] / sums[c][2], sums[c][1] / sums[c][2]
        print(f'  {c:16} {m:11.4f} {m / base_m - 1:+8.1%} {u:12.4f} {u / base_u - 1:+8.1%}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', nargs='+', required=True)
    ap.add_argument('--max-windows', type=int, default=512, dest='max_windows')
    args = ap.parse_args()
    torch.set_num_threads(8)
    for name in args.run:
        run(name, args.max_windows)


if __name__ == '__main__':
    main()
