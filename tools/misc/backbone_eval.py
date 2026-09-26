"""
Backbone evaluation on held-out pretrain windows, CPU only. Every backbone sees the SAME
windows and the SAME masks (seeded per window), so runs trained with different masking
schemes are compared on one task.

1. Test masks -- masked-patch MSE (plain patch MSE on masked real tokens, as _recon_loss's
   l_masked) for each mask type, next to two model-free baselines:
     idw     inverse-distance interpolation from the 4 nearest visible channels (channel masks)
     linear  linear interpolation over time across the hidden samples (time masks)
     zero    predicting 0 (= signal power of the normalised data)
   Types: token_runs (random 3-patch runs, 50%: the baseline's own task), random_channel
   (30% of channels), channel_cluster (30%), time_block (30% of time), and motor3_to_bci22
   (dense caps only: every channel except C3/Cz/C4 hidden, scored on the other 19 BCI-22
   channels -- the sparse-cap imputation BNCI2014004 needs). Also split sparse (<= 22 real
   channels) vs dense windows.
2. Embedding ablations under every test mask: coords shuffled across channels, all
   channels at the mean position, time_idx shuffled, time_idx constant (masked MSE per mask
   type; ablation_masked_mse keeps the token_runs row for older readers).
3. Structure: coordinate-embedding similarity vs electrode closeness (Spearman, 10-10
   channels); pos_emb drift from its sinusoidal init; with a RelativeSpatialBias, per block
   the Spearman correlation of the head-averaged bias with closeness (> 0: prefers
   neighbours) and its mean |value|.

Writes output/<run>/pretrain/analysis/backbone_eval.json and prints a summary.
Usage: python -m tools.misc.backbone_eval --run mesae_tiny_static16_base_s1 [...] [--max-windows 512]
"""
import argparse
import copy
import zlib
import json
import os
import random

import numpy as np
import torch
from scipy.stats import spearmanr

from IO.dataset import build_dataset_from_config, load_montage_channels
from IO.loader import get_standard_coords
from IO.masking import ChannelClusterMask, RandomChannelMask, TimeBlockMask, random_token_mask
from model.MeSAE.MeSAE_modules import fourier_features, get_sinusoidal_pos, overlap_add_patches
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


def eval_windows(config, max_windows):
    """Held-out windows, fixed order. Masks/subsampling of the dataset itself are unused."""
    ds = build_dataset_from_config(val_config(config), mode='pretrain')
    idx = torch.randperm(len(ds.base_dataset), generator=torch.Generator().manual_seed(0))[:max_windows].tolist()
    return ds, idx


def batches(ds, idx, bs=32):
    for i in range(0, len(idx), bs):
        items = [ds[j] for j in idx[i:i + bs]]
        x, coords, t, valid = (torch.stack([torch.as_tensor(it[k]) for it in items]) for k in (0, 1, 3, 5))
        yield idx[i:i + bs], x.float(), coords.float(), t.long(), valid.bool()


# ---------- test masks ----------

def make_mask(kind, valid_tok, coords, window_id, names_idx):
    """valid_tok [C, N] bool -> (mask [C, N], score [C, N]); score = tokens that count."""
    torch.manual_seed(1_000_003 * window_id + zlib.crc32(kind.encode()) % 997)   # stable across processes
    C, N = valid_tok.shape
    if kind == 'token_runs':
        m = random_token_mask(C, N, 0.5, valid_tok.flatten(), time_run=3).view(C, N)
    elif kind == 'random_channel':
        m = RandomChannelMask().generate(valid_tok, coords, 0.3)
    elif kind == 'channel_cluster':
        m = ChannelClusterMask().generate(valid_tok, coords, 0.3)
    elif kind == 'time_block':
        m = TimeBlockMask().generate(valid_tok, coords, 0.3)
    elif kind == 'motor3_to_bci22':
        real = valid_tok.any(1)
        motor = torch.zeros(C, dtype=torch.bool); motor[names_idx['motor-3']] = True
        bci = torch.zeros(C, dtype=torch.bool); bci[names_idx['bci-22']] = True
        if int(real.sum()) < 32 or not (motor & real).sum() == 3 or int((bci & real & ~motor).sum()) < 10:
            return None, None
        m = valid_tok & ~motor[:, None]
        return m, m & (bci & ~motor)[:, None]
    return m, m


def idw_predict(x, coords, visible_ch, k=4):
    """x [C, N, L]; predict every channel from the k nearest visible channels (1/d^2)."""
    vis = visible_ch.nonzero().flatten()
    if len(vis) == 0:
        return torch.zeros_like(x)
    d = torch.cdist(coords, coords[vis]).clamp_min(1e-3)                 # [C, V]
    k = min(k, len(vis))
    dk, ik = d.topk(k, dim=1, largest=False)
    w = 1.0 / dk.pow(2); w = w / w.sum(1, keepdim=True)                   # [C, k]
    return torch.einsum('ck,cknl->cnl', w, x[vis][ik])


def linear_time_predict(x, m, stride):
    """Overlap-add the visible patches to a signal, linearly interpolate the samples no visible
    patch covers, slice back into patches. x, m: [C, N, L] / [C, N]."""
    C, N, L = x.shape
    sig = overlap_add_patches(x[None], stride)[0]                          # [C, T]
    cover = overlap_add_patches((~m)[None, :, :, None].float().expand(1, C, N, L), stride)[0] > 1e-6
    out = sig.clone()
    t = np.arange(sig.shape[1])
    for c in range(C):
        hid = ~cover[c]
        if hid.any() and (~hid).any():
            out[c, hid] = torch.from_numpy(np.interp(t[hid.numpy()], t[~hid.numpy()], sig[c, ~hid].numpy())).float()
    idx = torch.arange(N)[:, None] * stride + torch.arange(L)[None, :]
    return out[:, idx]                                                    # [C, N, L]


KINDS = ['token_runs', 'random_channel', 'channel_cluster', 'time_block', 'motor3_to_bci22']


# ---------- ablations ----------

def ablate(name, coords, t, valid):
    coords, t = coords.clone(), t.clone()
    for b in range(coords.shape[0]):
        v = valid[b].nonzero().flatten()
        if name == 'coords_shuffle':
            coords[b, v] = coords[b, v[torch.randperm(len(v))]]
        elif name == 'coords_mean':
            coords[b, v] = coords[b, v].mean(0)
    if name == 'time_shuffle':
        t = torch.stack([row[torch.randperm(len(row))] for row in t])
    elif name == 'time_const':
        t.zero_()
    return coords, t


ABLATIONS = ['baseline', 'coords_shuffle', 'coords_mean', 'time_shuffle', 'time_const']


@torch.no_grad()
def run(name, max_windows):
    config = json.load(open(f'output/{name}/pretrain/artifacts/config.json'))
    model = load_model(config, f'output/{name}/pretrain/checkpoint/last.pth', torch.device('cpu'), mode='pretrain').eval()
    stride = config['preprocess_params'].get('patch_stride', config['preprocess_params']['patch_length'])
    ds, idx = eval_windows(config, max_windows)
    bd = ds.base_dataset
    norm = bd._normalize_label
    n2i = {norm(n): i for i, n in enumerate(bd.channel_names)}
    names_idx = {m: [n2i[norm(n)] for n in load_montage_channels(m) if norm(n) in n2i] for m in ('motor-3', 'bci-22')}

    acc = {}                                              # (kind, group, predictor) -> [sum, count]
    def add(key, err, sel):
        a = acc.setdefault(key, [0.0, 0])
        a[0] += float(err[sel].sum()); a[1] += int(sel.sum())

    abl = {k: {a: [0.0, 0] for a in ABLATIONS} for k in KINDS}
    for wids, x, coords, t, valid in batches(ds, idx):
        B, C, N, L = x.shape
        valid_tok = torch.stack([ds._valid_masks[w].view(C, N) for w in wids])
        for kind in KINDS:
            masks, scores = [], []
            for b in range(B):
                m, s = make_mask(kind, valid_tok[b], coords[b], wids[b], names_idx)
                masks.append(torch.zeros(C, N, dtype=torch.bool) if m is None else m)
                scores.append(torch.zeros(C, N, dtype=torch.bool) if s is None else s)
            mp, sc = torch.stack(masks), torch.stack(scores)
            if not sc.any():
                continue
            out = model(x, coords, t, bool_masked_pos=mp, valid_channels=valid)
            err_model = (out.recon.float() - x).pow(2).mean(-1)          # [B, C, N]
            for b in range(B):
                if not sc[b].any():
                    continue
                group = 'sparse' if int(valid[b].sum()) <= 22 else 'dense'
                preds = {'model': err_model[b], 'zero': x[b].pow(2).mean(-1)}
                if kind in ('random_channel', 'channel_cluster', 'motor3_to_bci22'):
                    vis_ch = valid[b] & ~mp[b].any(1)
                    preds['idw'] = (idw_predict(x[b], coords[b], vis_ch) - x[b]).pow(2).mean(-1)
                if kind in ('time_block', 'token_runs'):
                    preds['linear'] = (linear_time_predict(x[b], mp[b], stride) - x[b]).pow(2).mean(-1)
                for p, e in preds.items():
                    add((kind, 'all', p), e, sc[b]); add((kind, group, p), e, sc[b])
            for a in ABLATIONS:
                cc, tt = ablate(a, coords, t, valid)
                torch.manual_seed(0)
                o = model(x, cc, tt, bool_masked_pos=mp, valid_channels=valid)
                e = (o.recon.float() - x).pow(2).mean(-1)
                abl[kind][a][0] += float(e[sc].sum()); abl[kind][a][1] += int(sc.sum())

    # structure
    emb = model.embed
    labels = [n for n in load_montage_channels('10-10') if get_standard_coords(n) is not None]
    pos = torch.tensor(np.stack([get_standard_coords(n) for n in labels]), dtype=torch.float32)
    iu = torch.triu_indices(len(labels), len(labels), 1)
    closeness = -torch.cdist(pos, pos)
    struct = {'spatial_embedding': emb.coord_proj is not None}
    if emb.coord_proj is not None:
        ce = emb.coord_proj(fourier_features(pos))
        ce = ce / ce.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        struct['coord_sim_vs_closeness_spearman'] = float(spearmanr((ce @ ce.T)[iu[0], iu[1]], closeness[iu[0], iu[1]]).correlation)
    Np = 39
    pe = emb.pos_emb[0, :Np]
    init = get_sinusoidal_pos(emb.pos_emb.shape[1], emb.pos_emb.shape[2], torch.device('cpu'))[0, :Np]
    struct['pos_emb_drift'] = float((pe - init).norm() / init.norm())
    if getattr(model, 'spatial_bias', None) is not None:
        bias = model.spatial_bias(pos[None])[0].mean(1)                    # [depth, C, C], heads averaged
        off = ~torch.eye(len(labels), dtype=torch.bool)
        struct['spatial_bias_per_block'] = [
            {'closeness_spearman': float(spearmanr(bias[i][off], closeness[off]).correlation),
             'mean_abs': float(bias[i][off].abs().mean())} for i in range(bias.shape[0])]

    res = {'windows': len(idx),
           'test_masks': {f'{k}|{g}|{p}': v[0] / max(v[1], 1) for (k, g, p), v in acc.items()},
           'ablation_masked_mse': {a: v[0] / max(v[1], 1) for a, v in abl['token_runs'].items()},
           'ablation_by_mask': {k: {a: v[0] / max(v[1], 1) for a, v in d.items()} for k, d in abl.items()},
           'structure': struct}
    os.makedirs(f'output/{name}/pretrain/analysis', exist_ok=True)
    json.dump(res, open(f'output/{name}/pretrain/analysis/backbone_eval.json', 'w'), indent=2)

    print(f'\n=== {name} ({len(idx)} held-out windows)')
    print(f'  {"test mask":17} {"group":6} {"model":>7} {"idw":>7} {"linear":>7} {"zero":>7}')
    for k in KINDS:
        for g in ('all', 'sparse', 'dense'):
            row = {p: res['test_masks'].get(f'{k}|{g}|{p}') for p in ('model', 'idw', 'linear', 'zero')}
            if row['model'] is None:
                continue
            print(f'  {k:17} {g:6} ' + ' '.join(f'{v:7.4f}' if v is not None else f'{"-":>7}' for v in row.values()))
    print(f'  ablation, masked MSE change: {"mask":17} ' + ' '.join(f'{a:>15}' for a in ABLATIONS[1:]))
    for k, d in res['ablation_by_mask'].items():
        print(f'  {"":28} {k:17} ' + ' '.join(f'{d[a] / d["baseline"] - 1:>+15.0%}' for a in ABLATIONS[1:]))
    print(f'  coord sim vs closeness {struct.get("coord_sim_vs_closeness_spearman", float("nan")):.3f} | pos_emb drift {struct["pos_emb_drift"]:.1%}')
    if 'spatial_bias_per_block' in struct:
        print('  spatial bias per block (closeness rho / mean|b|): ' +
              ' '.join(f'{d["closeness_spearman"]:+.2f}/{d["mean_abs"]:.2f}' for d in struct['spatial_bias_per_block']))


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
