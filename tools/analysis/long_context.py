"""Long-input reconstruction: masked MSE on held-out inputs 1x, 2x and 4x the 5 s training window (19 / 39 / 79
patches), built by joining consecutive 5 s cache rows of one recording. Tests how a backbone's time position handles
sequences longer than any it was trained on (the case FoPE is designed for; 2026-10-07).

Rows are joined only where the signal continues across the boundary: compile cuts each continuous recording into
back-to-back 5 s windows (IO/loader.py _continuous_windows) but drops flat-line windows and starts a new run at every
recording, so a boundary counts as contiguous when its sample step is no larger than the recording's own
sample-to-sample steps (checked on the cached, pre-normalisation signal). Each 5 s piece keeps the loader's
per-window z-score, as in training. Every length uses the same held-out recordings and the same seeded masks
(backbone_eval's make_mask), so backbones compare directly; the 1x row is the reference.
"""
import copy
import glob
import json
import os

import numpy as np
import torch

from IO.dataset import build_dataset_from_config
from IO.preprocessing import slice_patches
from tools.analysis.backbone_eval import make_mask, val_config

# continuous recordings with dense caps and several paradigms
DATASETS = ('Weibo2014', 'Lee2019_MI', 'Schirrmeister2017', 'LEMON', 'Cho2017', 'NMT_Clinical')
KINDS = ('token_runs', 'time_block', 'random_channel')
LENGTHS = (1, 2, 4)                      # in 5 s cache rows
STEP_RATIO = 3.0                         # boundary step <= this x the recording's median sample step


def _contiguous_runs(raw, k):
    """raw [n, C, T] cached rows of one subject -> start rows of every k back-to-back contiguous rows."""
    steps = np.abs(np.diff(raw, axis=-1)).mean(1)                       # [n, T-1] mean over channels
    typical = np.median(steps)
    joint = np.abs(raw[1:, :, 0] - raw[:-1, :, -1]).mean(1)             # [n-1] boundary steps
    ok = joint <= STEP_RATIO * typical
    starts = []
    for s in range(len(raw) - k + 1):
        if ok[s:s + k - 1].all():
            starts.append(s)
    return starts[::k]                                                   # non-overlapping


def build_inputs(config, max_per_length=96):
    """{k: list of (x [C, T_k], coords [C, 3], valid [C])}, the k = 1 inputs being the first rows of the k = 4 runs."""
    cfg = val_config(config)
    cfg['dataset_params']['pretrain'] = {d: v for d, v in cfg['dataset_params']['pretrain'].items() if d in DATASETS}
    cfg = copy.deepcopy(cfg)
    cfg['preprocess_params']['window_fraction'] = 1.0
    for v in cfg['dataset_params']['pretrain'].values():
        v.pop('window_fraction', None)
    bd = build_dataset_from_config(cfg, mode='pretrain').base_dataset
    suffix = 'fs200_bp0.5-100.0_cont5.npz'
    out = {k: [] for k in LENGTHS}
    task_rows = {}
    for row, task in enumerate(bd.trial_to_coords_idx):
        task_rows.setdefault(task, []).append(row)
    for task, rows in task_rows.items():
        ds, sub = bd.dataset_names[rows[0]], int(bd.subject_data[rows[0]])
        f = glob.glob(f"{cfg['dataset_params']['pretrain'][ds]['dataset_path']}/cache/{sub}_{suffix}")
        if not f:
            continue
        raw = np.load(f[0])['data']
        if len(raw) != len(rows):                                        # rows not 1:1 with cache rows: skip
            continue
        starts = _contiguous_runs(raw, max(LENGTHS))
        coords, valid = bd.all_coords[task], bd.all_valid_channels[task]
        for s in starts:
            for k in LENGTHS:
                x = torch.cat([bd.data[rows[s + j]] for j in range(k)], dim=-1)
                out[k].append((x, coords, valid))
    rng = np.random.default_rng(0)
    n = len(out[max(LENGTHS)])
    keep = np.sort(rng.permutation(n)[:max_per_length])
    return {k: [v[i] for i in keep] for k, v in out.items()}, n


def evaluate(model, config, out_path, max_per_length=96, name=''):
    model = model.cpu().eval()
    pp = config['preprocess_params']
    L, S = pp['patch_length'], pp.get('patch_stride', pp['patch_length'])
    inputs, n_runs = build_inputs(config, max_per_length)
    res = {}
    for k, items in inputs.items():
        acc = {kind: [0.0, 0] for kind in KINDS}
        for w, (x, coords, valid) in enumerate(items):
            xp, t = slice_patches(x[None], L, S)                         # [1, C, N, L], [N]
            N = xp.shape[2]
            valid_tok = valid[:, None].expand(-1, N).clone()
            for kind in KINDS:
                m, sc = make_mask(kind, valid_tok, coords, 10_000 * k + w, {})
                with torch.no_grad():
                    r = model(xp, coords[None], t[None], bool_masked_pos=m[None], valid_channels=valid[None]).recon
                e = (r.float() - xp).pow(2).mean(-1)[0]
                acc[kind][0] += float(e[sc].sum()); acc[kind][1] += int(sc.sum())
        res[f'{5 * k}s'] = {'patches': N, 'inputs': len(items),
                            'masked_mse': {kind: v[0] / max(v[1], 1) for kind, v in acc.items()}}
    ref = res['5s']['masked_mse']
    for key in res:
        res[key]['vs_5s'] = {kind: res[key]['masked_mse'][kind] / ref[kind] for kind in KINDS}
    res['contiguous_20s_runs_found'] = n_runs
    json.dump(res, open(out_path, 'w'), indent=2)
    print(f'\n=== {name} long-context masked MSE ({n_runs} contiguous 20 s runs found, {len(inputs[1])} used)')
    for key in ('5s', '10s', '20s'):
        print(f"  {key:>4s} ({res[key]['patches']:2d} patches): " +
              '  '.join(f"{kind} {res[key]['masked_mse'][kind]:.4f} (x{res[key]['vs_5s'][kind]:.2f})" for kind in KINDS))
    return res
