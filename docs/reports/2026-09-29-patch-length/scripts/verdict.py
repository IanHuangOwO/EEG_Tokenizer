"""Patch-length card verdict: z-probe tail balanced accuracy per cell, mean over finetune seeds per backbone,
mean +- SE over the 3 pretrain seeds per patch length; win = |diff| > 2 x sqrt(SE50^2 + SE100^2)."""
import json, glob, os
import numpy as np
BB = {50: ['mesae_tiny_notrial_s1', 'mesae_tiny_p50_s16_s2', 'mesae_tiny_p50_s16_s3'],
      100: ['mesae_tiny_p100_s16_s1', 'mesae_tiny_p100_s16_s2', 'mesae_tiny_p100_s16_s3']}
CELLS = [f'{d}_{s}' for d in ('BNCI2014004', 'BNCI2014001', 'BNCI2014008') for s in ('loso', 'fewshot')]
DS = ['BNCI2014004', 'BNCI2014001', 'BNCI2014008']

def cell_acc(bb, cell):
    seeds = []
    for s in (1, 2, 3):
        d = json.load(open(f'output/{bb}/finetune/patch_probe/{cell}_seed{s}/artifacts/group_eval.json'))
        subj = {k: v['tail'] for f in d.values() for g in f['groups'].values() for k, v in g['subjects'].items()}
        seeds.append(np.mean(list(subj.values())) * 100)
    return np.mean(seeds), seeds

rows, wins, losses = [], 0, 0
print('| Cell | patch 50 (per seed) | patch 100 (per seed) | diff | threshold | result |\n|---|---|---|---|---|---|')
for c in CELLS:
    v = {p: np.array([cell_acc(b, c)[0] for b in BB[p]]) for p in BB}
    m = {p: v[p].mean() for p in v}; se = {p: v[p].std(ddof=1) / np.sqrt(3) for p in v}
    diff, thr = m[100] - m[50], 2 * np.sqrt(se[50] ** 2 + se[100] ** 2)
    res = 'p100 wins' if diff > thr else ('p100 loses' if diff < -thr else 'tie')
    wins += res == 'p100 wins'; losses += res == 'p100 loses'
    fmt = lambda p: f"{m[p]:.1f} ({' / '.join(f'{x:.1f}' for x in v[p])})"
    print(f'| {c} | {fmt(50)} | {fmt(100)} | {diff:+.1f} | {thr:.1f} | {res} |')
print(f'\np100 wins {wins}, loses {losses} of 6 cells')

print('\n| ridge probe on z | patch 50 | patch 100 | diff |\n|---|---|---|---|')
for ds in DS:
    v = {}
    for p in BB:
        xs = []
        for b in BB[p]:
            f = f'output/{b}/pretrain/analysis/ridge_probe.json'
            if os.path.exists(f): xs.append(json.load(open(f))[ds]['mean'] * 100)
        v[p] = np.array(xs)
    s = lambda p: f"{v[p].mean():.1f} +- {v[p].std(ddof=1) / np.sqrt(len(v[p])):.1f} (n={len(v[p])})"
    print(f'| {ds} | {s(50)} | {s(100)} | {v[100].mean() - v[50].mean():+.1f} |')

print('\n| masked MSE (not judged) | patch 50 | patch 100 |\n|---|---|---|')
for m_ in ['token_runs', 'random_channel', 'channel_cluster', 'time_block', 'motor3_to_bci22']:
    out = []
    for p in BB:
        xs = [json.load(open(f'output/{b}/pretrain/analysis/backbone_eval.json'))['test_masks'][f'{m_}|all|model'] for b in BB[p]]
        out.append(f'{np.mean(xs):.3f} +- {np.std(xs, ddof=1) / np.sqrt(3):.3f}')
    print(f'| {m_} | {out[0]} | {out[1]} |')
