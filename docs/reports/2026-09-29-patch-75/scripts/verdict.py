"""Patch-75 card verdict: z-probe tail balanced accuracy per cell (mean over finetune seeds per backbone,
mean +- SE over 3 pretrain seeds per patch length); won when |diff| > 2 x sqrt(SE_a^2 + SE_b^2)."""
import json, os
import numpy as np
BB = {50: ['mesae_tiny_notrial_s1', 'mesae_tiny_p50_s16_s2', 'mesae_tiny_p50_s16_s3'],
      75: ['mesae_tiny_p75_s16_s1', 'mesae_tiny_p75_s16_s2', 'mesae_tiny_p75_s16_s3'],
      100: ['mesae_tiny_p100_s16_s1', 'mesae_tiny_p100_s16_s2', 'mesae_tiny_p100_s16_s3']}
CELLS = [f'{d}_{s}' for s in ('loso', 'fewshot') for d in ('BNCI2014004', 'BNCI2014001', 'BNCI2014008')]
def acc(bb, cell):
    out = []
    for s in (1, 2, 3):
        d = json.load(open(f'output/{bb}/finetune/patch_probe/{cell}_seed{s}/artifacts/group_eval.json'))
        out.append(np.mean([v['tail'] for f in d.values() for g in f['groups'].values() for v in g['subjects'].values()]) * 100)
    return np.mean(out)
V = {c: {p: np.array([acc(b, c) for b in BB[p]]) for p in BB} for c in CELLS}
def cmp(a, b):
    print(f'\n| Cell | patch {a} | patch {b} | diff ({b} - {a}) | threshold | result |\n|---|---|---|---|---|---|')
    for c in CELLS:
        x, y = V[c][a], V[c][b]; se = lambda v: v.std(ddof=1) / np.sqrt(3)
        d, t = y.mean() - x.mean(), 2 * np.sqrt(se(x) ** 2 + se(y) ** 2)
        r = f'p{b} wins' if d > t else (f'p{b} loses' if d < -t else 'tie')
        f = lambda v: f"{v.mean():.1f} ({' / '.join(f'{u:.1f}' for u in v)})"
        print(f'| {c} | {f(x)} | {f(y)} | {d:+.1f} | {t:.1f} | {r} |')
cmp(50, 75); cmp(100, 75)
print('\n| ridge probe on z | patch 50 | patch 75 | patch 100 |\n|---|---|---|---|')
for ds in ('BNCI2014004', 'BNCI2014001', 'BNCI2014008'):
    row = []
    for p in BB:
        xs = np.array([json.load(open(f'output/{b}/pretrain/analysis/ridge_probe.json'))[ds]['mean'] * 100 for b in BB[p]])
        row.append(f'{xs.mean():.1f} +- {xs.std(ddof=1) / np.sqrt(3):.1f}')
    print(f'| {ds} | ' + ' | '.join(row) + ' |')
