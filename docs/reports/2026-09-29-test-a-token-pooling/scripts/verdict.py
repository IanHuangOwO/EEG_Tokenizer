"""Test A verdict: patch-50 z probe with and without latent_pool 2, against patch 100 (per backbone the mean over
finetune seeds 1-3; mean +- SE over 3 pretrain seeds; won when |diff| > 2 x sqrt(SE_a^2 + SE_b^2))."""
import json, os
import numpy as np
P50 = ['mesae_tiny_notrial_s1', 'mesae_tiny_p50_s16_s2', 'mesae_tiny_p50_s16_s3']
P100 = ['mesae_tiny_p100_s16_s1', 'mesae_tiny_p100_s16_s2', 'mesae_tiny_p100_s16_s3']
CELLS = [f'{d}_{s}' for s in ('loso', 'fewshot') for d in ('BNCI2014004', 'BNCI2014001', 'BNCI2014008')]
def acc(bb, head, cell):
    out = []
    for s in (1, 2, 3):
        p = f'output/{bb}/finetune/{head}/{cell}_seed{s}/artifacts/group_eval.json'
        if not os.path.exists(p): continue
        d = json.load(open(p))
        out.append(np.mean([v['tail'] for f in d.values() for g in f['groups'].values() for v in g['subjects'].values()]) * 100)
    return np.mean(out) if out else np.nan
f = lambda v: f"{np.nanmean(v):.1f} ({' / '.join('--' if np.isnan(u) else f'{u:.1f}' for u in v)})"
print('| Cell | p50 | p50 pool2 | p100 | pool2 - p50 | p100 - p50 |\n|---|---|---|---|---|---|')
for c in CELLS:
    a = np.array([acc(b, 'patch_probe', c) for b in P50]); p = np.array([acc(b, 'patch_probe_pool2', c) for b in P50])
    h = np.array([acc(b, 'patch_probe', c) for b in P100])
    print(f'| {c} | {f(a)} | {f(p)} | {f(h)} | {np.nanmean(p) - np.nanmean(a):+.1f} | {np.nanmean(h) - np.nanmean(a):+.1f} |')
print('\n| ridge | p50 | p50 pool2 | p100 |\n|---|---|---|---|')
for ds in ('BNCI2014004', 'BNCI2014001', 'BNCI2014008'):
    r = lambda bbs, fn: np.array([json.load(open(f'output/{b}/pretrain/analysis/{fn}'))[ds]['mean'] * 100 for b in bbs])
    print(f"| {ds} | {f(r(P50, 'ridge_probe.json'))} | {f(r(P50, 'ridge_probe_pool2.json'))} | {f(r(P100, 'ridge_probe.json'))} |")
print('\n| Cell | p50 pool2 vs p50: diff | threshold | result |\n|---|---|---|---|')
se = lambda v: v.std(ddof=1) / np.sqrt(3)
for c in CELLS:
    a = np.array([acc(b, 'patch_probe', c) for b in P50]); p = np.array([acc(b, 'patch_probe_pool2', c) for b in P50])
    d, t = p.mean() - a.mean(), 2 * np.sqrt(se(a) ** 2 + se(p) ** 2)
    print(f"| {c} | {d:+.1f} | {t:.1f} | {'pool2 wins' if d > t else 'pool2 loses' if d < -t else 'tie'} |")
