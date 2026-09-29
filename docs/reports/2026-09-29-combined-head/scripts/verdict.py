"""Combined-head card verdict: per cell, combined vs each parent (z probe, stamp head) over the 3 tiny patch-50 seeds
(mean over finetune seeds per backbone; mean +- SE over backbones; won when |diff| > 2 x sqrt(SE_a^2 + SE_b^2)),
the small backbone's one-seed numbers, and the train / held-out gap per head."""
import glob, json, os, re
import numpy as np
BB = ['mesae_tiny_p50_s16_s1', 'mesae_tiny_p50_s16_s2', 'mesae_tiny_p50_s16_s3']
CELLS = [f'{d}_{s}' for s in ('loso', 'fewshot') for d in ('BNCI2014004', 'BNCI2014001', 'BNCI2014008')]
HEADS = {'z': 'patch_probe', 'stamp': 'cw_stamp', 'combined': 'combined'}
def runs(bb, head, cell):
    return [f'output/{bb}/finetune/{head}/{cell}_seed{s}/artifacts' for s in (1, 2, 3)
            if os.path.exists(f'output/{bb}/finetune/{head}/{cell}_seed{s}/artifacts/group_eval.json')]
def test(d):
    g = json.load(open(f'{d}/group_eval.json'))
    return np.mean([v['tail'] for f in g.values() for x in f['groups'].values() for v in x['subjects'].values()]) * 100
def train(d):
    folds, fold = {}, None
    for log in glob.glob(f'{d}/train_*.log'):
        for line in open(log, errors='ignore'):
            m = re.search(r'\[(\S+)\] Epoch \d+/\d+ Summary', line)
            if m: fold = m.group(1)
            m = re.search(r'\[Train\].*bal_acc: ([\d.]+)', line)
            if m and fold: folds.setdefault(fold, []).append(float(m.group(1)))
    return np.mean([np.mean(v[-10:]) for v in folds.values()]) * 100
def per_bb(head, cell, f=test):
    return np.array([np.mean([f(d) for d in runs(b, head, cell)]) for b in BB])
se = lambda v: v.std(ddof=1) / np.sqrt(len(v))
fmt = lambda v: f"{v.mean():.1f} ({' / '.join(f'{u:.1f}' for u in v)})"
print('| Cell | z probe | stamp head | combined | vs z (thr) | vs stamp (thr) |\n|---|---|---|---|---|---|')
for c in CELLS:
    z, s, m = (per_bb(HEADS[k], c) for k in ('z', 'stamp', 'combined'))
    res = []
    for p in (z, s):
        d, t = m.mean() - p.mean(), 2 * np.sqrt(se(m) ** 2 + se(p) ** 2)
        res.append(f"{d:+.1f} ({t:.1f}) {'win' if d > t else 'loss' if d < -t else 'tie'}")
    print(f'| {c} | {fmt(z)} | {fmt(s)} | {fmt(m)} | ' + ' | '.join(res) + ' |')
print('\n| Cell | z probe train / test (gap) | stamp head | combined |\n|---|---|---|---|')
for c in CELLS:
    row = []
    for k in ('z', 'stamp', 'combined'):
        tr, te = per_bb(HEADS[k], c, train).mean(), per_bb(HEADS[k], c).mean()
        row.append(f'{tr:.1f} / {te:.1f} ({tr - te:+.1f})')
    print(f'| {c} | ' + ' | '.join(row) + ' |')
print('\n| Cell (small, 1 seed) | z probe | stamp head | combined | combined train / test (gap) |\n|---|---|---|---|---|')
sm = 'mesae_small_p50_s16_s1'
for c in CELLS:
    v = lambda h: np.mean([test(d) for d in runs(sm, h, c)])
    cd = runs(sm, 'combined', c)
    tr, te = np.mean([train(d) for d in cd]), np.mean([test(d) for d in cd])
    print(f"| {c} | {v('patch_probe'):.1f} | {v('cw_stamp'):.1f} | {te:.1f} | {tr:.1f} / {te:.1f} ({tr - te:+.1f}) |")
