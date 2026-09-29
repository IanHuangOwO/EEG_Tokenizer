"""Overfitting per head: train balanced accuracy (mean of each fold's last 10 epochs, dropout on) vs held-out tail.
Mean over finished finetune seeds; per backbone and cell."""
import glob, json, os, re, sys
import numpy as np
BB = ['mesae_tiny_p50_s16_s1', 'mesae_tiny_p50_s16_s2', 'mesae_tiny_p50_s16_s3']
CELLS = sys.argv[1:] or [f'{d}_{s}' for s in ('loso', 'fewshot') for d in ('BNCI2014004', 'BNCI2014001', 'BNCI2014008')]
def gap(bb, head, cell):
    tr, te = [], []
    for s in (1, 2, 3):
        d = f'output/{bb}/finetune/{head}/{cell}_seed{s}/artifacts'
        if not os.path.exists(f'{d}/group_eval.json'):
            continue
        folds = {}
        for log in glob.glob(f'{d}/train_*.log'):
            fold = None
            for line in open(log, errors='ignore'):
                m = re.search(r'\[(\S+)\] Epoch \d+/\d+ Summary', line)
                if m: fold = m.group(1)
                m = re.search(r'\[Train\].*bal_acc: ([\d.]+)', line)
                if m and fold: folds.setdefault(fold, []).append(float(m.group(1)))
        tr.append(np.mean([np.mean(v[-10:]) for v in folds.values()]) * 100)
        g = json.load(open(f'{d}/group_eval.json'))
        te.append(np.mean([v['tail'] for f in g.values() for x in f['groups'].values() for v in x['subjects'].values()]) * 100)
    return (np.mean(tr), np.mean(te)) if tr else None
print('| Cell | bb | z probe train/test (gap) | stamp head | combined |')
for c in CELLS:
    for b in BB:
        row = []
        for h in ('patch_probe', 'cw_stamp', 'combined'):
            r = gap(b, h, c)
            row.append('--' if r is None else f'{r[0]:.1f}/{r[1]:.1f} ({r[0]-r[1]:+.1f})')
        print(f'| {c} | {b[-2:]} | ' + ' | '.join(row) + ' |')
