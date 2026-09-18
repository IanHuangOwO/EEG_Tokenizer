"""Per-subject val balanced_acc from train_finetune logs: best epoch (by val bal_acc), last
epoch, and the max-over-last-10 mean. Optional paired t between runs.
Usage: python ft_summary.py <log> [<log> ...]"""
import os, re, sys
import numpy as np
from scipy import stats
runs = {}
for path in sys.argv[1:]:
    per = {}
    tag = None
    for line in open(path, encoding='utf-8', errors='ignore'):
        m = re.search(r'--- \[(\S+)\] Epoch (\d+)/(\d+) Summary', line)
        if m:
            tag = m.group(1); continue
        m = re.search(r'\[Val\].*bal_acc: ([0-9.]+)', line)
        if m and tag:
            per.setdefault(tag, []).append(float(m.group(1)))
    name = os.path.basename(path)[:-4]
    runs[name] = per
    subs = sorted(per, key=lambda t: int(t.split('_S')[-1]))
    best = np.array([max(per[s]) for s in subs]); last = np.array([per[s][-1] for s in subs])
    tail = np.array([np.mean(per[s][-10:]) for s in subs])
    print(f'{name}: {len(subs)} subjects, {len(per[subs[0]])} epochs')
    print('   ' + ' '.join(f'{s.split("_")[-1]:>5s}' for s in subs))
    print('best ' + ' '.join(f'{v:5.2f}' for v in best) + f'   mean {best.mean():.3f}')
    print('last ' + ' '.join(f'{v:5.2f}' for v in last) + f'   mean {last.mean():.3f}')
    print('tail ' + ' '.join(f'{v:5.2f}' for v in tail) + f'   mean {tail.mean():.3f}  (mean of last 10 epochs)')
names = list(runs)
if len(names) > 1:
    ref = runs[names[0]]
    for n in names[1:]:
        subs = sorted(set(ref) & set(runs[n]), key=lambda t: int(t.split('_S')[-1]))
        a = np.array([np.mean(ref[s][-10:]) for s in subs]); b = np.array([np.mean(runs[n][s][-10:]) for s in subs])
        print(f'{n} - {names[0]} (tail): {np.mean(b - a):+.3f}  p={stats.ttest_rel(b, a).pvalue:.3f}  wins {(b > a).sum()}/{len(subs)}')
