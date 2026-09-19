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
def subject_tail_means(per):
    """Collapse a run's per-tag tail-means to one value per SUBJECT, averaging folds
    together first. A CV run's tags are f'{ds}_f{k}_S{s}' -- 5 folds per subject are not
    independent samples (same subject, same held-out data pool), so pairing/testing on raw
    per-tag rows pseudoreplicates n (e.g. 9 subjects x 5 folds -> 45 "samples") and can
    manufacture significance that isn't there (this is exactly the bug that produced a
    fold-level p=0.022 where the correct subject-level stat is p=0.248 -- see ADR 0014
    step 8). For the older non-CV format (one row per subject, tag f'{ds}_S{s}', no fold
    component) every subject already has exactly one tag, so this grouping is a no-op.
    """
    by_subj = {}
    for tag, vals in per.items():
        subj = tag.split('_S')[-1]
        by_subj.setdefault(subj, []).append(np.mean(vals[-10:]))
    return {subj: np.mean(tail_means) for subj, tail_means in by_subj.items()}


names = list(runs)
if len(names) > 1:
    ref = subject_tail_means(runs[names[0]])
    for n in names[1:]:
        cur = subject_tail_means(runs[n])
        subs = sorted(set(ref) & set(cur), key=lambda s: int(s))
        a = np.array([ref[s] for s in subs]); b = np.array([cur[s] for s in subs])
        print(f'{n} - {names[0]} (tail, subject-level n={len(subs)}): {np.mean(b - a):+.3f}  p={stats.ttest_rel(b, a).pvalue:.3f}  wins {(b > a).sum()}/{len(subs)}')
