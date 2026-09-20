"""Stats over train_finetune's subject_groups output (artifacts/group_eval.json).
Usage: python group_summary.py <group_eval.json> [<group_eval.json> ...]
Each file = one head (name = dir above 'artifacts'); the first is the reference.
Per run/group: per-subject tail (mean of last 10 epochs) and mean +- sd. Groups sharing a
name across runs (kfold 'heldout') are pooled per subject. Then, per group, paired
head - reference over the same subjects (subject level, like ft_summary.py); per head,
seen vs unseen (Welch, different subjects). p is information, not a gate."""
import json, os, sys
import numpy as np
from scipy import stats


def head_name(path):
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path))))


def load(path):
    """-> (runs, pooled): pooled[group][subject] = tail, averaged if a subject repeats."""
    runs = json.load(open(path))
    acc = {}
    for run in runs.values():
        for g, d in run['groups'].items():
            for s, v in d['subjects'].items():
                acc.setdefault(g, {}).setdefault(s, []).append(v['tail'])
    return runs, {g: {s: float(np.mean(v)) for s, v in d.items()} for g, d in acc.items()}


def fmt(vals):
    a = np.array(list(vals))
    return f'{a.mean():.3f} +- {a.std(ddof=1) if len(a) > 1 else 0.0:.3f} (n={len(a)})'


def paired(b, a):
    """b - a over shared subjects -> (n, mean diff, p, wins)"""
    subs = sorted(set(a) & set(b), key=lambda s: (len(s), s))
    x = np.array([b[s] for s in subs]); y = np.array([a[s] for s in subs])
    p = stats.ttest_rel(x, y).pvalue if len(subs) > 1 else float('nan')
    return len(subs), float(np.mean(x - y)) if subs else float('nan'), p, int((x > y).sum())


def main(paths):
    heads = {}
    for path in paths:
        runs, pooled = load(path)
        heads[head_name(path)] = pooled
        print(f'== {head_name(path)}')
        for rn, run in runs.items():
            for g, d in run['groups'].items():
                ss = sorted(d['subjects'], key=lambda s: (len(s), s))
                print(f'  {rn}/{g}: ' + ' '.join(f'{s}:{d["subjects"][s]["tail"]:.2f}' for s in ss)
                      + f'   {fmt(d["subjects"][s]["tail"] for s in ss)}')
        for g, d in pooled.items():
            print(f'  pooled {g}: {fmt(d.values())}')
    names = list(heads)
    for n in names[1:]:
        for g in heads[names[0]]:
            if g in heads[n]:
                k, d, p, w = paired(heads[n][g], heads[names[0]][g])
                print(f'{n} - {names[0]} [{g}] (tail, subject-level n={k}): {d:+.3f}  p={p:.3f}  wins {w}/{k}')
    for n, pooled in heads.items():
        if 'seen' in pooled and 'unseen' in pooled:
            s, u = list(pooled['seen'].values()), list(pooled['unseen'].values())
            print(f'{n} seen - unseen: {np.mean(s) - np.mean(u):+.3f}  Welch p={stats.ttest_ind(s, u, equal_var=False).pvalue:.3f}  (n={len(s)} vs {len(u)})')


if __name__ == '__main__':
    main(sys.argv[1:])
