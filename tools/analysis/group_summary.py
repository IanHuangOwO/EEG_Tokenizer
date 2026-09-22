"""Stats over train_finetune.py's subject_groups output (artifacts/group_eval.json).
Ported from probes/group_summary.py -- schema unchanged (train_finetune.py:276-327 still
writes {run_name: {'groups': {group: {'subjects': {subject: {'tail','last','n_trials'}}}}}}).
Each file = one head (name = dir above 'artifacts'); the first is the reference. Per
run/group: per-subject tail (mean of last 10 epochs) and mean +- sd. Groups sharing a
name across runs (kfold 'heldout') are pooled per subject. Then, per group, paired
head - reference over the same subjects (subject level); per head, seen vs unseen
(Welch, different subjects). p is information, not a gate.
"""
import json
import os

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
    p = float('nan')
    if len(subs) >= 5 and np.ptp(x - y) > 0:  # constant diffs -> NaN t-test
        p = stats.ttest_rel(x, y).pvalue
    return len(subs), float(np.mean(x - y)) if subs else float('nan'), p, int((x > y).sum())


def print_group_summary(paths):
    """Prints the full report for one or more group_eval.json paths -- the first path is
    the reference every later one gets paired against. Pure side-effecting print, no
    return value (matches the tools/panels/ contract: the analysis layer prints/saves,
    the panel just passes paths through)."""
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
                ps = 'descriptive only, n<5' if k < 5 else f'p={p:.3f}'
                print(f'{n} - {names[0]} [{g}] (tail, subject-level n={k}): {d:+.3f}  {ps}  wins {w}/{k}')
    for n, pooled in heads.items():
        if 'seen' in pooled and 'unseen' in pooled:
            s, u = list(pooled['seen'].values()), list(pooled['unseen'].values())
            w = (f'Welch p={stats.ttest_ind(s, u, equal_var=False).pvalue:.3f}'
                 if min(len(s), len(u)) >= 5 else 'descriptive only, n<5')
            print(f'{n} seen - unseen: {np.mean(s) - np.mean(u):+.3f}  {w}  (n={len(s)} vs {len(u)})')
