"""Stats over train_finetune.py's subject_groups output (artifacts/group_eval.json).
Ported from probes/group_summary.py -- schema unchanged (train_finetune.py:276-327 still
writes {run_name: {'groups': {group: {'subjects': {subject: {'tail','last','n_trials'}}}}}}).
Each file = one head; the first is the reference. Per run/group: per-subject tail (mean
of last 10 epochs) and mean +- sd. Groups sharing a name across runs (kfold 'heldout')
are pooled per subject. Then, per group, paired head - reference over the same subjects
(subject level); per head, seen vs unseen (Welch, different subjects). p is information,
not a gate.
"""
import csv
import json
import os

import numpy as np
from scipy import stats


def _locate(path):
    """-> (backbone_dir, label). Current layout nests a finetune run under its backbone --
    output/<backbone>/finetune/<head>/<dataset>_<mode>/artifacts/group_eval.json -- so the
    head lives 3 dirs above the file, not 2 (the old output/baseline/<dataset>_<mode>_<head>/
    layout encoded the head in the run's own leaf dir name, 1 level above 'artifacts'; that
    layout is detected here by the ABSENCE of a 'finetune' marker dir 4 levels up, and still
    supported since output/archive/'s older runs and output/baseline/'s own raw_signal runs
    both use it -- backbone_dir is None there, label is just the run's own leaf dir name).
    label is '<head>/<dataset>_<mode>' (not just '<head>') for the new layout -- a
    whole-tree glob (e.g. 'output/<backbone>/finetune/*/*/artifacts/group_eval.json', see
    panel_group_summary.py) spans every dataset_mode too, and two different datasets under
    the SAME head would otherwise collide on one dict key and silently overwrite each
    other's numbers. backbone_dir (output/<backbone>, the top-level per-backbone dir) is
    where write_group_summary_csv saves its CSVs by default (backbone_dir/finetune/analysis/)."""
    p = os.path.abspath(path)
    run_dir = os.path.dirname(os.path.dirname(p))          # .../<dataset>_<mode> (or old-style leaf)
    finetune_marker = os.path.dirname(os.path.dirname(run_dir))
    if os.path.basename(finetune_marker) == 'finetune':
        head = os.path.basename(os.path.dirname(run_dir))  # .../finetune/<head>/<dataset>_<mode>
        label = f'{head}/{os.path.basename(run_dir)}'
        return os.path.dirname(finetune_marker), label
    return None, os.path.basename(run_dir)


def head_name(path):
    return _locate(path)[1]


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


def _collect(paths):
    """-> (names [in path order], runs_by_name, pooled_by_name, backbone_dir). Loads every
    path once -- shared by print_group_summary and write_group_summary_csv so a big
    whole-tree glob doesn't get parsed twice. backbone_dir is the first path's (see
    _locate), used as the CSV save location's default."""
    names, runs_by_name, pooled_by_name = [], {}, {}
    backbone_dir = None
    for i, path in enumerate(paths):
        runs, pooled = load(path)
        bd, name = _locate(path)
        if i == 0:
            backbone_dir = bd
        names.append(name)
        runs_by_name[name] = runs
        pooled_by_name[name] = pooled
    return names, runs_by_name, pooled_by_name, backbone_dir


def print_group_summary(paths):
    """Prints the full report for one or more group_eval.json paths -- the first path is
    the reference every later one gets paired against. Pure side-effecting print, no
    return value (matches the tools/panels/ contract: the analysis layer prints/saves,
    the panel just passes paths through)."""
    names, runs_by_name, pooled_by_name, _ = _collect(paths)
    for n in names:
        runs, pooled = runs_by_name[n], pooled_by_name[n]
        print(f'== {n}')
        for rn, run in runs.items():
            for g, d in run['groups'].items():
                ss = sorted(d['subjects'], key=lambda s: (len(s), s))
                print(f'  {rn}/{g}: ' + ' '.join(f'{s}:{d["subjects"][s]["tail"]:.2f}' for s in ss)
                      + f'   {fmt(d["subjects"][s]["tail"] for s in ss)}')
        for g, d in pooled.items():
            print(f'  pooled {g}: {fmt(d.values())}')
    for n in names[1:]:
        for g in pooled_by_name[names[0]]:
            if g in pooled_by_name[n]:
                k, d, p, w = paired(pooled_by_name[n][g], pooled_by_name[names[0]][g])
                ps = 'descriptive only, n<5' if k < 5 else f'p={p:.3f}'
                print(f'{n} - {names[0]} [{g}] (tail, subject-level n={k}): {d:+.3f}  {ps}  wins {w}/{k}')
    for n in names:
        pooled = pooled_by_name[n]
        if 'seen' in pooled and 'unseen' in pooled:
            s, u = list(pooled['seen'].values()), list(pooled['unseen'].values())
            w = (f'Welch p={stats.ttest_ind(s, u, equal_var=False).pvalue:.3f}'
                 if min(len(s), len(u)) >= 5 else 'descriptive only, n<5')
            print(f'{n} seen - unseen: {np.mean(s) - np.mean(u):+.3f}  {w}  (n={len(s)} vs {len(u)})')


def write_group_summary_csv(paths, out_dir=None):
    """Writes two CSVs from the same paths print_group_summary reports on -- the first
    path is the reference every later one is paired against, same as there:

    - group_summary.csv (page 1, the rollup): one row per (label, group) -- pooled
      mean/std/n, and (for every label after the first) the paired diff/p/wins against
      the reference for that same group.
    - group_summary_folds.csv (page 2, the detail): one row per (label, group, run/fold
      tag, subject) -- the raw per-fold tail/last/n_trials that group_summary.csv's
      pooled numbers were averaged from, so a fold that looks off can be traced back.

    out_dir defaults to the first path's own backbone_dir/finetune/analysis/ (see
    _locate); pass it explicitly to save somewhere else, or when paths use the old
    layout (_locate's backbone_dir is None there, no default to fall back to -- pass
    out_dir yourself). Returns (summary_csv_path, folds_csv_path)."""
    names, runs_by_name, pooled_by_name, backbone_dir = _collect(paths)
    if out_dir is None:
        if backbone_dir is None:
            raise ValueError("can't auto-derive out_dir from an old-layout path -- pass out_dir explicitly")
        out_dir = os.path.join(backbone_dir, 'finetune', 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, 'group_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label', 'group', 'mean', 'std', 'n', 'diff_vs_reference', 'p_value', 'wins', 'n_paired'])
        ref = names[0]
        for n in names:
            pooled = pooled_by_name[n]
            for g, d in pooled.items():
                vals = np.array(list(d.values()))
                std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                if n == ref or g not in pooled_by_name[ref]:
                    diff = p_val = wins = n_paired = ''
                else:
                    n_paired, diff, p_val, wins = paired(d, pooled_by_name[ref][g])
                    diff, p_val = f'{diff:.6f}', f'{p_val:.6f}'
                w.writerow([n, g, f'{vals.mean():.6f}', f'{std:.6f}', len(vals), diff, p_val, wins, n_paired])

    folds_path = os.path.join(out_dir, 'group_summary_folds.csv')
    with open(folds_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label', 'run', 'group', 'subject', 'tail', 'last', 'n_trials'])
        for n in names:
            for rn, run in runs_by_name[n].items():
                for g, d in run['groups'].items():
                    for s, v in d['subjects'].items():
                        w.writerow([n, rn, g, s, v['tail'], v['last'], v['n_trials']])

    print(f'  wrote {summary_path}')
    print(f'  wrote {folds_path}')
    return summary_path, folds_path
