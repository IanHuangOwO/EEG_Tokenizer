"""
One table for any set of finetune runs (each a dir with artifacts/group_eval.json), whatever
produced them: a backbone comparison, a head comparison, a hyperparameter grid.

A run dir output/<backbone>/finetune/<label...>/<leaf> is read as column '<backbone>:<label>'
and row '<leaf>' (for learned/BNCI2014001_loso: label 'learned', row 'BNCI2014001_loso'; for
tune/mi_loso/lr0.001_...: label 'tune/mi_loso', row 'lr0.001_...'). A cell is the mean over
subjects of each subject's mean over folds (metric: tail | last | kappa_tail | kappa_last).

    # backbones x heads on the benchmark cells, paired test of every column against a reference
    python -m tools.analysis.summarize_runs 'output/mesae_tiny_static16_*_s1/finetune/learned/*' \\
        --ref mesae_tiny_static16_base_s1:learned
    # a grid: rank the configs of each column (top 10)
    python -m tools.analysis.summarize_runs 'output/mesae_tiny_static16_base_s1/finetune/tune/*/*' --rank 10
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats


def load(run_dir, metric):
    p = os.path.join(run_dir, 'artifacts', 'group_eval.json')
    if not os.path.exists(p):
        return None
    subj = {}
    for fold in json.load(open(p)).values():
        for g in fold['groups'].values():
            for s, x in g['subjects'].items():
                subj.setdefault(s, []).append(x[metric])
    return {s: float(np.mean(v)) for s, v in subj.items()}


def parse(run_dir):
    rel = os.path.relpath(run_dir, 'output').split(os.sep)
    k = rel.index('finetune')
    return '/'.join(rel[:k]), '/'.join(rel[k + 1:-1]), rel[-1]      # backbone, label, row


def collect(patterns, metric='tail'):
    """Globs of run dirs -> {(column 'backbone:label', row leaf): {subject: score}}."""
    table = {}
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            if os.path.isdir(d) and 'finetune' in d.split(os.sep):
                bb, label, row = parse(d)
                scores = load(d, metric)
                if scores:
                    table[(f'{bb}:{label}', row)] = scores
    return table


def render(table, metric='tail', ref=None, rank=0):
    """The table as markdown (plus paired tests of every column against `ref`), or with rank=N the
    top N rows per column."""
    cols = sorted({c for c, _ in table}); rows = sorted({r for _, r in table})
    mean = {k: 100 * np.mean(list(v.values())) for k, v in table.items()}
    out = []
    if rank:
        for c in cols:
            ranked = sorted(((mean[(c, r)], r, len(table[(c, r)])) for r in rows if (c, r) in table), reverse=True)
            out.append(f'\n## {c} ({metric}, %; {len(ranked)} runs)')
            out += [f'  {i:2}. {m:5.1f}  {r}  (n={n})' for i, (m, r, n) in enumerate(ranked[:rank], 1)]
        return '\n'.join(out)
    w = max(len(r) for r in rows)
    out.append(f'\n{metric} balanced accuracy %' + (' (kappa)' if 'kappa' in metric else '') + ', mean over subjects')
    out.append(f'| {"":{w}} | ' + ' | '.join(cols) + ' |')
    out.append(f'|{"-" * (w + 2)}|' + '|'.join('-' * (len(c) + 2) for c in cols) + '|')
    for r in rows:
        out.append(f'| {r:{w}} | ' + ' | '.join(f'{mean[(c, r)]:{len(c)}.1f}' if (c, r) in table else f'{"-":>{len(c)}}'
                                                for c in cols) + ' |')
    if ref:
        out.append(f'\npaired over subjects, each column minus {ref}:')
        for c in cols:
            if c == ref:
                continue
            for r in rows:
                a, b = table.get((c, r)), table.get((ref, r))
                if not a or not b:
                    continue
                d = [100 * (a[s] - b[s]) for s in a if s in b]
                if len(d) > 2:
                    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=np.mean(d), scale=stats.sem(d))
                    out.append(f'  {c} | {r}: {np.mean(d):+5.1f}  95% CI [{lo:+.1f}, {hi:+.1f}]  '
                               f'p={stats.ttest_1samp(d, 0).pvalue:.3f}  (n={len(d)})')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('patterns', nargs='+', help='glob(s) of run dirs')
    ap.add_argument('--metric', default='tail', choices=['tail', 'last', 'kappa_tail', 'kappa_last'])
    ap.add_argument('--ref', default=None, help="column 'backbone:label' to test the others against")
    ap.add_argument('--rank', type=int, default=0, help='instead of the table: top N rows per column')
    args = ap.parse_args()
    table = collect(args.patterns, args.metric)
    if not table:
        raise SystemExit('no runs with artifacts/group_eval.json matched')
    print(render(table, args.metric, args.ref, args.rank))


if __name__ == '__main__':
    main()
