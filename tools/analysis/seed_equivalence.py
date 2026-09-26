"""Multi-seed backbone comparison with an equivalence (TOST) test.

Reads output/<backbone>/finetune/<head>_seed<s>/<cell>/artifacts/group_eval.json (seeds change only
weight init and batch order, training_params.finetune.seed; folds stay fixed via split.seed).
Per cell: mean +- std across seeds of the subject-mean score for both backbones; then each subject's
score averaged over the seeds both finished, paired over subjects: mean difference B - A, its 90% CI,
a paired t-test p, and a TOST p against +-margin. "equivalent" = the 90% CI lies inside the margin
(TOST at alpha 0.05).
"""
import glob
import os
import re

import numpy as np
from scipy import stats

from tools.analysis.summarize_runs import load


def collect(backbone, head, metric='tail'):
    """-> {cell: {seed: {subject: score}}}"""
    out = {}
    for d in glob.glob(f'output/{backbone}/finetune/{head}_seed*/*'):
        m = re.search(r'_seed(\d+)$', os.path.basename(os.path.dirname(d)))
        if m and (subj := load(d, metric)):
            out.setdefault(os.path.basename(d), {})[int(m.group(1))] = subj
    return out


def compare(a, b, head, margin=0.02, metric='tail'):
    """Backbone b minus backbone a, per cell -> markdown table (empty string if no seeded runs)."""
    A, B = collect(a, head, metric), collect(b, head, metric)
    rows = []
    for cell in sorted(set(A) & set(B)):
        seeds = sorted(set(A[cell]) & set(B[cell]))
        subs = sorted(set.intersection(*(set(A[cell][s]) & set(B[cell][s]) for s in seeds))) if seeds else []
        if len(subs) < 3:
            continue
        sd = lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0
        a_s = [np.mean(list(A[cell][s].values())) for s in seeds]
        b_s = [np.mean(list(B[cell][s].values())) for s in seeds]
        xa = np.array([np.mean([A[cell][s][u] for s in seeds]) for u in subs])
        xb = np.array([np.mean([B[cell][s][u] for s in seeds]) for u in subs])
        d = xb - xa
        n, se = len(d), d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = d.mean() + np.array([-1, 1]) * stats.t.ppf(0.95, n - 1) * se
        p_diff = stats.ttest_rel(xb, xa).pvalue
        p_tost = max(stats.t.sf((d.mean() + margin) / se, n - 1), stats.t.cdf((d.mean() - margin) / se, n - 1))
        verdict = 'equivalent' if lo > -margin and hi < margin else ('different' if p_diff < 0.05 else 'inconclusive')
        rows.append(f'| {cell} | {np.mean(a_s):.3f} +- {sd(a_s):.3f} | {np.mean(b_s):.3f} +- {sd(b_s):.3f} | '
                    f'{len(seeds)} | {d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p_diff:.2f} | {p_tost:.3f} | {verdict} |')
    if not rows:
        return ''
    return '\n'.join([f'\n### {b} minus {a} ({head}_seed*, {metric}, margin +-{margin})\n',
                      '| cell | A mean +- sd | B mean +- sd | seeds | B-A | 90% CI | p diff | p TOST | verdict |',
                      '|---|---|---|---|---|---|---|---|---|'] + rows)
