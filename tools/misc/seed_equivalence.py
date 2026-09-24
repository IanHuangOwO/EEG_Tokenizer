"""One-off: multi-seed backbone comparison with an equivalence (TOST) test.

Reads output/<backbone>/finetune/<head>_seed<s>/<dataset_mode>/artifacts/group_eval.json
(configs in config/runs/<backbone>/finetune/<head>_seed<s>/; seeds change only weight
init + batch order -- training_params.finetune.seed -- folds stay fixed via split.seed).

Per backbone x dataset_mode: mean +- std of the subject-mean balanced accuracy (tail)
across seeds. Per dataset_mode, B - A (second backbone minus first): each subject's score
is averaged over the seeds both backbones finished, then a paired comparison across
subjects gives the mean difference, its 90% CI, a paired t-test p, and a TOST p against
+-margin. "equivalent" = the 90% CI lies inside [-margin, +margin] (TOST at alpha=0.05).

Usage: python -m tools.misc.seed_equivalence mesae_v10_small mesae_v11_small
  [--head learned] [--margin 0.02] [--metric tail|kappa_tail]
"""
import argparse
import glob
import os
import re

import numpy as np
from scipy import stats

from tools.analysis.group_summary import load


def _collect(backbone, head, metric):
    """-> {dataset_mode: {seed: {subject: score}}}"""
    out = {}
    for p in glob.glob(f'output/{backbone}/finetune/{head}_seed*/*/artifacts/group_eval.json'):
        run_dir = os.path.dirname(os.path.dirname(p))
        seed = int(re.search(r'_seed(\d+)$', os.path.basename(os.path.dirname(run_dir))).group(1))
        _, pooled = load(p, metric=metric)
        subj = {s: v for g in pooled.values() for s, v in g.items()}
        if subj:
            out.setdefault(os.path.basename(run_dir), {})[seed] = subj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--head', default='learned')
    ap.add_argument('--margin', type=float, default=0.02)
    ap.add_argument('--metric', default='tail')
    args = ap.parse_args()

    A, B = _collect(args.a, args.head, args.metric), _collect(args.b, args.head, args.metric)
    m = args.margin
    print(f"{args.head} / {args.metric}, B={args.b} minus A={args.a}, equivalence margin +-{m}\n")
    print(f"{'dataset_mode':<20}{'A mean+-sd':>16}{'B mean+-sd':>16}{'seeds':>7}{'B-A':>8}"
          f"{'90% CI':>18}{'p_diff':>8}{'p_TOST':>8}  verdict")
    for dm in sorted(set(A) & set(B), key=lambda d: (d.rsplit('_', 1)[1], d)):
        seeds = sorted(set(A[dm]) & set(B[dm]))
        if not seeds:
            continue
        per_seed = lambda X: [np.mean(list(X[dm][s].values())) for s in seeds]
        a_s, b_s = per_seed(A), per_seed(B)
        subs = sorted(set.intersection(*(set(A[dm][s]) & set(B[dm][s]) for s in seeds)), key=lambda x: (len(x), x))
        a = np.array([np.mean([A[dm][s][u] for s in seeds]) for u in subs])
        b = np.array([np.mean([B[dm][s][u] for s in seeds]) for u in subs])
        d = b - a
        n, se = len(d), d.std(ddof=1) / np.sqrt(len(d))
        t90 = stats.t.ppf(0.95, n - 1)
        lo, hi = d.mean() - t90 * se, d.mean() + t90 * se
        p_diff = stats.ttest_rel(b, a).pvalue
        p_tost = max(stats.t.sf((d.mean() + m) / se, n - 1), stats.t.cdf((d.mean() - m) / se, n - 1))
        verdict = 'equivalent' if lo > -m and hi < m else ('different' if p_diff < 0.05 else 'inconclusive')
        sd = lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0
        print(f"{dm:<20}{np.mean(a_s):>9.3f}+-{sd(a_s):.3f}{np.mean(b_s):>9.3f}+-{sd(b_s):.3f}{len(seeds):>7}"
              f"{d.mean():>+8.3f}   [{lo:+.3f},{hi:+.3f}]{p_diff:>8.2f}{p_tost:>8.3f}  {verdict}")


if __name__ == '__main__':
    main()
