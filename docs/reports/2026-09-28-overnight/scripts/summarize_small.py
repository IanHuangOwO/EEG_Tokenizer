"""Overnight step D: the scale-up mechanism card, tiny winner -> the same recipe on the small corpus (4x
windows, one pretrain). Card set 2026-09-28 before the small run.

Backbone: both scored on the tiny corpus's held-out windows and masks (small via eval_tiny_windows.json).
  Pass per metric: better than tiny by more than 3 x the graded-vs-rankfix spread (near-identical tiny
  runs) and by at least 2%.
Downstream: pre-stamp probe (K 8) and stamp head, loso + few-shot, BNCI2014004 / 001 / 008, finetune
  seeds 1-3. Per cell: mean over seeds (sd over seeds = head-training noise); small vs tiny paired
  Wilcoxon over subjects on the seed-averaged scores.
Usage: python summarize_small.py <tiny_run> <small_run>
"""
import json
import statistics as st
import sys

import numpy as np
from scipy.stats import wilcoxon

tiny, small = sys.argv[1], sys.argv[2]
A = 'output/{}/pretrain/analysis/{}.json'
BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']


def metrics(r):
    be, rp = json.load(open(A.format(r, 'backbone_eval'))), json.load(open(A.format(r, 'ridge_probe')))
    tm, ms = be['test_masks'], be['masked_spectrum']
    band = lambda masks: st.mean(ms[m]['error_ratio'][b] for m in masks for b in BANDS)
    m = {f'masked MSE {k}': (tm[f'{k}|all|model'], -1) for k in
         ('token_runs', 'random_channel', 'channel_cluster', 'time_block', 'motor3_to_bci22')}
    m['random_channel MSE / idw'] = (tm['random_channel|all|model'] / tm['random_channel|all|idw'], -1)
    m['motor3_to_bci22 MSE / idw'] = (tm['motor3_to_bci22|all|model'] / tm['motor3_to_bci22|all|idw'], -1)
    m['band error, channel masks'] = (band(['random_channel', 'channel_cluster']), -1)
    m['band error, time block'] = (band(['time_block']), -1)
    m['band error, unmasked'] = (band(['unmasked']), -1)
    m['seam disagreement'] = (be['seam_disagreement'], -1)
    for ds, v in rp.items():
        m[f'ridge probe {ds}'] = (v['mean'] * 100, +1)
    return m


t, s = metrics(tiny), metrics(small)
g, r = metrics('mesae_tiny_skipdrop_graded_s1'), metrics('mesae_tiny_rankfix_s1')
lines = ['# Scale-up card: tiny -> small corpus', '',
         f'Tiny: `{tiny}`. Small: `{small}` (same recipe, 4x windows, one pretrain). Same held-out windows and masks.', '',
         '## Backbone', '', 'Pass: better than tiny by > max(3 x spread, 2%); spread = graded vs rankfix.', '',
         '| Metric | Tiny | Small | Change | Spread | Pass |', '|---|---|---|---|---|---|']
n_pass = 0
for k, (tv, sign) in t.items():
    sv = s[k][0]
    rel = (sv - tv) / abs(tv)
    spread = abs(g[k][0] - r[k][0]) / abs(r[k][0])
    ok = rel * sign > max(3 * spread, 0.02)
    n_pass += ok
    lines.append(f'| {k} | {tv:.4g} | {sv:.4g} | {rel:+.1%} | {spread:.1%} | {"yes" if ok else "no"} |')
lines += ['', f'{n_pass} / {len(t)} backbone metrics improve beyond the spread.', '',
          '## Downstream (3 finetune seeds)', '',
          '| Cell | Head | Tiny mean (sd seeds) | Small mean (sd seeds) | Diff | Subjects better | p |',
          '|---|---|---|---|---|---|---|']


def scores(bb, head, ds, split):
    per = []
    for seed in (1, 2, 3):
        d = json.load(open(f'output/{bb}/finetune/scale_{head}/{ds}_{split}_seed{seed}/artifacts/group_eval.json'))
        per.append({sub: r['tail'] for v in d.values() for sub, r in v['groups']['heldout']['subjects'].items()})
    subs = sorted(per[0], key=int)
    return subs, np.array([[p[x] for x in subs] for p in per])          # [seed, subject]


for ds in ('BNCI2014004', 'BNCI2014001', 'BNCI2014008'):
    for split in ('loso', 'fewshot'):
        for head in ('probe', 'stamp'):
            subs, T = scores(tiny, head, ds, split)
            _, S = scores(small, head, ds, split)
            tm_, sm_ = T.mean(0), S.mean(0)                              # seed-averaged per subject
            p = wilcoxon(sm_, tm_).pvalue if np.any(sm_ != tm_) else 1.0
            lines.append(f'| {ds} {split} | {head} | {T.mean(1).mean()*100:.1f} ({T.mean(1).std()*100:.1f}) | '
                         f'{S.mean(1).mean()*100:.1f} ({S.mean(1).std()*100:.1f}) | {(sm_ - tm_).mean()*100:+.1f} | '
                         f'{int((sm_ > tm_).sum())}/{len(subs)} | {p:.3f} |')
open('output/reports/overnight/small_card.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
