"""Old (overnight, 'scale') vs Compass trial windows ('cw'): per backbone, cell and head, seed-averaged
tail balanced accuracy, paired over subjects. Usage: python compare_windows.py <backbone> [...]"""
import json
import sys

import numpy as np
from scipy.stats import wilcoxon


def scores(bb, label, head, ds, split):
    per = []
    for seed in (1, 2, 3):
        d = json.load(open(f'output/{bb}/finetune/{label}_{head}/{ds}_{split}_seed{seed}/artifacts/group_eval.json'))
        per.append({s: r['tail'] for v in d.values() for s, r in v['groups']['heldout']['subjects'].items()})
    subs = sorted(per[0], key=int)
    return np.array([[p[x] for x in subs] for p in per])


lines = ['# Trial windows: ours vs EEG-FM-Compass (3 finetune seeds)', '',
         '| Backbone | Cell | Head | Old windows | Compass windows | Diff | Subjects better | p |', '|---|---|---|---|---|---|---|---|']
for bb in sys.argv[1:]:
    for ds in ('BNCI2014004', 'BNCI2014001', 'BNCI2014008'):
        for split in ('loso', 'fewshot'):
            for head in ('probe', 'stamp'):
                try:
                    o, n = scores(bb, 'scale', head, ds, split).mean(0), scores(bb, 'cw', head, ds, split).mean(0)
                except FileNotFoundError:
                    continue                                  # cell not run with the Compass windows
                p = wilcoxon(n, o).pvalue if np.any(n != o) else 1.0
                lines.append(f'| {bb} | {ds} {split} | {head} | {o.mean()*100:.1f} | {n.mean()*100:.1f} | '
                             f'{(n - o).mean()*100:+.1f} | {int((n > o).sum())}/{len(o)} | {p:.3f} |')
open('output/reports/overnight/compass_windows.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
