"""One-off: side-by-side finetune table, one row per head/dataset_mode, one column pair
(balanced accuracy tail, Cohen's kappa tail) per backbone, from each run's
artifacts/group_eval.json (mean over every run -> group -> subject entry, same pooling
as tools/analysis/group_summary.py). Missing runs print as '-'.

Usage: python -m tools.misc.compare_backbones mesae_v10_small mesae_v11_small [--heads learned flat]
"""
import argparse
import glob
import json
import os

import numpy as np


def _score(path):
    tails, kappas = [], []
    for run in json.load(open(path)).values():
        for g in run['groups'].values():
            for v in g['subjects'].values():
                tails.append(v['tail'])
                if 'kappa_tail' in v:
                    kappas.append(v['kappa_tail'])
    return np.mean(tails), (np.mean(kappas) if kappas else np.nan), len(tails)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('backbones', nargs='+')
    ap.add_argument('--heads', nargs='+', default=None)
    args = ap.parse_args()

    scores = {}  # (head, dataset_mode) -> {backbone: (acc, kappa, n)}
    for bb in args.backbones:
        for p in glob.glob(f'output/{bb}/finetune/*/*/artifacts/group_eval.json'):
            run_dir = os.path.dirname(os.path.dirname(p))
            head, dm = os.path.basename(os.path.dirname(run_dir)), os.path.basename(run_dir)
            if args.heads and head not in args.heads:
                continue
            scores.setdefault((head, dm), {})[bb] = _score(p)

    w = 16
    print(f"{'head':<11}{'dataset_mode':<22}" + ''.join(f'{bb[:w]:>{w}}{"":>9}' for bb in args.backbones))
    print(f"{'':<33}" + ''.join(f'{"acc":>{w}}{"kappa":>9}' for _ in args.backbones))
    for head, dm in sorted(scores, key=lambda k: (k[0], k[1].rsplit('_', 1)[1], k[1])):
        row = f'{head:<11}{dm:<22}'
        for bb in args.backbones:
            s = scores[(head, dm)].get(bb)
            row += f'{s[0]:>{w}.3f}{s[1]:>9.3f}' if s else f'{"-":>{w}}{"-":>9}'
        print(row)


if __name__ == '__main__':
    main()
