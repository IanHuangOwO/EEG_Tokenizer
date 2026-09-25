"""
Compare groups of backbones (each group = the same recipe over pretrain seeds) on the finetune
results and on tools/misc/backbone_eval.py's summaries.

Downstream: per group and cell (head/dataset_mode), mean +- sd over seeds of the subject-mean
tail balanced accuracy; each group against --ref paired over (subject, seed index) pairs:
mean difference, 95% CI, paired t-test p. Imputation: learned_impute minus learned on the
same backbone. Backbone eval: test-mask masked MSE (model, and model / best model-free
baseline), ablation changes and structure, averaged over the group's seeds.

Usage: python -m tools.misc.compare_backbones --ref base \\
    --group base=mesae_tiny_static16_base_s1,mesae_tiny_static16_base_s2 \\
    --group A=mesae_tiny_static16_groupA_s1,mesae_tiny_static16_groupA_s2 \\
    --group B=mesae_tiny_static16_groupB_s1,mesae_tiny_static16_groupB_s2
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

CELLS = ['BNCI2014001_loso', 'BNCI2014001_fewshot', 'BNCI2014004_loso', 'BNCI2014004_fewshot']


def subject_scores(backbone, head, cell):
    p = f'output/{backbone}/finetune/{head}/{cell}/artifacts/group_eval.json'
    if not os.path.exists(p):
        return None
    subj = {}
    for fold in json.load(open(p)).values():
        for s, x in fold['groups']['heldout']['subjects'].items():
            subj.setdefault(s, []).append(x['tail'])
    return {s: float(np.mean(v)) for s, v in subj.items()}


def downstream(groups, ref):
    print('\n## Downstream (tail balanced accuracy %, mean +- sd over pretrain seeds)')
    for head in ('learned', 'learned_impute'):
        for cell in CELLS:
            per = {g: [subject_scores(b, head, cell) for b in bbs] for g, bbs in groups.items()}
            if not any(x for v in per.values() for x in v):
                continue
            line = f'  {head:15} {cell:20}'
            for g, runs in per.items():
                means = [100 * np.mean(list(r.values())) for r in runs if r]
                line += f' | {g}: ' + (f'{np.mean(means):5.1f} +- {np.std(means):3.1f} (n={len(means)})' if means else '  -  ')
            print(line)
            for g, runs in per.items():
                if g == ref:
                    continue
                d = [100 * (r[s] - rr[s]) for r, rr in zip(runs, per[ref]) if r and rr for s in r if s in rr]
                if len(d) > 2:
                    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=np.mean(d), scale=stats.sem(d))
                    print(f'      {g} - {ref}: {np.mean(d):+5.1f}  95% CI [{lo:+.1f}, {hi:+.1f}]  '
                          f'p={stats.ttest_1samp(d, 0).pvalue:.3f}  ({len(d)} subject x seed pairs)')
    print('\n## Imputation gain (learned_impute - learned, same backbone)')
    for g, bbs in groups.items():
        for b in bbs:
            gains = []
            for cell in CELLS:
                a, i = subject_scores(b, 'learned', cell), subject_scores(b, 'learned_impute', cell)
                if a and i:
                    gains.append(f'{cell} {100 * (np.mean(list(i.values())) - np.mean(list(a.values()))):+.1f}')
            if gains:
                print(f'  {g} {b}: ' + ', '.join(gains))


def backbone(groups):
    print('\n## Backbone eval (held-out windows, identical masks; lower MSE is better)')
    evals = {g: [json.load(open(p)) for b in bbs
                 if os.path.exists(p := f'output/{b}/pretrain/analysis/backbone_eval.json')] for g, bbs in groups.items()}
    kinds = ['token_runs', 'random_channel', 'channel_cluster', 'time_block', 'motor3_to_bci22']
    print(f'  {"test mask":17}' + ''.join(f' | {g:>22}' for g in evals))
    for k in kinds:
        row = f'  {k:17}'
        for g, ev in evals.items():
            ms = [e['test_masks'].get(f'{k}|all|model') for e in ev]
            base = [min(v for p in ('idw', 'linear') if (v := e['test_masks'].get(f'{k}|all|{p}')) is not None)
                    for e in ev if any(e['test_masks'].get(f'{k}|all|{p}') is not None for p in ('idw', 'linear'))]
            ms = [m for m in ms if m is not None]
            row += (f' | {np.mean(ms):.3f} ({np.mean(ms) / np.mean(base):.2f}x baseline)' if ms and base else
                    f' | {np.mean(ms):>22.3f}' if ms else f' | {"-":>22}')
        print(row)
    for a in ('coords_shuffle', 'coords_mean', 'time_shuffle', 'time_const'):
        print(f'  ablation {a:15}' + ''.join(
            f' | {np.mean([e["ablation_masked_mse"][a] / e["ablation_masked_mse"]["baseline"] - 1 for e in ev]):>+21.0%}'
            if ev else f' | {"-":>22}' for ev in evals.values()))
    print('  coord sim ~ closeness   ' + ''.join(
        f' | {np.mean([e["structure"]["coord_sim_vs_closeness_spearman"] for e in ev]):>22.3f}' if ev else f' | {"-":>22}'
        for ev in evals.values()))
    for g, ev in evals.items():
        sb = [e['structure'].get('spatial_bias_per_block') for e in ev if e['structure'].get('spatial_bias_per_block')]
        if sb:
            rho = np.mean([[d['closeness_spearman'] for d in s] for s in sb], 0)
            mag = np.mean([[d['mean_abs'] for d in s] for s in sb], 0)
            print(f'  {g} spatial bias per block (closeness rho / mean|b|): '
                  + ' '.join(f'{r:+.2f}/{m:.2f}' for r, m in zip(rho, mag)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', action='append', required=True, help='name=backbone1,backbone2,...')
    ap.add_argument('--ref', required=True)
    args = ap.parse_args()
    groups = {g.split('=')[0]: g.split('=')[1].split(',') for g in args.group}
    downstream(groups, args.ref)
    backbone(groups)


if __name__ == '__main__':
    main()
