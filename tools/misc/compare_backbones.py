"""
Compare groups of backbones (each group = the same recipe over pretrain seeds) on
tools/misc/backbone_eval.py's held-out-window summaries: test-mask masked MSE (model, and
model / best model-free baseline), ablation changes, and embedding structure, averaged over
each group's seeds.

Downstream finetune comparison (tail balanced accuracy, paired test, imputation gain) now
lives in tools/misc/summarize_runs.py -- it reads the same group_eval.json files without
this script's fixed CELLS/HEADS lists, so it covers a grid or a head comparison too, not
just the backbone x dataset_mode table this file used to print.

Usage: python -m tools.misc.compare_backbones \
    --group base=mesae_tiny_static16_base_s1,mesae_tiny_static16_base_s2 \
    --group A=mesae_tiny_static16_groupA_s1,mesae_tiny_static16_groupA_s2 \
    --group B=mesae_tiny_static16_groupB_s1,mesae_tiny_static16_groupB_s2
"""
import argparse
import json
import os

import numpy as np


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
    args = ap.parse_args()
    groups = {g.split('=')[0]: g.split('=')[1].split(',') for g in args.group}
    backbone(groups)


if __name__ == '__main__':
    main()
