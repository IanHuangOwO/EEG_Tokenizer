"""stamp_distribution: per-stamp a / b / amplitude / phase violins over real trials, one plot per
dataset (tools/analysis/stamp_dist.py) -> analysis/stamp_distribution/<dataset>.png.
Options: --datasets (default: every pretrain dataset), --max-trials (per dataset)."""
import copy
import os
import random

from IO.dataset import build_dataset_from_config
from tools.analysis import cap_subjects
from tools.analysis.stamp_dist import accumulate_stamp_ab
from tools.viz.stamp_plots import plot_stamp_ab_violin

STAGES = frozenset({'pretrain'})


def run(ctx):
    ds_params = ctx.config['dataset_params']['pretrain']
    out_dir = os.path.join(ctx.out_dir, 'stamp_distribution')
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(0)
    for name in ctx.args.datasets or list(ds_params):
        cfg = copy.deepcopy(ctx.config)
        subjects = cap_subjects(cfg, ds_params[name], ctx.args.max_trials, rng, min_subjects=5)
        cfg['dataset_params']['pretrain'] = {name: {**ds_params[name], 'subject_to_use': subjects}}
        ds = build_dataset_from_config(cfg, mode='pretrain')
        trials = list(range(min(ctx.args.max_trials, len(ds.base_dataset))))
        ab = accumulate_stamp_ab(ctx.model, ds, trials, ctx.device, max_stamps=ctx.model.n_stamps)
        out = os.path.join(out_dir, f'{name}.png')
        plot_stamp_ab_violin(out, ab, title=f'Stamp a/b/amp/phase: {name} ({len(subjects)} subjects, '
                             f'{len(trials)} windows)')
        print(f"  -> {out}")
