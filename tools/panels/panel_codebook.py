"""codebook: cross-dataset Q-atom-usage diagnostics (model/base_codebook_checker.py via the plugin's
codebook_checker_cls) -> analysis/. One dataset per pretrain entry (usage stays attributable to its
source), real trials with their labels, each capped at --max-trials trials (tools.analysis.cap_subjects).
Options: --datasets, --max-trials."""
import copy
import gc
import random

from IO.dataset import build_dataset_from_config
from model.factory import MODEL_REGISTRY
from tools.analysis import cap_subjects

STAGES = frozenset({'pretrain'})


def run(ctx):
    ds_params = ctx.config['dataset_params']['pretrain']
    max_trials, rng = ctx.args.max_trials, random.Random(0)
    by_name = {}
    for name in ctx.args.datasets or list(ds_params):
        cfg = copy.deepcopy(ctx.config)
        subjects = cap_subjects(cfg, ds_params[name], max_trials, rng)
        cfg['dataset_params']['pretrain'] = {name: {**ds_params[name], 'subject_to_use': subjects}}
        by_name[name] = build_dataset_from_config(cfg, mode='pretrain', assemble_trials=False)
        print(f"  {name}: {len(subjects)} subject(s)")
    checker = MODEL_REGISTRY['Qtome'].codebook_checker_cls()
    checker.check_codebook(ctx.config, ctx.out_dir, ctx.model, by_name, max_trials_per_dataset=max_trials)
    del by_name
    gc.collect()
