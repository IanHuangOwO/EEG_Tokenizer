"""snapshot: one real trial per target, original vs reconstruction (band-filtered grid) and the Q-atom
gallery -> analysis/snapshot/. Targets: training_params.visualize_params.pretrain.targets (the same
{dataset, subject, trial} list the periodic training visualisation uses), or --datasets (trial 0 of the
first subject loaded). Only a few subjects per dataset are loaded (tools.analysis.cap_subjects)."""
import copy
import os
import random

from IO.dataset import build_dataset_from_config
from tools.analysis import cap_subjects, pick_trial
from tools.analysis.snapshot import build_pretrain_bundle
from tools.viz.snapshot import render_recon, render_atom_gallery

STAGES = frozenset({'pretrain'})


def run(ctx):
    ds_params = ctx.config['dataset_params']['pretrain']
    targets = ([{'dataset': d, 'subject': None, 'trial': 0} for d in ctx.args.datasets] if ctx.args.datasets else
               ctx.config['training_params'].get('visualize_params', {}).get('pretrain', {}).get('targets')
               or [{'dataset': next(iter(ds_params)), 'subject': None, 'trial': 0}])
    out_dir = os.path.join(ctx.out_dir, 'snapshot')
    rng, built = random.Random(1), {}
    for t in targets:
        name = t['dataset']
        if name not in built:
            cfg = copy.deepcopy(ctx.config)
            subjects = [str(t['subject'])] if t.get('subject') is not None else \
                cap_subjects(cfg, ds_params[name], 50, rng, min_subjects=1)
            cfg['dataset_params']['pretrain'] = {name: {**ds_params[name], 'subject_to_use': subjects}}
            built[name] = build_dataset_from_config(cfg, mode='pretrain', assemble_trials=False)
        ds = built[name]
        subject = t.get('subject') if t.get('subject') is not None else ds.base_dataset.subject_data[0].item()
        idx, subject = pick_trial(ds, subject, trial=t.get('trial'), dataset_name=name)
        bundle, metrics = build_pretrain_bundle(ctx.model, ds, idx, ctx.config, ctx.device, subject_id=subject)
        bundle.filename_tag = f'_{name}'
        render_recon(bundle, ctx.config, out_dir)
        render_atom_gallery(bundle, ctx.config, out_dir, cmap=ctx.cmap)
        print(f"  {name} subject {subject} trial {idx}: " + '  '.join(f"{k}={v:.4f}" for k, v in metrics.items()))
