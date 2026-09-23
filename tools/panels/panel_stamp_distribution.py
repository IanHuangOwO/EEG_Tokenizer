"""stamp_distribution panel: per-stamp a/b/amp/phase violin plots
(tools/viz/stamp_plots.py's plot_stamp_ab_violin, driven by tools/analysis/stamp_dist.py's
accumulate_stamp_ab) -- per DATASET (one dataset_params.pretrain entry at a time, same
split as the stamp_gallery/stamp_by_patch panels' --analysis loop, see
analysis_pretrain.py's `out = resolve_output_dir(cfg, 'analysis', ds_name, mode=mode)`),
one plot accumulated over a single subject's trials and one over every trial IN THAT
DATASET (not combined across every configured dataset -- see docs/agents/adding-a-tool.md,
a dataset's own recon/ dir is the established per-dataset output unit). Builds its own
PretrainDataset internally, one per dataset (same self-contained pattern as
panel_profile.py) -- no ctx.bundle involved, this needs many trials at once, not one.

CLI (via ctx.args): --dataset (restrict to one dataset_params.pretrain entry; default:
every entry), --subject (which subject id gets its own plot per dataset; default: the
first subject present in that dataset), --sd-max-stamps (violin count cap, default 30),
--sd-max-trials (cap trials accumulated per dataset, default: every trial)."""
import copy
import os

from IO.dataset import build_dataset_from_config
from tools.analysis import resolve_output_dir
from tools.analysis.stamp_dist import accumulate_stamp_ab
from tools.viz.stamp_plots import plot_stamp_ab_violin

STAGES = frozenset({'pretrain'})
NEEDS_CHECKPOINT = True
NEEDS_DATASET = False


def run(ctx):
    model = ctx.model
    config = ctx.config
    max_trials = ctx.args.sd_max_trials
    max_stamps = ctx.args.sd_max_stamps
    subject_id_arg = ctx.args.subject

    ds_params = config.get('dataset_params', {}).get('pretrain', {})
    wanted = [ctx.args.dataset] if ctx.args.dataset else list(ds_params)

    for ds_name in wanted:
        single_cfg = copy.deepcopy(config)
        single_cfg['dataset_params']['pretrain'] = {ds_name: ds_params[ds_name]}
        dataset = build_dataset_from_config(single_cfg, mode='pretrain')
        base = dataset.base_dataset

        viz_dir = os.path.join(resolve_output_dir(config, 'analysis', ds_name, mode='pretrain'), 'recon')
        os.makedirs(viz_dir, exist_ok=True)

        all_trials = list(range(len(base)))
        if max_trials:
            all_trials = all_trials[:max_trials]

        present_subjects = {int(base.subject_data[i]) for i in all_trials}
        subject_id = subject_id_arg if subject_id_arg in present_subjects \
            else int(base.subject_data[all_trials[0]])
        subj_trials = [i for i in all_trials if int(base.subject_data[i]) == subject_id]

        subj_ab = accumulate_stamp_ab(model, dataset, subj_trials, ctx.device, max_stamps=max_stamps)
        subj_path = os.path.join(viz_dir, f'sub{subject_id}_stamp_ab_violin.png')
        plot_stamp_ab_violin(subj_path, subj_ab,
                              title=f'Stamp a/b/amp/phase -- {ds_name} subject {subject_id}',
                              n_routed=model.n_routed_stamps)
        print(f"  [panel] -> {subj_path}")

        ds_ab = accumulate_stamp_ab(model, dataset, all_trials, ctx.device, max_stamps=max_stamps)
        ds_path = os.path.join(viz_dir, f'{ds_name}_stamp_ab_violin.png')
        plot_stamp_ab_violin(ds_path, ds_ab, title=f'Stamp a/b/amp/phase -- {ds_name} (all subjects)',
                              n_routed=model.n_routed_stamps)
        print(f"  [panel] -> {ds_path}")
