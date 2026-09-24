"""
Post-training checker for the PRETRAIN stage only: per-subject topo/PSD/attention
snapshot (MeSAE, resolved from the model instance) via BaseEpochChecker.check_pretrain
(model/base_checker.py), plus cross-dataset codebook/vocab diagnostics
(model/base_codebook_checker.py). Finetune-stage analysis lives in analysis_finetune.py.

Config resolution: configs/analysis_pretrain.template.json (or --config) is a small overlay — checkpoint,
dataset_params.pretrain (one dataset entry, subject_to_use = subjects to visualize;
shared by Tokenizer and Pretrain-stage checkpoints, see CLAUDE.md), check.plot_* toggles.
It's deep-merged onto the full run config, taken from the checkpoint's own
output/<output_path>/artifacts/config.json snapshot unless overlay['base_config'] or
--base-config points elsewhere. Output goes to
output/<output_path>/analysis/<dataset_name>/recon/ (separate from training's own
output/<output_path>/visualization/).
"""

import os
import json


def _cap_subjects_by_trial_budget(cfg, ds_args, max_trials, rng, min_subjects=20):
    """Shuffle ds_args's resolved subject list and keep only as many subjects as needed
    to cover ~max_trials real trials, peeking each subject's cheap 'labels' array
    (shape (N,), a few hundred int64s -- negligible) to learn its trial count WITHOUT
    touching that subject's 'data' array (the actual multi-hundred-MB payload). Without
    this, subject_to_use=["all"] on a big dataset (e.g. PhysionetMI's 109 subjects) loads
    every real trial from every subject into RAM via build_dataset_from_config, even
    though check_codebook only ever samples up to max_trials_per_dataset of them right
    afterward -- the rest sat in memory for nothing. Real instance: this OOM'd a run at
    ~55GB RSS loading all-subjects for every dataset at once (see
    docs/model-analysis-checklist.md). Returns the capped subject id list; a dataset
    whose FIRST subject alone already exceeds max_trials still returns just that one
    subject (never zero), so a small max_trials never drops a dataset entirely.

    min_subjects: keeps adding subjects past max_trials until at least this many are
    picked (or the dataset runs out), even if that overshoots the trial budget --
    trial count alone gives a stable per-unit usage-rate ESTIMATE (variance shrinks with
    sqrt(n) regardless of subject count), but a handful of subjects each contributing
    hundreds of correlated trials (e.g. PhysionetMI: 9 subjects hit 3000 trials on their
    own) under-covers SUBJECT diversity -- inter-subject variability is usually the
    dominant source of variance in EEG, not trial-to-trial noise within one subject.
    A small dataset with fewer than min_subjects subjects total just returns all of
    them, same as before."""
    import numpy as np
    dataset_path = ds_args['dataset_path']
    with open(os.path.join(dataset_path, 'metadata.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)
    all_subjects = sorted(meta.get('data_structure', {}).keys())
    requested = ds_args.get('subject_to_use', ['all'])
    if requested in (['all'], 'all'):
        subjects = list(all_subjects)
    else:
        req = {str(s) for s in requested}
        subjects = [s for s in all_subjects if s in req]
    rng.shuffle(subjects)

    from IO.preprocessing import cache_suffix
    pp = cfg['preprocess_params']
    suffix = cache_suffix(pp['sample_freq'], pp['bandpass_filter'],
                           pp.get('pre_event_seconds', 0.0), pp.get('post_event_seconds', 0.0))

    picked, total = [], 0
    for sid in subjects:
        cache_path = os.path.join(dataset_path, 'cache', f"{sid}_{suffix}.npz")
        if not os.path.exists(cache_path):
            continue
        n = len(np.load(cache_path)['labels'])
        picked.append(sid)
        total += n
        if total >= max_trials and len(picked) >= min_subjects:
            break
    return picked


if __name__ == '__main__':
    import argparse
    import copy
    import json
    import torch
    from IO.dataset import build_dataset_from_config
    from tools.analysis import (
        _deep_merge, load_model,
        select_subject_dataset, filter_config_to_subject, pick_trial, resolve_output_dir,
    )
    from tools.analysis.snapshot import build_pretrain_bundle
    from tools.panels import PanelContext, run_panels

    parser = argparse.ArgumentParser(description='Post-training EEG checker (MeSAE)')
    parser.add_argument('--config',      default='configs/analysis_pretrain.template.json')
    parser.add_argument('--base-config', default=None, dest='base_config')
    parser.add_argument('--checkpoint',  default=None)
    parser.add_argument('--analysis',    default=None, choices=['snapshot', 'codebook', 'both'],
                         help='Overrides check.analysis in --config. snapshot: existing per-trial '
                              'topo/PSD/attn checker. codebook: cross-dataset codebook/vocab '
                              'diagnostics (model/base_codebook_checker.py). both: run both.')
    parser.add_argument('--subject',     type=int, default=None)
    parser.add_argument('--trial',       type=int, default=None)
    parser.add_argument('--dataset',     type=str, default=None)
    parser.add_argument('--recon_cmap',  type=str, default=None)
    parser.add_argument('--panel',       action='append', default=[],
                         help='Run one or more panels (repeatable) that do not need a '
                              'dataset/bundle (NEEDS_DATASET=False), instead of the legacy '
                              '--analysis path. Panels that need a bundle (recon_signal, '
                              'stamp_by_patch, stamp_gallery) are only reachable via '
                              '--analysis today. See tools/panels/.')
    parser.add_argument('--train',       action='store_true',
                         help='(panel_profile only) profile in train mode (eigh skipped)')
    parser.add_argument('--se-datasets', action='append', default=[], dest='se_datasets',
                         help='(panel_select_eval_subsets only) dataset name(s) from '
                              'tools.analysis.select_eval_subsets.DATASETS (repeatable; '
                              'default: every key)')
    parser.add_argument('--se-run-config', default=None, dest='se_run_config',
                         help='(panel_select_eval_subsets only) path to a config.json with '
                              'dataset_params.pretrain subject pools (default: that module\'s RUN)')
    parser.add_argument('--se-out-dir', default=None, dest='se_out_dir',
                         help='(panel_select_eval_subsets only) output dir '
                              '(default: configs/finetune_eval_splits)')
    parser.add_argument('--sd-max-stamps', type=int, default=30, dest='sd_max_stamps',
                         help='(panel_stamp_distribution only) cap on stamps shown per '
                              'violin plot, ranked by firing count')
    parser.add_argument('--sd-max-trials', type=int, default=None, dest='sd_max_trials',
                         help='(panel_stamp_distribution only) cap on trials accumulated '
                              'per dataset (default: every trial)')
    args = parser.parse_args()

    if args.panel:
        from tools.panels import build_panel_context, run_panels

        def _resolve_base_path(args, overlay, checkpoint):
            base_path = args.base_config or overlay.pop('base_config', None)
            if not base_path:
                model_dir = os.path.dirname(os.path.dirname(checkpoint))
                base_path = os.path.join(model_dir, 'artifacts', 'config.json')
            return base_path

        ctx = build_panel_context(args, args.panel, 'pretrain', _resolve_base_path)
        run_panels(args.panel, 'pretrain', ctx)
        raise SystemExit(0)

    with open(args.config, 'r') as f:
        overlay = json.load(f)

    checkpoint = args.checkpoint or overlay.get('checkpoint', '')
    mode       = 'pretrain'
    # dataset_params only has 'pretrain'/'finetune' — the Tokenizer stage shares the
    # Pretrain stage's dataset entries (same raw data, no masking), see CLAUDE.md.
    data_mode  = 'pretrain'

    base_path = args.base_config or overlay.pop('base_config', None)
    if not base_path:
        model_dir = os.path.dirname(os.path.dirname(checkpoint))
        base_path = os.path.join(model_dir, 'artifacts', 'config.json')
    with open(base_path, 'r') as f:
        base = json.load(f)
    cfg = _deep_merge(base, overlay)
    # dataset_params picks the (possibly unseen) dataset to check — an overlay section
    # replaces the base's wholesale rather than deep-merging into it, so a target dataset
    # absent from the base config (e.g. checking generalization to a new dataset) doesn't
    # end up sitting alongside the base's own datasets with an ambiguous "first" pick.
    for m, dsp in overlay.get('dataset_params', {}).items():
        cfg['dataset_params'][m] = dsp

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl    = load_model(cfg, checkpoint, device, mode=mode)

    check_cfg = cfg.get('check', {})
    cmap = args.recon_cmap or check_cfg.get('cmap', 'YlOrRd')
    analysis = args.analysis or check_cfg.get('analysis', 'snapshot')

    ds_params    = cfg['dataset_params'][data_mode]

    if analysis in ('codebook', 'both'):
        from model.factory import MODEL_REGISTRY

        probe = mdl.backbone if hasattr(mdl, 'backbone') else mdl
        model_type = 'MeSAE'
        plugin = MODEL_REGISTRY[model_type]
        if plugin.codebook_checker_cls is None:
            raise NotImplementedError(f"{model_type} has no codebook_checker_cls yet (see model/base_plugin.py)")
        checker = plugin.codebook_checker_cls()

        codebook_cfg = check_cfg.get('codebook', {})
        max_trials   = codebook_cfg.get('max_trials_per_dataset', 200)

        # codebook diagnostics read the backbone's own patchified forward (extract_usage
        # calls model(x_in, c_in, time_idx=, valid_channels=)) regardless of --mode, so
        # always use the 'pretrain' dataset shape (PretrainDataset's 7-tuple) even
        # when checking a finetune checkpoint -- FinetuneDataset's 5-tuple (raw [C,T], no
        # patch/mask) doesn't match what BaseCodebookChecker._trial_tensors unpacks.
        codebook_ds_params = cfg['dataset_params']['pretrain']

        # one dataset per entry (not the full multi-dataset concatenation) so usage stats
        # stay attributable to a single source dataset, see viz/codebook.py.
        # assemble_trials=False: codebook's by-target plots need each patch's real trial
        # label; data_mode='pretrain' would otherwise window-assemble and every label comes
        # back torch.zeros(...) (see IO/preprocessing.py window_continuous_signal), collapsing every
        # target to a single dummy class.
        import random as _random
        cap_rng = _random.Random(0)
        datasets_by_name = {}
        for ds_name, ds_args in codebook_ds_params.items():
            capped_subjects = _cap_subjects_by_trial_budget(cfg, ds_args, max_trials, cap_rng)
            single_cfg = copy.deepcopy(cfg)
            single_cfg['dataset_params']['pretrain'] = {ds_name: {**ds_args, 'subject_to_use': capped_subjects}}
            datasets_by_name[ds_name] = build_dataset_from_config(
                single_cfg, mode='pretrain', assemble_trials=False)
            print(f"  [codebook] {ds_name}: loading {len(capped_subjects)} subject(s) "
                  f"(capped to a ~{max_trials}-trial budget, from subject_to_use={ds_args.get('subject_to_use')})")

        out = resolve_output_dir(cfg, 'analysis', mode=mode)
        checker.check_codebook(cfg, out, probe, datasets_by_name, max_trials_per_dataset=max_trials)
        print(f"[check] codebook analysis done: datasets={list(datasets_by_name)}  |  out={out}")

        # analysis='both' builds a SECOND full dataset below (the snapshot branch's own
        # combined-multi-dataset `ds`, covering the exact same dataset_params) while
        # datasets_by_name (12 separate full datasets, all subjects) is still a live local
        # var -- Python won't free it just because check_codebook is done with it, only
        # once nothing references it. Drop it explicitly so the snapshot phase's own load
        # isn't peaking on top of the whole codebook phase's data, not just competing with it.
        del datasets_by_name
        import gc
        gc.collect()

    if analysis in ('snapshot', 'both'):
        if check_cfg.get('plot_topo_psd', True):
            # Runs ONCE here, not inside the per-target loop below -- unlike recon_signal/
            # stamp_by_patch/stamp_gallery (one real trial's bundle each), stamp_distribution
            # builds its own dataset and loops every dataset_params.pretrain entry itself
            # (tools/panels/panel_stamp_distribution.py), writing into the same per-dataset
            # output/<model>/analysis/<Dataset>/recon/ dirs the per-target loop below also
            # writes into -- calling it per-target would just repeat the same full-dataset
            # scan once per target trial for no reason.
            panel_ctx = PanelContext(config=cfg, output_dir=resolve_output_dir(cfg, 'analysis', mode=mode),
                                      device=device, args=args, model=mdl, cmap=cmap)
            run_panels(['stamp_distribution'], 'pretrain', panel_ctx)

        if cfg.get('training_params', {}).get('visualize_params', {}).get(data_mode, {}).get('targets'):
            # Same visualize_params.<mode>.targets key/shape as config.json's own periodic-viz config
            # (see train_pretrain.py's viz_targets) — a list of {dataset, subject, trial} triples
            # picking exactly which snapshots to render, instead of dataset_params.<mode>'s
            # one-subject-per-dataset-entry mechanism below. dataset_params.<mode> still governs
            # what data gets loaded (all datasets/subjects it lists, combined into one dataset);
            # targets only selects which trials within that get plotted.
            # assemble_trials=False: one snapshot = one REAL trial, so patch count matches the
            # trial's own length (e.g. a 260-pt Inria P300 epoch -> 2 patches) instead of an
            # 800-pt assembled window that spans ~3 concatenated epochs, which makes the
            # stamp_by_patch panel's "patch position within trial" axis meaningless for
            # short-trial datasets. Long-trial datasets (~800pt) are unaffected. Same call the
            # codebook path already uses. Note: trial_idx now indexes real trials, not windows.
            # Restrict to just the datasets `targets` actually reference (with subject_to_use
            # still whatever dataset_params.<mode> already says, e.g. "all") -- every OTHER
            # dataset in dataset_params.<mode> would otherwise get loaded in full here too,
            # just to pick 1-2 trials out of the handful this config's targets actually name
            # (see check_model_v5_fullall_log's OOM: the codebook phase's per-dataset,
            # trial-budget-capped load below fixes that half; this branch was the other half).
            targets_pre = cfg['training_params']['visualize_params'][data_mode]['targets']
            wanted_datasets = {t.get('dataset') for t in targets_pre if t.get('dataset')}
            filtered_dsp = ({k: v for k, v in ds_params.items() if k in wanted_datasets}
                            if wanted_datasets else ds_params)
            # This branch only ever plots ONE trial per target (pick_trial below), so even
            # the small codebook budget above is overkill here -- a big dataset's
            # subject_to_use=["all"] (e.g. PhysionetMI's 109) still loads every real trial from
            # every subject otherwise, just to render 1-2 snapshots (the exact same shape of
            # bug the codebook loop's _cap_subjects_by_trial_budget already fixes, so reuse
            # it -- small budget, small min_subjects since diversity doesn't matter for a
            # single representative pick).
            import random as _random
            snap_rng = _random.Random(1)
            filtered_dsp = {
                ds_name: {**ds_args, 'subject_to_use': _cap_subjects_by_trial_budget(
                    cfg, ds_args, max_trials=50, rng=snap_rng, min_subjects=1)}
                for ds_name, ds_args in filtered_dsp.items()
            }
            snapshot_cfg = copy.deepcopy(cfg)
            snapshot_cfg['dataset_params'][data_mode] = filtered_dsp
            ds = build_dataset_from_config(snapshot_cfg, mode=data_mode, assemble_trials=False)

            def _first_subject(dataset_name=None):
                sub_data = ds.base_dataset.subject_data
                if dataset_name is not None:
                    names = ds.base_dataset.dataset_names
                    idx = next((i for i, n in enumerate(names) if n == dataset_name), None)
                    if idx is not None:
                        return sub_data[idx].item()
                return sub_data[0].item()

            targets = cfg['training_params']['visualize_params'][data_mode]['targets']
            for t in targets:
                t_dataset = t.get('dataset')
                subj = t.get('subject') if t.get('subject') is not None else _first_subject(t_dataset)
                t_idx, subject_id = pick_trial(ds, subj, trial=t.get('trial'), dataset_name=t_dataset)
                out = resolve_output_dir(cfg, 'analysis', t_dataset or 'multi', mode=mode)
                try:
                    bundle, metrics = build_pretrain_bundle(mdl, ds, t_idx, cfg, device, subject_id=subject_id)
                    panels = []
                    if check_cfg.get('plot_recon', True):
                        panels.append('recon_signal')
                    if check_cfg.get('plot_topo_psd', True):
                        panels += ['stamp_by_patch', 'stamp_gallery']
                    panel_ctx = PanelContext(config=cfg, output_dir=out, device=device, args=args,
                                              model=mdl, dataset=ds, cmap=cmap, bundle=bundle)
                    run_panels(panels, 'pretrain', panel_ctx)
                    metrics_str = '  '.join(f"{k}={v:.4f}" for k, v in metrics.items())
                    print(f"[check] done: dataset={t_dataset} subject={subject_id} trial_idx={t_idx}  |  {metrics_str}")
                except Exception as e:
                    print(f"[check] FAILED: dataset={t_dataset} subject={subject_id} trial_idx={t_idx}  |  {e}")
                    continue

        else:
            dataset_name = args.dataset or next(iter(ds_params))
            ds_cfg       = ds_params[dataset_name]
            subjects     = [args.subject] if args.subject is not None else ds_cfg.get('subject_to_use', [])

            for subject in subjects:
                ds_name, subject = select_subject_dataset(cfg, subject, dataset_name=dataset_name, mode=data_mode)
                filtered  = filter_config_to_subject(cfg, ds_name, subject, mode=data_mode)
                # assemble_trials=False -> one snapshot = one real trial, patch count matches
                # the trial's own length (see the targets branch above for the full rationale).
                ds        = build_dataset_from_config(filtered, mode=data_mode, assemble_trials=False)
                trial_cfg = args.trial if args.trial is not None else ds_cfg.get('trial_to_use')
                t_idx, subject_id = pick_trial(ds, subject, trial_cfg, dataset_name=ds_name)
                out = resolve_output_dir(filtered, 'analysis', ds_name, mode=mode)
                try:
                    bundle, metrics = build_pretrain_bundle(mdl, ds, t_idx, filtered, device, subject_id=subject_id)
                    panels = []
                    if check_cfg.get('plot_recon', True):
                        panels.append('recon_signal')
                    if check_cfg.get('plot_topo_psd', True):
                        panels += ['stamp_by_patch', 'stamp_gallery']
                    panel_ctx = PanelContext(config=filtered, output_dir=out, device=device, args=args,
                                              model=mdl, dataset=ds, cmap=cmap, bundle=bundle)
                    run_panels(panels, 'pretrain', panel_ctx)
                    metrics_str = '  '.join(f"{k}={v:.4f}" for k, v in metrics.items())
                    print(f"[check] done: dataset={ds_name} subject={subject_id} trial_idx={t_idx}  |  {metrics_str}")
                except Exception as e:
                    print(f"[check] FAILED: dataset={ds_name} subject={subject_id} trial_idx={t_idx}  |  {e}")
                    continue
