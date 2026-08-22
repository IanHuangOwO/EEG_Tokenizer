"""
Post-training checker: per-subject topo/PSD/attention snapshot (MeFSQ or MeSAE, resolved
from the model instance) via BaseEpochChecker.check_pretrain/check_finetune
(model/base_checker.py).

Config resolution: config/analysis.json (or --config) is a small overlay — checkpoint,
mode, dataset_params.pretrain (one dataset entry, subject_to_use = subjects to visualize;
shared by Tokenizer and Pretrain-stage checkpoints, see CLAUDE.md), check.plot_* toggles.
It's deep-merged onto the full run config, taken from the checkpoint's own
output/<model_name>/artifacts/config.json snapshot unless overlay['base_config'] or
--base-config points elsewhere. Output goes to
output/<model_name>/analysis/<dataset_name>/recon/ (separate from training's own
output/<model_name>/visualization/).
"""

import os
import json

from model.factory import MODEL_REGISTRY


def _load_target_names(dataset_path, num_classes):
    """data_metadata.targets["<idx>"].label, e.g. BCICIV2a's {"0": {"label": "Left hand"}, ...}
    — falls back to "class<idx>" for any index missing from metadata (or if metadata has no
    targets section at all, e.g. a dataset that hasn't been annotated with class names)."""
    try:
        with open(os.path.join(dataset_path, 'metadata.json'), 'r', encoding='utf-8') as f:
            meta = json.load(f)
        targets = meta.get('data_metadata', {}).get('targets', {})
    except Exception:
        targets = {}
    return [targets.get(str(i), {}).get('label', f'class{i}') for i in range(num_classes)]


def _safe_name(s):
    """Filename-safe version of a target label, e.g. 'Left hand' -> 'Left_hand'."""
    return ''.join(c if c.isalnum() else '_' for c in s).strip('_') or 'unnamed'


def _predict_all(model, dataset, patch_len, device):
    """Runs the finetune model over every trial in `dataset` (in index order, no shuffle)
    and returns (preds, labels) numpy arrays aligned to dataset indices — used to find one
    correctly- and one incorrectly-classified trial per target class."""
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from train_finetune import FinetuneCollate

    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=FinetuneCollate(patch_len))
    was_training = model.training
    model.eval()
    all_preds, all_labels = [], []
    try:
        with torch.no_grad():
            for batch in loader:
                x, coords, time_idx, labels, valid_channels, pad_mask = batch
                logits = model(
                    x.to(device), coords.to(device), time_idx=time_idx.to(device),
                    valid_channels=valid_channels.to(device), pad_mask=pad_mask.to(device),
                )[0]
                all_preds.append(logits.argmax(dim=-1).cpu())
                all_labels.append(labels)
    finally:
        model.train(was_training)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def run(config, output_dir, model, dataset, trial_idx, mode='pretrain', subject_id=None,
        epoch=None, cmap='YlOrRd', plot_recon=True, plot_topo_psd=True, plot_attn_topo=True,
        tag=''):
    # finetune mode's `model` is the Finetune wrapper (model.backbone/model.head) — the
    # attribute that tells MeFSQ from MeSAE apart lives on the backbone submodule then,
    # not on the wrapper itself.
    probe = model.backbone if hasattr(model, 'backbone') else model
    model_type = 'MeFSQ' if hasattr(probe, 'n_routed_experts') else 'MeSAE'
    plugin  = MODEL_REGISTRY[model_type]
    checker = plugin.checker_cls()
    trainer = plugin.trainer_cls()
    if mode == 'finetune':
        return checker.check_finetune(
            config, output_dir, model, dataset, trial_idx,
            subject_id=subject_id, epoch=epoch, cmap=cmap,
            plot_recon=plot_recon, plot_topo_psd=plot_topo_psd, plot_attn_topo=plot_attn_topo,
            trainer=trainer, tag=tag,
        )
    return checker.check_pretrain(
        config, output_dir, model, dataset, trial_idx,
        subject_id=subject_id, epoch=epoch, cmap=cmap,
        plot_recon=plot_recon, plot_topo_psd=plot_topo_psd, plot_attn_topo=plot_attn_topo,
        trainer=trainer,
    )


if __name__ == '__main__':
    import argparse
    import copy
    import json
    import torch
    from IO.dataset import build_dataset_from_config
    from viz import (
        _deep_merge, load_model,
        select_subject_dataset, filter_config_to_subject, pick_trial, resolve_output_dir,
    )

    parser = argparse.ArgumentParser(description='Post-training EEG checker (MeFSQ or MeSAE)')
    parser.add_argument('--config',      default='config/analysis.json')
    parser.add_argument('--base-config', default=None, dest='base_config')
    parser.add_argument('--checkpoint',  default=None)
    parser.add_argument('--mode',        default=None, choices=['tokenizer', 'pretrain', 'finetune'])
    parser.add_argument('--analysis',    default=None, choices=['snapshot', 'codebook', 'both'],
                         help='Overrides check.analysis in --config. snapshot: existing per-trial '
                              'topo/PSD/attn checker. codebook: cross-dataset codebook/vocab '
                              'diagnostics (model/base_codebook_checker.py). both: run both.')
    parser.add_argument('--subject',     type=int, default=None)
    parser.add_argument('--trial',       type=int, default=None)
    parser.add_argument('--dataset',     type=str, default=None)
    parser.add_argument('--recon_cmap',  type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        overlay = json.load(f)

    checkpoint = args.checkpoint or overlay.get('checkpoint', '')
    mode       = args.mode or overlay.get('mode', 'pretrain')
    # dataset_params only has 'pretrain'/'finetune' — the Tokenizer stage shares the
    # Pretrain stage's dataset entries (same raw data, no masking), see CLAUDE.md.
    data_mode  = 'finetune' if mode == 'finetune' else 'pretrain'

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
        model_type = 'MeFSQ' if hasattr(probe, 'n_routed_experts') else 'MeSAE'
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
        datasets_by_name = {}
        for ds_name, ds_args in codebook_ds_params.items():
            single_cfg = copy.deepcopy(cfg)
            single_cfg['dataset_params']['pretrain'] = {ds_name: ds_args}
            datasets_by_name[ds_name] = build_dataset_from_config(
                single_cfg, mode='pretrain', assemble_trials=False)

        out = resolve_output_dir(cfg, 'analysis', mode=mode)
        checker.check_codebook(cfg, out, probe, datasets_by_name, max_trials_per_dataset=max_trials)
        print(f"[check] codebook analysis done: datasets={list(datasets_by_name)}  |  out={out}")

    if analysis in ('snapshot', 'both'):
        if mode == 'finetune':
            # Per-target correct/wrong snapshot pairs, not a single per-subject trial pick:
            # for each of the num_classes targets, find one trial the model got right and one
            # it got wrong, and render the full 3-panel snapshot (recon_signal, topo_psd_filter,
            # attn_topo) for each — num_classes * 2 * 3 files total, searched across every
            # subject in dataset_params.finetune[dataset_name].subject_to_use (not one subject
            # at a time), since a single subject isn't guaranteed to contain both a correct and
            # a wrong example of every class.
            dataset_name = args.dataset or next(iter(ds_params))
            ds_cfg       = ds_params[dataset_name]

            filtered = copy.deepcopy(cfg)
            filtered['dataset_params'][data_mode] = {dataset_name: ds_cfg}
            # subject_to_use=["all"] needs the same resolution train_finetune.py's
            # build_subject_split_datasets does — build_dataset_from_config takes it literally
            # and fails ("Subject all not found") since EEGDataset expects real subject ids.
            from train_finetune import _resolve_all_subjects, _resolve_requested_subjects
            all_subjects = _resolve_all_subjects(ds_cfg['dataset_path'])
            filtered['dataset_params'][data_mode][dataset_name]['subject_to_use'] = \
                _resolve_requested_subjects(ds_cfg, all_subjects)
            ds = build_dataset_from_config(filtered, mode=data_mode)

            patch_len = filtered.get('preprocess_params', {}).get('patch_length', 100)
            preds, labels = _predict_all(mdl, ds, patch_len, device)
            num_classes  = int(labels.max()) + 1
            target_names = _load_target_names(ds_cfg['dataset_path'], num_classes)

            out = resolve_output_dir(filtered, 'analysis', dataset_name, mode=mode)
            for cls_idx in range(num_classes):
                name = target_names[cls_idx]
                safe = _safe_name(name)
                cls_mask = labels == cls_idx
                correct_idxs = (cls_mask & (preds == cls_idx)).nonzero()[0]
                wrong_idxs   = (cls_mask & (preds != cls_idx)).nonzero()[0]
                for status, idxs in (('correct', correct_idxs), ('wrong', wrong_idxs)):
                    if len(idxs) == 0:
                        print(f"[check] target{cls_idx}_{safe}: no {status} example found in {dataset_name}, skipping")
                        continue
                    t_idx = int(idxs[0])
                    subject_id = int(ds.base_dataset.subject_data[t_idx].item())
                    tag = f'_target{cls_idx}_{safe}_{status}'
                    metrics = run(
                        filtered, out, mdl, ds, t_idx, mode=mode, subject_id=subject_id, cmap=cmap,
                        plot_recon=check_cfg.get('plot_recon', True),
                        plot_topo_psd=check_cfg.get('plot_topo_psd', True),
                        plot_attn_topo=check_cfg.get('plot_attn_topo', True),
                        tag=tag,
                    )
                    metrics_str = '  '.join(f"{k}={v:.4f}" for k, v in metrics.items())
                    print(f"[check] done: target={cls_idx}({name}) status={status} subject={subject_id} "
                          f"trial_idx={t_idx}  |  {metrics_str}")

        elif cfg.get('training_params', {}).get('visualize', {}).get(data_mode, {}).get('targets'):
            # Same visualize.<mode>.targets key/shape as config.json's own periodic-viz config
            # (see train_pretrain.py's viz_targets) — a list of {dataset, subject, trial} triples
            # picking exactly which snapshots to render, instead of dataset_params.<mode>'s
            # one-subject-per-dataset-entry mechanism below. dataset_params.<mode> still governs
            # what data gets loaded (all datasets/subjects it lists, combined into one dataset,
            # same as training does); targets only selects which trials within that get plotted.
            ds = build_dataset_from_config(cfg, mode=data_mode)

            def _first_subject(dataset_name=None):
                sub_data = ds.base_dataset.subject_data
                if dataset_name is not None:
                    names = ds.base_dataset.dataset_names
                    idx = next((i for i, n in enumerate(names) if n == dataset_name), None)
                    if idx is not None:
                        return sub_data[idx].item()
                return sub_data[0].item()

            targets = cfg['training_params']['visualize'][data_mode]['targets']
            for t in targets:
                t_dataset = t.get('dataset')
                subj = t.get('subject') if t.get('subject') is not None else _first_subject(t_dataset)
                t_idx, subject_id = pick_trial(ds, subj, trial=t.get('trial'), dataset_name=t_dataset)
                out = resolve_output_dir(cfg, 'analysis', t_dataset or 'multi', mode=mode)
                metrics = run(
                    cfg, out, mdl, ds, t_idx, mode=mode, subject_id=subject_id, cmap=cmap,
                    plot_recon=check_cfg.get('plot_recon', True),
                    plot_topo_psd=check_cfg.get('plot_topo_psd', True),
                    plot_attn_topo=check_cfg.get('plot_attn_topo', True),
                )
                metrics_str = '  '.join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[check] done: dataset={t_dataset} subject={subject_id} trial_idx={t_idx}  |  {metrics_str}")

        else:
            dataset_name = args.dataset or next(iter(ds_params))
            ds_cfg       = ds_params[dataset_name]
            subjects     = [args.subject] if args.subject is not None else ds_cfg.get('subject_to_use', [])

            for subject in subjects:
                ds_name, subject = select_subject_dataset(cfg, subject, dataset_name=dataset_name, mode=data_mode)
                filtered  = filter_config_to_subject(cfg, ds_name, subject, mode=data_mode)
                ds        = build_dataset_from_config(filtered, mode=data_mode)
                trial_cfg = args.trial if args.trial is not None else ds_cfg.get('trial_to_use')
                t_idx, subject_id = pick_trial(ds, subject, trial_cfg, dataset_name=ds_name)
                out = resolve_output_dir(filtered, 'analysis', ds_name, mode=mode)
                metrics = run(
                    filtered, out, mdl, ds, t_idx, mode=mode, subject_id=subject_id, cmap=cmap,
                    plot_recon=check_cfg.get('plot_recon', True),
                    plot_topo_psd=check_cfg.get('plot_topo_psd', True),
                    plot_attn_topo=check_cfg.get('plot_attn_topo', True),
                )
                metrics_str = '  '.join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[check] done: dataset={ds_name} subject={subject_id} trial_idx={t_idx}  |  {metrics_str}")
