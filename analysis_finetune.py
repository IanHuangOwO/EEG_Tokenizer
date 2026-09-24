"""
Post-training checker for the FINETUNE stage only: per-target correct/wrong snapshot
pairs via tools/analysis/snapshot.py's build_finetune_bundle + tools/panels/. Pretrain-stage
analysis (snapshot + codebook) lives in analysis_pretrain.py.

Config resolution: configs/analysis_finetune.template.json (or --config) is a small overlay
— checkpoint (a finetune head.pth),
dataset_params.finetune (one dataset entry), check.plot_* toggles. It's deep-merged onto
the head checkpoint's own run config, taken via --base-config (finetune runs write a
timestamped artifacts/config_<timestamp>.json, not a fixed name, so this cannot be
auto-derived from the checkpoint path the way analysis_pretrain.py's --base-config default
can — always pass --base-config explicitly). Output goes to
output/<output_path>/analysis/<dataset_name>/recon/.
"""

import os
import json


def _load_target_names(dataset_path, num_classes):
    """data_metadata.targets["<idx>"].label, e.g. BNCI2014001's {"0": {"label": "Left hand"}, ...}
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


def _predict_all(model, dataset, patch_len, patch_stride, device):
    """Runs the finetune model over every trial in `dataset` (in index order, no shuffle)
    and returns (preds, labels) numpy arrays aligned to dataset indices — used to find one
    correctly- and one incorrectly-classified trial per target class.

    One trial at a time, no DataLoader/collate: `FinetuneCollate`, which used to batch and
    pad variable-length trials for this loop, no longer exists in train_finetune.py — its
    whole batching pipeline moved to a cached-feature/`source` model (ADR 0016) that has no
    equivalent for raw per-trial patches. `FinetuneModel.forward` itself is unchanged
    (model/MeSAE/MeSAE.py's docstring: "matches the old finetune classes"), and
    tools/analysis/snapshot.py's own `_patchify` patchifies one trial the same way (via
    IO/preprocessing.py's `slice_patches`, patch_stride-aware) with no collate/padding
    needed — this mirrors that exact pattern instead of reimplementing the vanished
    collate function. patch_stride matters: a naive non-overlapping `T // patch_len`
    reshape gives the wrong patch count whenever patch_stride < patch_len, which breaks
    any head whose modules need an exact match (e.g. LearnedTimePool's fixed-size weights,
    sized to training's own patch count)."""
    import torch
    import numpy as np
    from IO.preprocessing import slice_patches

    was_training = model.training
    model.eval()
    preds, labels = [], []
    try:
        with torch.no_grad():
            for i in range(len(dataset)):
                x_raw, coords, label, valid_channels, _valid_length = dataset[i]
                x_patches, time_idx = slice_patches(x_raw, patch_len, patch_stride)
                x_patches = x_patches.unsqueeze(0).to(device)
                time_idx = time_idx.unsqueeze(0).to(device)
                c_in = coords.unsqueeze(0).to(device)
                vc_in = valid_channels.unsqueeze(0).to(device)
                logits = model(x_patches, c_in, time_idx=time_idx, valid_channels=vc_in)[0]
                preds.append(int(logits.argmax(dim=-1).item()))
                labels.append(int(label))
    finally:
        model.train(was_training)
    return np.array(preds), np.array(labels)


if __name__ == '__main__':
    import argparse
    import copy
    import torch
    from IO.dataset import build_dataset_from_config
    from tools.analysis import _deep_merge, load_model, resolve_finetune_analysis_dir
    from tools.analysis.snapshot import build_finetune_bundle
    from tools.panels import PanelContext, run_panels

    parser = argparse.ArgumentParser(description='Post-training EEG finetune checker (MeSAE)')
    parser.add_argument('--config',      default='configs/analysis_finetune.template.json')
    parser.add_argument('--base-config', default=None, dest='base_config')
    parser.add_argument('--checkpoint',  default=None)
    parser.add_argument('--dataset',     type=str, default=None)
    parser.add_argument('--recon_cmap',  type=str, default=None)
    parser.add_argument('--panel',       action='append', default=[],
                         help='Run one or more panels (repeatable) that do not need a '
                              'dataset/bundle (NEEDS_DATASET=False), instead of the legacy '
                              'per-target snapshot path. Panels that need a bundle '
                              '(recon_signal, stamp_by_patch, stamp_gallery) are only '
                              'reachable via the legacy path today. See tools/panels/.')
    parser.add_argument('--train',       action='store_true',
                         help='(panel_profile only) profile in train mode (eigh skipped)')
    parser.add_argument('--group-eval',  action='append', default=[], dest='group_eval',
                         help='(panel_group_summary only) path to an artifacts/group_eval.json, '
                              'or a glob pattern (e.g. '
                              '"output/<backbone>/finetune/*/*/artifacts/group_eval.json" for '
                              'every head/dataset_mode in the whole baseline matrix at once) '
                              '(repeatable; first path/match is the reference every later one '
                              'is paired against)')
    parser.add_argument('--group-eval-out', default=None, dest='group_eval_out',
                         help='(panel_group_summary only) dir to write group_summary.csv/'
                              'group_summary_folds.csv into (default: the first --group-eval '
                              "match's own output/<backbone>/finetune/analysis/)")
    args = parser.parse_args()

    if args.panel:
        from tools.panels import build_panel_context, run_panels

        def _resolve_base_path(args, overlay, checkpoint):
            base_path = args.base_config or overlay.pop('base_config', None)
            if not base_path:
                raise ValueError(
                    "analysis_finetune.py needs --base-config (or overlay['base_config']): "
                    "a finetune run's artifacts/config_<timestamp>.json has no fixed name "
                    "to guess.")
            return base_path

        ctx = build_panel_context(args, args.panel, 'finetune', _resolve_base_path)
        run_panels(args.panel, 'finetune', ctx)
        raise SystemExit(0)

    with open(args.config, 'r') as f:
        overlay = json.load(f)

    checkpoint = args.checkpoint or overlay.get('checkpoint', '')
    mode       = 'finetune'
    data_mode  = 'finetune'

    base_path = args.base_config or overlay.pop('base_config', None)
    if not base_path:
        raise ValueError(
            "analysis_finetune.py needs --base-config (or overlay['base_config']): a "
            "finetune run's artifacts/config_<timestamp>.json has no fixed name to guess.")
    with open(base_path, 'r') as f:
        base = json.load(f)
    cfg = _deep_merge(base, overlay)
    for m, dsp in overlay.get('dataset_params', {}).items():
        cfg['dataset_params'][m] = dsp

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl    = load_model(cfg, checkpoint, device, mode=mode)

    check_cfg = cfg.get('check', {})
    cmap = args.recon_cmap or check_cfg.get('cmap', 'YlOrRd')
    ds_params = cfg['dataset_params'][data_mode]

    # Per-target correct/wrong snapshot pairs, not a single per-subject trial pick:
    # for each of the num_classes targets, find one trial the model got right and one
    # it got wrong, and render the full 3-panel snapshot (recon_signal, topo_psd_filter,
    # stamp_panel) for each — num_classes * 2 * 3 files total, searched across every
    # subject in dataset_params.finetune[dataset_name].subject_to_use (not one subject
    # at a time), since a single subject isn't guaranteed to contain both a correct and
    # a wrong example of every class.
    dataset_name = args.dataset or next(iter(ds_params))
    ds_cfg       = ds_params[dataset_name]

    filtered = copy.deepcopy(cfg)
    filtered['dataset_params'][data_mode] = {dataset_name: ds_cfg}
    # subject_to_use=["all"] needs the same resolution train_finetune.py's own dataset
    # builder does (same call shape as its own subject_to_use resolution, see
    # train_finetune.py's build_dataset_from_config caller) — build_dataset_from_config
    # takes it literally and fails ("Subject all not found") since EEGDataset expects
    # real subject ids.
    from train_finetune import _resolve_all_subjects, resolve_subjects
    all_subjects = _resolve_all_subjects(ds_cfg['dataset_path'])
    filtered['dataset_params'][data_mode][dataset_name]['subject_to_use'] = \
        resolve_subjects(ds_cfg['subject_to_use'], all_subjects)
    ds = build_dataset_from_config(filtered, mode=data_mode)

    patch_len = filtered.get('preprocess_params', {}).get('patch_length', 100)
    patch_stride = filtered.get('preprocess_params', {}).get('patch_stride', patch_len)
    preds, labels = _predict_all(mdl, ds, patch_len, patch_stride, device)
    num_classes  = int(labels.max()) + 1
    target_names = _load_target_names(ds_cfg['dataset_path'], num_classes)

    out = resolve_finetune_analysis_dir(filtered, dataset_name)
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
            try:
                bundle, metrics = build_finetune_bundle(mdl, ds, t_idx, filtered, device,
                                                          subject_id=subject_id, tag=tag)
                panels = []
                if check_cfg.get('plot_recon', True):
                    panels.append('recon_signal')
                if check_cfg.get('plot_topo_psd', True):
                    panels.append('stamp_gallery')
                panel_ctx = PanelContext(config=filtered, output_dir=out, device=device, args=args,
                                          model=mdl, dataset=ds, cmap=cmap, bundle=bundle)
                run_panels(panels, 'finetune', panel_ctx)
                metrics_str = '  '.join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[check] done: target={cls_idx}({name}) status={status} subject={subject_id} "
                      f"trial_idx={t_idx}  |  {metrics_str}")
            except Exception as e:
                print(f"[check] FAILED: target={cls_idx}({name}) status={status} subject={subject_id} "
                      f"trial_idx={t_idx}  |  {e}")
                continue
