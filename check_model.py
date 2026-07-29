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

from model.factory import MODEL_REGISTRY


def run(config, output_dir, model, dataset, trial_idx, mode='pretrain', subject_id=None,
        epoch=None, cmap='YlOrRd', plot_recon=True, plot_topo_psd=True, plot_attn_topo=True):
    model_type = 'MeFSQ' if hasattr(model, 'n_routed_experts') else 'MeSAE'
    plugin  = MODEL_REGISTRY[model_type]
    checker = plugin.checker_cls()
    trainer = plugin.trainer_cls()
    check_fn = checker.check_finetune if mode == 'finetune' else checker.check_pretrain
    return check_fn(
        config, output_dir, model, dataset, trial_idx,
        subject_id=subject_id, epoch=epoch, cmap=cmap,
        plot_recon=plot_recon, plot_topo_psd=plot_topo_psd, plot_attn_topo=plot_attn_topo,
        trainer=trainer,
    )


if __name__ == '__main__':
    import argparse
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

    ds_params    = cfg['dataset_params'][data_mode]
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
