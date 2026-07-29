"""
Standalone per-epoch topo/PSD/attention snapshot (MeFSQ or MeSAE, resolved from the model
instance) via BaseEpochChecker.check_pretrain (model/base_checker.py).
"""

from model.factory import MODEL_REGISTRY


def run(config, output_dir, model, dataset, trial_idx, subject_id=None, epoch=None, cmap='YlOrRd'):
    model_type = 'MeFSQ' if hasattr(model, 'n_routed_experts') else 'MeSAE'
    checker = MODEL_REGISTRY[model_type].checker_cls()
    checker.check_pretrain(
        config, output_dir, model, dataset, trial_idx,
        subject_id=subject_id, epoch=epoch, cmap=cmap,
    )


if __name__ == '__main__':
    import argparse
    import torch
    from IO.dataset import build_dataset_from_config
    from viz import (
        load_config, load_model, resolve_output_dir,
        select_subject_dataset, filter_config_to_subject, pick_trial,
    )

    parser = argparse.ArgumentParser(description='EEG Pretrain Epoch Snapshot (MeFSQ or MeSAE)')
    parser.add_argument('--config',     default='config/analysis.json')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--subject',    type=int, default=None)
    parser.add_argument('--trial',      type=int, default=None)
    parser.add_argument('--dataset',    type=str, default=None)
    parser.add_argument('--recon_cmap', type=str, default='YlOrRd')
    args = parser.parse_args()

    cfg        = load_config(args.config)
    checkpoint = args.checkpoint or cfg.get('checkpoint', '')
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl        = load_model(cfg, checkpoint, device)
    cmap       = args.recon_cmap or cfg.get('check', {}).get('reconstruction', {}).get('cmap', 'YlOrRd')

    if args.subject is not None:
        targets = [{'dataset': args.dataset, 'subject': args.subject, 'trial': args.trial}]
    else:
        targets = cfg.get('training_params', {}).get('visualize', {}).get('targets') or [{}]

    for t in targets:
        ds_name, subject = select_subject_dataset(cfg, t.get('subject'), dataset_name=t.get('dataset'))
        filtered = filter_config_to_subject(cfg, ds_name, subject)
        ds       = build_dataset_from_config(filtered, mode='pretrain')
        trial_cfg = t.get('trial') if t.get('trial') is not None else cfg['dataset_params']['pretrain'][ds_name].get('trial_to_use')
        t_idx, subject_id = pick_trial(ds, subject, trial_cfg, dataset_name=ds_name)
        out = resolve_output_dir(filtered, 'check')
        run(filtered, out, mdl, ds, t_idx, subject_id=subject_id, cmap=cmap)
        print(f"[check] done: dataset={ds_name} subject={subject_id} trial_idx={t_idx}")
