"""
Analysis runner. Loads model and trial once, dispatches to enabled modules.

Usage
-----
    python run_analysis.py                          # all analyses
    python run_analysis.py --codebook               # codebook only
    python run_analysis.py --reconstruction         # recon only
    python run_analysis.py --subject 3 --trial 7   # override subject/trial
    python run_analysis.py --checkpoint output/attnvq_v05/tokenizer/best_tokenizer.pth
"""

import argparse
import torch

from IO.dataset import build_dataset_from_config
from viz import (
    load_config,
    load_model,
    resolve_output_dir,
    select_subject_dataset,
    filter_config_to_subject,
    pick_trial,
)

import viz.check_codebook as _codebook
import viz.check_epoch    as _reconstruction

ALL_ANALYSES = {
    'codebook':       _codebook,
    'reconstruction': _reconstruction,
}


def main():
    parser = argparse.ArgumentParser(
        description='EEG Analysis Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config',     default='config/analysis.json')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--subject',    type=int, default=None)
    parser.add_argument('--trial',      type=int, default=None)
    parser.add_argument('--dataset',    type=str, default=None)

    for name in ALL_ANALYSES:
        parser.add_argument(f'--{name}', action='store_true')
    for module in ALL_ANALYSES.values():
        module.add_args(parser)

    args = parser.parse_args()

    selected = [name for name in ALL_ANALYSES if getattr(args, name, False)]
    if not selected:
        selected = list(ALL_ANALYSES.keys())
    print(f"Analyses: {', '.join(selected)}\n")

    cfg        = load_config(args.config)
    checkpoint = args.checkpoint or cfg.get('checkpoint', '')
    ds_key     = args.dataset or next(
        (ds for ds, dc in cfg['dataset_params'].items() if 'trial_to_use' in dc),
        next(iter(cfg['dataset_params'])))
    trial_cfg  = args.trial if args.trial is not None else cfg['dataset_params'][ds_key].get('trial_to_use')

    ds_name, subject = select_subject_dataset(cfg, args.subject, args.dataset)
    filtered   = filter_config_to_subject(cfg, ds_name, subject)
    output_dir = resolve_output_dir(filtered, 'check')
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model   = load_model(filtered, checkpoint, device)
    dataset = build_dataset_from_config(filtered, mode='pretrain')
    trial_idx, subject_id = pick_trial(dataset, subject, trial_cfg)

    print(f"Dataset : {ds_name}")
    print(f"Subject : {subject_id}")
    print(f"Trial   : {trial_idx}\n")

    for name in selected:
        sep = '=' * 60
        print(f"\n{sep}\n  {name.upper()}\n{sep}")
        try:
            ALL_ANALYSES[name].run(
                filtered, output_dir, args,
                model=model, dataset=dataset,
                trial_idx=trial_idx, subject_id=subject_id,
            )
        except Exception as exc:
            print(f"  [ERROR] {name} failed: {exc}")
            raise

    print(f"\nAll done -> {output_dir}")


if __name__ == '__main__':
    main()
