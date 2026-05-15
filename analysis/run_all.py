"""
Central analysis runner. Loads the model and selects a trial once, then
dispatches to every enabled analysis module so they all see the same trial.

Usage
-----
# Run all analyses (default):
    python -m analysis.run_all

# Run a specific subset:
    python -m analysis.run_all --codebook --topomap

# Override subject / trial / checkpoint:
    python -m analysis.run_all --subject 3 --trial 7 --checkpoint output/attnvq_v05/tokenizer/best_tokenizer.pth

# To add a new analysis in future: implement add_args() + run() in a new
# analysis/<name>.py module, then register it in ALL_ANALYSES below.
"""

import argparse
import torch

from IO.dataset import build_dataset_from_config
from analysis import (
    load_config,
    load_model,
    resolve_output_dir,
    select_subject_dataset,
    filter_config_to_subject,
    pick_trial,
)

import analysis.codebook  as _codebook
import analysis.data_check as _data_check
# import analysis.spatial_coherence as _spatial  # temporarily disabled
import analysis.topomap   as _topomap

# Registry: order controls execution order; key is the CLI flag name.
ALL_ANALYSES = {
    'data':     _data_check,
    'codebook': _codebook,
    # 'spatial':  _spatial,  # temporarily disabled
    'topomap':  _topomap,
}

# Analyses that need a tokenizer-mode dataset + trial selection.
_NEEDS_TRIAL = {'topomap', 'data'}
# Analyses that need the model loaded.
_NEEDS_MODEL = {'codebook', 'topomap', 'data'}


def main():
    parser = argparse.ArgumentParser(
        description='EEG Analysis Suite — run all or selected analyses on one trial',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Shared args
    parser.add_argument('--config',     default='config/analysis.json',
                        help='Path to analysis config (default: config/analysis.json)')
    parser.add_argument('--checkpoint', default=None,
                        help='Override checkpoint path from config')
    parser.add_argument('--subject',    type=int, default=None,
                        help='Override subject ID from config')
    parser.add_argument('--trial',      type=int, default=None,
                        help='Override trial index from config (None = random)')
    parser.add_argument('--dataset',    type=str, default=None,
                        help='Override dataset name from config')

    # Per-analysis enable flags
    for name in ALL_ANALYSES:
        parser.add_argument(f'--{name}', action='store_true',
                            help=f'Enable {name} analysis (default: all enabled)')

    # Per-analysis extra args (registered by each module)
    for module in ALL_ANALYSES.values():
        module.add_args(parser)

    args = parser.parse_args()

    # Determine which analyses to run
    selected = [name for name in ALL_ANALYSES if getattr(args, name, False)]
    if not selected:
        selected = list(ALL_ANALYSES.keys())   # default: all
    print(f"Analyses: {', '.join(selected)}\n")

    # ── Load config ──────────────────────────────────────────────────────────
    cfg        = load_config(args.config)
    checkpoint = args.checkpoint or cfg.get('checkpoint', '')
    # Resolve dataset early to read trial_to_use; mirrors select_subject_dataset logic
    _ds_early  = args.dataset or next(
        (ds for ds, dc in cfg['dataset_params'].items() if 'trial_to_use' in dc),
        next(iter(cfg['dataset_params'])))
    trial_cfg  = args.trial if args.trial is not None else cfg['dataset_params'][_ds_early].get('trial_to_use')

    # ── Resolve subject + dataset ────────────────────────────────────────────
    ds_name, subject = select_subject_dataset(cfg, args.subject, args.dataset)
    filtered = filter_config_to_subject(cfg, ds_name, subject)

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = resolve_output_dir(filtered, 'check')

    # ── Load model once (shared by all model-based analyses) ─────────────────
    needs_model = any(name in _NEEDS_MODEL for name in selected)
    model = load_model(filtered, checkpoint, device) if needs_model else None

    # ── Build tokenizer dataset + pick trial once ─────────────────────────────
    needs_trial   = any(name in _NEEDS_TRIAL for name in selected)
    dataset       = None
    trial_idx     = None
    subject_id    = subject

    if needs_trial:
        dataset = build_dataset_from_config(filtered, mode='tokenizer')
        trial_idx, subject_id = pick_trial(dataset, subject, trial_cfg)
        print(f"Dataset : {ds_name}")
        print(f"Subject : {subject_id}")
        print(f"Trial   : {trial_idx}\n")

    # ── Dispatch ─────────────────────────────────────────────────────────────
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

    print(f"\nAll done → {output_dir}")


if __name__ == '__main__':
    main()
