"""Pretrain-stage analysis of one backbone: runs a preset (or --panel list) of tools/panels/ panels.

    python analysis_pretrain.py --run <backbone>            # output/<backbone>/pretrain/checkpoint/last.pth
    python analysis_pretrain.py --checkpoint <path.pth> [--preset quick] [--panel snapshot ...]
    python analysis_pretrain.py --panel profile [--train]   # no checkpoint: the pretrain template

The config is the run's own artifacts/config.json (next to the checkpoint), deep-merged with an optional
--config overlay; the model is rebuilt from the checkpoint's build_config. Output:
output/<output_path>/analysis/ (e.g. output/<backbone>/pretrain/analysis/). Presets: tools/panels PRESETS.
"""
import argparse
import json
import os
import sys

import torch

from tools.analysis import _deep_merge, resolve_output_dir
from tools.panels import PRESETS, PanelContext, discover_panel_names, resolve_panels, run_panels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--run', help='backbone name: output/<run>/pretrain/checkpoint/last.pth')
    src.add_argument('--checkpoint', help='a pretrain checkpoint .pth')
    ap.add_argument('--config', help='JSON overlay deep-merged onto the run config')
    ap.add_argument('--preset', default='standard', choices=sorted(PRESETS['pretrain']))
    ap.add_argument('--panel', action='append', default=[],
                    help=f'run these panels instead of the preset: {discover_panel_names("pretrain")}')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--datasets', nargs='+', help='snapshot/atom_distribution/codebook: these pretrain datasets')
    ap.add_argument('--max-trials', type=int, default=1000, help='codebook/atom_distribution: trials per dataset')
    ap.add_argument('--max-windows', type=int, default=512, help='backbone_eval: held-out windows')
    ap.add_argument('--dup-threshold', type=float, default=0.9, help='atom_duplicates: pair threshold')
    ap.add_argument('--train', action='store_true', help='profile: train-mode timing')
    args = ap.parse_args()

    names = resolve_panels('pretrain', args.preset, args.panel)
    checkpoint = args.checkpoint or (f'output/{args.run}/pretrain/checkpoint/last.pth' if args.run else None)
    if checkpoint:
        config = json.load(open(os.path.join(os.path.dirname(os.path.dirname(checkpoint)), 'artifacts', 'config.json')))
    elif names == ['profile']:
        config = json.load(open('configs/pretrain.template.json'))
    else:
        ap.error('--run or --checkpoint is required (only the profile panel runs without one)')
    if args.config:
        config = _deep_merge(config, json.load(open(args.config)))
    out_dir = resolve_output_dir(config, 'analysis') if checkpoint else 'output/tools-profile'
    os.makedirs(out_dir, exist_ok=True)

    ctx = PanelContext(stage='pretrain', config=config, out_dir=out_dir, args=args,
                       device=torch.device(args.device), checkpoint=checkpoint)
    sys.exit(1 if run_panels(names, ctx) else 0)


if __name__ == '__main__':
    main()
