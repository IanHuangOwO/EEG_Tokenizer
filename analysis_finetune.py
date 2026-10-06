"""Finetune-stage analysis across backbones: runs a preset (or --panel list) of tools/panels/ panels over
the finetune runs of every group's backbone under one head label (output/<backbone>/finetune/<head>/<cell>).

    python analysis_finetune.py --group base=qtome_tiny_base --group AB=qtome_tiny_ab --ref base
    python analysis_finetune.py --group AB=qtome_tiny_ab --panel class_snapshots \\
        --checkpoint output/Qtome/qtome_tiny_ab/finetune/frozen_learned/BNCI2014001_loso/finetune/run_fold0/head.pth

Output: --out, default output/analysis/<group names joined by _>/. Presets: tools/panels PRESETS.
"""
import argparse
import os
import sys

import torch

from tools.panels import PRESETS, PanelContext, discover_panel_names, resolve_panels, run_panels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--group', action='append', required=True, metavar='NAME=BACKBONE',
                    help='a backbone to compare (repeatable, report order)')
    ap.add_argument('--ref', help='the reference group (summary/report/seed_equivalence test against it)')
    ap.add_argument('--head', default='frozen_learned', help='finetune label: output/<backbone>/finetune/<head>/')
    ap.add_argument('--preset', default='standard', choices=sorted(PRESETS['finetune']))
    ap.add_argument('--panel', action='append', default=[],
                    help=f'run these panels instead of the preset: {discover_panel_names("finetune")}')
    ap.add_argument('--out', help='output dir (default output/analysis/<group names>/)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--metric', default='tail', choices=['tail', 'last', 'kappa_tail', 'kappa_last'],
                    help='summary/seed_equivalence: score')
    ap.add_argument('--rank', type=int, default=0, help='summary: top N rows per column instead of the table')
    ap.add_argument('--margin', type=float, default=0.02, help='seed_equivalence: equivalence margin')
    ap.add_argument('--checkpoint', help='class_snapshots: a head.pth')
    ap.add_argument('--event-onset', action='append', default=[], metavar='DATASET=SECONDS', dest='event_onset',
                    help="probe_maps: the event's time in the trial window, for runs made on an older trial "
                         "window than the dataset's current metadata (e.g. BNCI2014001=1.0)")
    args = ap.parse_args()

    groups = dict(g.split('=', 1) for g in args.group)
    if args.ref and args.ref not in groups:
        ap.error(f'--ref {args.ref!r} is not a group name ({list(groups)})')
    names = resolve_panels('finetune', args.preset, args.panel)
    out_dir = args.out or os.path.join('output', 'analysis', '_'.join(groups))
    os.makedirs(out_dir, exist_ok=True)

    ctx = PanelContext(stage='finetune', config={}, out_dir=out_dir, args=args, device=torch.device(args.device),
                       groups=groups, ref=args.ref, head=args.head)
    sys.exit(1 if run_panels(names, ctx) else 0)


if __name__ == '__main__':
    main()
