"""stamp_maps: where a head with a stamp_power entry reads from, per cell and group (a latent_signed entry of the
same head, if any, is drawn on top: virtual channel x time and spatial filters, as probe_maps): each stamp's learned time weights
(stamps ranked by decision importance, labelled with their template's peak frequency) and, for the most important
stamps, which electrodes' power the decision uses (tools/analysis/probe_maps.py stamp_head_maps), over every fold and
finetune seed of the cell -> stamp_maps_<cell>_<group>.png. Run it with --head <a label whose head has stamp_power>,
e.g. combined or cw_stamp. One figure per group: stamps are each backbone's own dictionary, not comparable across."""
import glob
import json
import os
import re

import numpy as np
import torch

from IO.dataset import resolve_canonical_channels
from IO.loader import get_standard_coords
from model.factory import build_from_checkpoint
from tools.analysis.probe_maps import summarise, summarise_stamps
from tools.panels.panel_probe_maps import _time_axis
from tools.viz.probe_plots import plot_stamp_maps

STAGES = frozenset({'finetune'})


def _stamp_labels(backbone_ckpt, sample_freq):
    """'s<i> <peak> Hz' per stamp, from the backbone's unit templates."""
    stamps = build_from_checkpoint(torch.load(backbone_ckpt, map_location='cpu', weights_only=False)).stamps
    with torch.no_grad():
        D, _ = stamps.templates()
    f = torch.fft.rfftfreq(D.shape[-1], 1.0 / sample_freq)
    peak = f[torch.fft.rfft(D.float(), dim=-1).abs().argmax(-1)]
    return [f's{i} {float(p):.0f} Hz' for i, p in enumerate(peak)]


def run(ctx):
    override = dict(e.split('=', 1) for e in getattr(ctx.args, 'event_onset', []))
    for g, bb in ctx.groups.items():
        cells = sorted({re.sub(r'_seed\d+$', '', os.path.basename(d)) for d in glob.glob(f'output/{bb}/finetune/{ctx.head}/*')})
        for cell in cells:
            dirs = sorted(glob.glob(f'output/{bb}/finetune/{ctx.head}/{cell}') + glob.glob(f'output/{bb}/finetune/{ctx.head}/{cell}_seed*'))
            heads = [p for d in dirs for p in sorted(glob.glob(f'{d}/finetune/run_*/head.pth'))]
            if not heads:
                continue
            try:
                tw, imp, chan, n = summarise_stamps(heads)
            except KeyError:
                continue                                    # no stamp_power entry in this head
            try:
                z_imp, z_sp, _ = summarise(heads)             # the same head's latent_signed half, if it has one
                z_maps = (z_imp.numpy(), z_sp.numpy())
            except KeyError:
                z_maps = None
            cfg = json.load(open(f'{dirs[0]}/artifacts/config.json'))
            t, note = _time_axis(cfg, tw.shape[1], override)
            h0 = torch.load(heads[0], map_location='cpu', weights_only=False)
            chans = resolve_canonical_channels(cfg['preprocess_params']['canonical_channels'])
            names = [chans[i] for i in h0['head_config']['channel_idx']]
            xy = np.array([get_standard_coords(c)[:2] for c in names])
            labels = _stamp_labels(h0['backbone_checkpoint'], float(cfg['preprocess_params']['sample_freq']))
            out = os.path.join(ctx.out_dir, f'stamp_maps_{cell}_{g}.png')
            plot_stamp_maps(out, tw.numpy(), imp.numpy(), chan.numpy(), t, labels, xy, names,
                            f'{cell}, {g} ({bb}): {ctx.head}, {n} heads{note}',
                            event_s=None if note.startswith(' (event line omitted') else 0.0, z_maps=z_maps)
            print(f"  -> {out}")
