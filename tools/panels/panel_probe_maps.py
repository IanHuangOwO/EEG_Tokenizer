"""probe_maps: where the linear probe (latent_signed head entry) reads from, per cell: virtual channel x
time decision weight (event at 0 s) and the spatial filters (tools/analysis/probe_maps.py), over every
fold and finetune seed of the cell (cells named *_seed<N> are pooled) -> probe_maps_<cell>.png.
Run it with --head <the probe's head label>, e.g. cw_probe."""
import glob
import json
import os
import re

import numpy as np
import torch

from IO.dataset import resolve_canonical_channels
from tools.analysis import lookup_event_onset_sample
from tools.analysis.probe_maps import summarise
from tools.viz.probe_plots import plot_probe_maps

STAGES = frozenset({'finetune'})


def run(ctx):
    cells = sorted({re.sub(r'_seed\d+$', '', os.path.basename(d)) for bb in ctx.groups.values()
                    for d in glob.glob(f'output/{bb}/finetune/{ctx.head}/*')})
    for cell in cells:
        maps, cfg, n_patch = {}, None, None
        for g, bb in ctx.groups.items():
            dirs = sorted(glob.glob(f'output/{bb}/finetune/{ctx.head}/{cell}')
                          + glob.glob(f'output/{bb}/finetune/{ctx.head}/{cell}_seed*'))
            heads = [p for d in dirs for p in sorted(glob.glob(f'{d}/finetune/run_*/head.pth'))]
            if not heads:
                continue
            try:
                imp, sp, n = summarise(heads)
            except KeyError:
                continue                                    # not a latent_signed head
            maps[f'{g} ({n} heads)'] = (imp.numpy(), sp.numpy())
            n_patch = imp.shape[1]
            if cfg is None:
                cfg = json.load(open(f'{dirs[0]}/artifacts/config.json'))
        if not maps:
            continue
        ds = next(iter(cfg['dataset_params']['finetune']))
        pp = cfg['preprocess_params']
        L, stride, sf = pp.get('patch_length', 50), pp.get('patch_stride', 50), float(pp['sample_freq'])
        meta = json.load(open(f"{cfg['dataset_params']['finetune'][ds]['dataset_path']}/metadata.json"))['data_metadata']
        n_expected = int((meta['acquisition']['window_size_seconds'] * sf - L) // stride + 1)
        onset = lookup_event_onset_sample(cfg, ds)
        note = ''
        if onset is not None and n_expected != n_patch:   # the run predates the dataset's current trial window
            onset, note = None, ' (event line omitted: run predates the current trial window)'
        t = (np.arange(n_patch) * stride + L / 2 - (onset or 0)) / sf
        chans = resolve_canonical_channels(pp['canonical_channels'])
        hc = torch.load(glob.glob(f'output/{next(iter(ctx.groups.values()))}/finetune/{ctx.head}/{cell}*/finetune/run_*/head.pth')[0],
                        map_location='cpu', weights_only=False)['head_config']
        names = [chans[i] for i in hc['channel_idx']]
        out = os.path.join(ctx.out_dir, f'probe_maps_{cell}.png')
        plot_probe_maps(out, maps, names, f'{cell}: linear probe ({ctx.head}){note}', t,
                        event_s=0.0 if onset is not None else None)
        print(f"  -> {out}")
