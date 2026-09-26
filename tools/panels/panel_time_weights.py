"""time_weights: where in the trial a learned time pool reads from (tools.analysis.stamp_dist.
time_pool_weights). Per cell, one stamp x patch heatmap per group and head entry, averaged over the
cell's fold/subject heads (each trains its own) -> time_weights_<cell>.png, event onset marked."""
import glob
import json
import os

import numpy as np

from tools.analysis import event_onset_patch
from tools.analysis.stamp_dist import time_pool_weights
from tools.viz.stamp_plots import plot_time_weights

STAGES = frozenset({'finetune'})


def run(ctx):
    cells = sorted({os.path.basename(d) for bb in ctx.groups.values()
                    for d in glob.glob(f'output/{bb}/finetune/{ctx.head}/*')})
    for cell in cells:
        maps, cfg = {}, None
        for g, bb in ctx.groups.items():
            run_dir = f'output/{bb}/finetune/{ctx.head}/{cell}'
            heads = [time_pool_weights(p) for p in sorted(glob.glob(f'{run_dir}/finetune/run_*/head.pth'))]
            for entry in sorted({e for w in heads for e in w}):
                ws = [w[entry] for w in heads if entry in w]
                ws = [w for w in ws if w.shape == ws[0].shape]
                maps[f'{g} / {entry} (mean of {len(ws)} heads)'] = np.mean(ws, 0)
            if cfg is None and os.path.exists(f'{run_dir}/artifacts/config.json'):
                cfg = json.load(open(f'{run_dir}/artifacts/config.json'))
        if not maps:
            continue
        ev = event_onset_patch(cfg, next(iter(cfg['dataset_params']['finetune']))) if cfg else None
        out = os.path.join(ctx.out_dir, f'time_weights_{cell}.png')
        plot_time_weights(out, maps, f'{cell}: learned time-pool weights ({ctx.head})', ev)
        print(f"  -> {out}")
