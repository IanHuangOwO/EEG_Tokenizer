"""coord_robustness: loso ridge probe trained on normal coordinates, tested with jittered / mirrored coordinates
(tools/analysis/ridge_probe.py coord_robustness)
-> analysis/coord_robustness.json."""
import os

from tools.analysis.ridge_probe import coord_robustness

STAGES = frozenset({'pretrain'})


def run(ctx):
    coord_robustness(ctx.config, ctx.checkpoint, os.path.join(ctx.out_dir, 'coord_robustness.json'))
