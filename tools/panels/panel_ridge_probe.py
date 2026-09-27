"""ridge_probe: deterministic closed-form linear probe on the frozen pre-stamp z, Compass LOSO on
BNCI2014004 / 001 / 008 (tools/analysis/ridge_probe.py) -> analysis/ridge_probe.json."""
import os

from tools.analysis.ridge_probe import ridge_probe

STAGES = frozenset({'pretrain'})


def run(ctx):
    ridge_probe(ctx.config, ctx.checkpoint, os.path.join(ctx.out_dir, 'ridge_probe.json'))
