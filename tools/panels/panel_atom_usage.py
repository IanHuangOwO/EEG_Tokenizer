"""atom_usage: per-atom ranking, energy share and remove-one-atom cost on held-out windows
(tools/analysis/atom_usage.py) -> analysis/atom_usage.json. Options: --max-windows (capped at 256)."""
import os

from tools.analysis.atom_usage import atom_usage

STAGES = frozenset({'pretrain'})


def run(ctx):
    atom_usage(ctx.model, ctx.config, os.path.join(ctx.out_dir, 'atom_usage.json'),
                max_windows=min(ctx.args.max_windows, 256))
    ctx.model.to(ctx.device)   # atom_usage() runs on the CPU
