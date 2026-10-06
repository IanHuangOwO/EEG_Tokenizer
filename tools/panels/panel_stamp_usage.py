"""stamp_usage: per-atom ranking, energy share and remove-one-stamp cost on held-out windows
(tools/analysis/stamp_usage.py) -> analysis/stamp_usage.json. Options: --max-windows (capped at 256)."""
import os

from tools.analysis.stamp_usage import stamp_usage

STAGES = frozenset({'pretrain'})


def run(ctx):
    stamp_usage(ctx.model, ctx.config, os.path.join(ctx.out_dir, 'stamp_usage.json'),
                max_windows=min(ctx.args.max_windows, 256))
    ctx.model.to(ctx.device)   # stamp_usage() runs on the CPU
