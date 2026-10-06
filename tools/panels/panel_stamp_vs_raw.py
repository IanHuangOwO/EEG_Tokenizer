"""stamp_vs_raw: closed-form loso ridge on log Q-atom power vs on log raw band power, same trials and split
(tools/analysis/ridge_probe.py stamp_vs_raw) -> analysis/stamp_vs_raw.json. Positive Q-atom - raw: the Q-atom
code carries class information a raw spectral filterbank does not."""
import os

from tools.analysis.ridge_probe import stamp_vs_raw

STAGES = frozenset({'pretrain'})


def run(ctx):
    stamp_vs_raw(ctx.config, ctx.checkpoint, os.path.join(ctx.out_dir, 'stamp_vs_raw.json'))
