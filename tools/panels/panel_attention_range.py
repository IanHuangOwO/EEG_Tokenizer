"""attention_range: measured temporal and spatial attention range per encoder block on held-out
windows (tools/analysis/attention_range.py) -> analysis/attention_range.json. Options: --max-windows
(capped at 256 here: attention maps are held in memory per batch)."""
import os

from tools.analysis.attention_range import attention_range

STAGES = frozenset({'pretrain'})


def run(ctx):
    attention_range(ctx.model, ctx.config, os.path.join(ctx.out_dir, 'attention_range.json'),
                    max_windows=min(ctx.args.max_windows, 256))
    ctx.model.to(ctx.device)   # attention_range() runs on the CPU
