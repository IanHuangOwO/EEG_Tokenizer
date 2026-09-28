"""stamp_templates: every stamp's template D and quadrature partner H
(tools.analysis.stamp_dist.stamp_templates) -> analysis/stamp_templates.png."""
import os

from tools.analysis.stamp_dist import stamp_templates
from tools.viz.stamp_plots import plot_stamp_templates

STAGES = frozenset({'pretrain'})


def run(ctx):
    out = os.path.join(ctx.out_dir, 'stamp_templates.png')
    plot_stamp_templates(out, *stamp_templates(ctx.model))
    print(f"  -> {out}")
