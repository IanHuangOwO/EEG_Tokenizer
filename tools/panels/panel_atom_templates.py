"""atom_templates: every Q-atom's template D and quadrature partner H
(tools.analysis.atom_dist.atom_templates) -> analysis/atom_templates.png."""
import os

from tools.analysis.atom_dist import atom_templates
from tools.viz.atom_plots import plot_atom_templates

STAGES = frozenset({'pretrain'})


def run(ctx):
    out = os.path.join(ctx.out_dir, 'atom_templates.png')
    plot_atom_templates(out, *atom_templates(ctx.model))
    print(f"  -> {out}")
