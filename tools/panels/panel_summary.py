"""summary: one table of every finetune run of every group's backbone under --head (all cells, tail
balanced accuracy), paired tests of each group against --ref -> summary.md. With --metric kappa_tail
etc. for another score, --rank N for the top N rows per column instead."""
import os

from tools.analysis.summarize_runs import collect, render

STAGES = frozenset({'finetune'})


def run(ctx):
    table = collect([f'output/{bb}/finetune/{ctx.head}/*' for bb in ctx.groups.values()], ctx.args.metric)
    if not table:
        raise RuntimeError(f"no finetune runs with group_eval.json under output/<backbone>/finetune/{ctx.head}/")
    ref = f'{ctx.groups[ctx.ref]}:{ctx.head}' if ctx.ref else None
    text = render(table, ctx.args.metric, ref, ctx.args.rank)
    print(text)
    out = os.path.join(ctx.out_dir, 'summary.md')
    with open(out, 'w') as f:
        f.write(text + '\n')
    print(f"  -> {out}")
