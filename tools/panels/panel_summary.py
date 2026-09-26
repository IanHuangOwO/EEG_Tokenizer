"""summary: one table of every finetune run of every group's backbone under --head (all cells, tail
balanced accuracy), paired tests of each group against --ref -> summary.md, plus summary.csv (per group x cell:
mean / sd of tail, last, kappa_tail, kappa_last) and summary_subjects.csv (every subject score). With --metric kappa_tail
etc. for another score, --rank N for the top N rows per column instead."""
import os

from tools.analysis import write_csv
from tools.analysis.summarize_runs import collect, mean_rows, render, subject_rows

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
    subj = subject_rows(ctx.groups, ctx.head)
    for name, rows in (('summary.csv', mean_rows(subj)), ('summary_subjects.csv', subj)):
        print(f"  -> {write_csv(os.path.join(ctx.out_dir, name), rows)}")
