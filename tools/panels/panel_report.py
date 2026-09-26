"""report: the backbone comparison report (tools/analysis/backbone_report.py): downstream scores per
cell with Holm-corrected paired tests against --ref, each backbone's backbone_eval.json (run
analysis_pretrain.py's backbone_eval panel first), and the pre-registered verdicts -> report.md, plus CSVs: report_tests, report_verdicts,
backbone_eval_masks, backbone_eval_ablations, backbone_eval_structure."""
import os

from tools.analysis import write_csv
from tools.analysis.backbone_report import backbone_eval_rows, report

STAGES = frozenset({'finetune'})


def run(ctx):
    if not ctx.ref:
        raise RuntimeError('report needs --ref')
    tables = {}
    text = report(ctx.groups, ctx.ref, head=ctx.head, tables=tables)
    out = os.path.join(ctx.out_dir, 'report.md')
    with open(out, 'w') as f:
        f.write(text)
    print(text)
    print(f"  -> {out}")
    tables.update(zip(('backbone_eval_masks', 'backbone_eval_ablations', 'backbone_eval_structure'),
                      backbone_eval_rows(ctx.groups)))
    for name, rows in tables.items():
        if rows:
            print(f"  -> {write_csv(os.path.join(ctx.out_dir, name + '.csv'), rows)}")
