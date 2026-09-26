"""report: the backbone comparison report (tools/analysis/backbone_report.py): downstream scores per
cell with Holm-corrected paired tests against --ref, each backbone's backbone_eval.json (run
analysis_pretrain.py's backbone_eval panel first), and the pre-registered verdicts -> report.md."""
import os

from tools.analysis.backbone_report import report

STAGES = frozenset({'finetune'})


def run(ctx):
    if not ctx.ref:
        raise RuntimeError('report needs --ref')
    text = report(ctx.groups, ctx.ref, head=ctx.head)
    out = os.path.join(ctx.out_dir, 'report.md')
    with open(out, 'w') as f:
        f.write(text)
    print(text)
    print(f"  -> {out}")
