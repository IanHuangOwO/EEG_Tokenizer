"""group_summary panel: cross-fold/cross-subject stats over train_finetune.py's
artifacts/group_eval.json (tools/analysis/group_summary.py's print_group_summary).
No checkpoint/dataset needed -- reads one or more group_eval.json paths from
ctx.args.group_eval (--group-eval, repeatable; first path is the reference every later
one gets paired against)."""
from tools.analysis.group_summary import print_group_summary

STAGES = frozenset({'finetune'})  # group_eval.json only ever comes from train_finetune.py
NEEDS_CHECKPOINT = False
NEEDS_DATASET = False


def run(ctx):
    paths = ctx.args.group_eval
    if not paths:
        raise ValueError("panel 'group_summary' needs --group-eval <path/to/group_eval.json> "
                          "(repeatable, first is the reference)")
    print_group_summary(paths)
