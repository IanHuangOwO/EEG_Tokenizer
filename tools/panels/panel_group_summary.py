"""group_summary panel: cross-fold/cross-subject stats over train_finetune.py's
artifacts/group_eval.json (tools/analysis/group_summary.py's print_group_summary +
write_group_summary_csv). No checkpoint/dataset needed -- reads one or more
group_eval.json paths from ctx.args.group_eval (--group-eval, repeatable; first path is
the reference every later one gets paired against). Each value may be a literal path or
a glob pattern (e.g. 'output/<backbone>/finetune/*/*/artifacts/group_eval.json' -- the
whole baseline matrix's group_eval.json files in one shot, per head/dataset_mode); glob
matches are sorted and deduped against any already-collected path, preserving
first-seen order (a literal path listed before a glob still becomes the reference).
Always writes group_summary.csv (pooled rollup + paired comparisons) and
group_summary_folds.csv (every raw per-fold row) to
output/<backbone>/finetune/analysis/, or --group-eval-out if given."""
from tools.analysis.group_summary import expand_glob_paths, print_group_summary, write_group_summary_csv

STAGES = frozenset({'finetune'})  # group_eval.json only ever comes from train_finetune.py
NEEDS_CHECKPOINT = False
NEEDS_DATASET = False


def run(ctx):
    raw = ctx.args.group_eval
    if not raw:
        raise ValueError("panel 'group_summary' needs --group-eval <path/to/group_eval.json> "
                          "(repeatable, first is the reference; a glob pattern like "
                          "'output/<backbone>/finetune/*/*/artifacts/group_eval.json' "
                          "expands to every matching file)")
    paths = expand_glob_paths(raw)
    if not paths:
        raise ValueError(f"no group_eval.json files matched: {raw}")
    print_group_summary(paths)
    out_dir = getattr(ctx.args, 'group_eval_out', None) or None
    write_group_summary_csv(paths, out_dir=out_dir)
