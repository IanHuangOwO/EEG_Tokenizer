"""select_eval_subsets panel: reproducible seen/unseen subject subsets for group-holdout
finetune evals (tools/analysis/select_eval_subsets.py). Builds its own dataset(s)
internally (same self-contained pattern as panel_profile.py) -- no checkpoint, no
pre-built ctx.dataset. Writes configs/finetune_eval_splits/<name>.json.

CLI (via ctx.args): --se-datasets (repeatable, default: every key in
tools.analysis.select_eval_subsets.DATASETS), --se-run-config (default: that module's
RUN), --se-out-dir (default: that module's 'configs/finetune_eval_splits')."""
from tools.analysis.select_eval_subsets import DATASETS, RUN, select_eval_subsets

STAGES = frozenset({'pretrain', 'finetune'})
NEEDS_CHECKPOINT = False
NEEDS_DATASET = False


def run(ctx):
    args = ctx.args
    names = getattr(args, 'se_datasets', None) or None
    run_config = getattr(args, 'se_run_config', None) or RUN
    out_dir = getattr(args, 'se_out_dir', None) or 'configs/finetune_eval_splits'
    select_eval_subsets(names=names, run_config=run_config, out_dir=out_dir)
