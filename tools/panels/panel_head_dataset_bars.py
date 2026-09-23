"""head_dataset_bars panel: grouped bar chart comparing every HEAD's balanced accuracy
and Cohen's Kappa across every DATASET, for one backbone (tools/viz/bars.py's
plot_grouped_bars, driven by tools/analysis/head_dataset_matrix.py's build_matrix). One
figure per metric (accuracy, kappa) -- "per plot" -- not a combined figure. Same
--group-eval glob input as panel_group_summary.py.

Intra-subject and inter-subject/LOSO runs are NEVER combined into one bar -- different
subject counts and fold semantics (see tools/analysis/head_dataset_matrix.py). Split
into two figure sets by the dataset_mode's _intra/_inter suffix instead:
head_dataset_bars_<protocol>_{accuracy,kappa}.png."""
import os

from tools.analysis.group_summary import _locate, expand_glob_paths
from tools.analysis.head_dataset_matrix import build_matrix, order_heads
from tools.viz.bars import plot_grouped_bars

STAGES = frozenset({'finetune'})  # group_eval.json only ever comes from train_finetune.py
NEEDS_CHECKPOINT = False
NEEDS_DATASET = False


def _grid(matrix, heads, datasets):
    means = [[matrix.get((h, d), (float('nan'), float('nan'), 0))[0] for d in datasets] for h in heads]
    stds  = [[matrix.get((h, d), (float('nan'), float('nan'), 0))[1] for d in datasets] for h in heads]
    ns    = [[matrix.get((h, d), (float('nan'), float('nan'), 0))[2] for d in datasets] for h in heads]
    return means, stds, ns


def run(ctx):
    raw = ctx.args.group_eval
    if not raw:
        raise ValueError("panel 'head_dataset_bars' needs --group-eval <path/to/group_eval.json> "
                          "(repeatable; a glob like "
                          "'output/<backbone>/finetune/*/*/artifacts/group_eval.json' "
                          "expands to the whole baseline matrix)")
    paths = expand_glob_paths(raw)
    if not paths:
        raise ValueError(f"no group_eval.json files matched: {raw}")

    backbone_dir = _locate(paths[0])[0]
    out_dir = getattr(ctx.args, 'group_eval_out', None) or (
        os.path.join(backbone_dir, 'finetune', 'analysis') if backbone_dir else '.')
    os.makedirs(out_dir, exist_ok=True)

    for protocol, suffix in (('intra', '_intra'), ('inter', '_inter')):
        protocol_paths = [p for p in paths if _locate(p)[1].split('/', 1)[1].endswith(suffix)]
        if not protocol_paths:
            continue

        for metric, ylabel, tag in (('tail', 'Balanced accuracy (tail)', 'accuracy'),
                                     ('kappa_tail', "Cohen's Kappa (tail)", 'kappa')):
            matrix = build_matrix(protocol_paths, metric=metric)
            heads = order_heads({h for h, _ in matrix})
            datasets = sorted({d for _, d in matrix})
            means, stds, ns = _grid(matrix, heads, datasets)
            out_path = os.path.join(out_dir, f'head_dataset_bars_{protocol}_{tag}.png')
            plot_grouped_bars(out_path, datasets, heads, means, stds, ns, ylabel=ylabel,
                               title=f'Head x Dataset -- {tag} ({protocol})', gap_after={0})
            print(f"  [panel] -> {out_path}")
