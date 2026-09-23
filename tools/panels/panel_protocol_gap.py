"""protocol_gap panel: intra-subject minus inter-subject/LOSO tail-accuracy (and kappa)
delta, per head per dataset -- positive = subject transfer cost (LOSO scored lower)
(tools/viz/bars.py's plot_grouped_bars, driven by tools/analysis/head_dataset_matrix.py's
build_protocol_gap). Only a (head, dataset) pair with BOTH an _intra and _inter
group_eval.json contributes a bar -- a head/dataset run under only one protocol is
silently skipped, no gap to compute. Usable today on this repo's pre-fixed-roster
dual-protocol leftovers (several datasets already have both). Same --group-eval glob
input as panel_group_summary.py/panel_head_dataset_bars.py."""
import os

from tools.analysis.group_summary import _locate, expand_glob_paths
from tools.analysis.head_dataset_matrix import build_protocol_gap, order_heads
from tools.viz.bars import plot_grouped_bars

STAGES = frozenset({'finetune'})  # group_eval.json only ever comes from train_finetune.py
NEEDS_CHECKPOINT = False
NEEDS_DATASET = False


def run(ctx):
    raw = ctx.args.group_eval
    if not raw:
        raise ValueError("panel 'protocol_gap' needs --group-eval <path/to/group_eval.json> "
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

    for metric, ylabel, tag in (('tail', 'Intra - LOSO balanced accuracy (tail)', 'accuracy'),
                                 ('kappa_tail', "Intra - LOSO Cohen's Kappa (tail)", 'kappa')):
        gap = build_protocol_gap(paths, metric=metric)
        if not gap:
            print(f"  [panel] protocol_gap: no head/dataset pair has both intra and inter "
                  f"for metric={metric}, skipping")
            continue
        heads = order_heads({h for h, _ in gap})
        datasets = sorted({d for _, d in gap})
        means = [[gap.get((h, d), (float('nan'), 0))[0] for d in datasets] for h in heads]
        stds = [[0.0 for _ in datasets] for _ in heads]
        ns = [[gap.get((h, d), (float('nan'), 0))[1] for d in datasets] for h in heads]
        out_path = os.path.join(out_dir, f'protocol_gap_{tag}.png')
        plot_grouped_bars(out_path, datasets, heads, means, stds, ns, ylabel=ylabel,
                           title='Intra-subject vs LOSO transfer cost, per head', gap_after={0})
        print(f"  [panel] -> {out_path}")
