"""Cross-head x cross-dataset accuracy/kappa matrix for panel_head_dataset_bars.py and
panel_protocol_gap.py -- built from the same group_eval.json files group_summary.py
already loads (_locate/load), just grouped by (head, dataset_mode) instead of
pooled/paired against one reference head."""
import numpy as np

from tools.analysis.group_summary import _locate, load


def order_heads(heads):
    """raw_signal first, raw_band second (both pulled to the front, in that order),
    everything else alphabetical after. Pair with plot_grouped_bars' gap_after={0} --
    raw_signal stays visually set apart with a wider gap, while raw_band (easy to
    mistake for raw_signal at a glance) sits right next to it instead of alphabetized
    away among the rest. Shared by panel_head_dataset_bars.py/panel_protocol_gap.py so
    their bar order stays identical."""
    heads = sorted(heads)
    for name in ('raw_band', 'raw_signal'):
        if name in heads:
            heads.remove(name)
            heads.insert(0, name)
    return heads


def build_matrix(paths, metric='tail'):
    """-> {(head, dataset_mode): (mean, std, n)}. dataset_mode keeps its _intra/_inter
    suffix (_locate's label is '<head>/<dataset>_<mode>') -- callers that need to split
    or pair across protocols do it themselves (see build_protocol_gap below). n is the
    number of pooled subject/fold readings -- annotate it on every bar, see
    tools/viz/bars.py's plot_grouped_bars docstring: fold counts range 5-16x across this
    repo's real datasets, an error bar alone doesn't show that.

    A (head, dataset_mode) whose metric is missing everywhere (e.g. kappa_tail on a
    group_eval.json written before kappa was tracked) gets (nan, nan, 0) rather than
    being dropped from the dict -- callers can tell "ran but no kappa recorded" apart
    from "never ran" by whether the key exists at all."""
    matrix = {}
    for path in paths:
        _, label = _locate(path)
        head, dataset_mode = label.split('/', 1)
        _, pooled = load(path, metric=metric)
        vals = [v for g in pooled.values() for v in g.values()]
        if vals:
            arr = np.array(vals)
            matrix[(head, dataset_mode)] = (
                float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, len(arr))
        else:
            matrix[(head, dataset_mode)] = (float('nan'), float('nan'), 0)
    return matrix


def build_protocol_gap(paths, metric='tail'):
    """-> {(head, dataset): (intra_mean - inter_mean, min(n_intra, n_inter))}, for every
    (head, dataset) pair that has BOTH an _intra and _inter group_eval.json among paths
    -- a pair with only one protocol contributes no gap (not (nan, 0); simply absent).
    dataset here has the _intra/_inter suffix stripped (the pairing key). Positive delta
    = intra-subject scored higher than LOSO -- the subject-transfer cost."""
    matrix = build_matrix(paths, metric=metric)
    by_dataset = {}
    for (head, dataset_mode), (mean, _std, n) in matrix.items():
        if dataset_mode.endswith('_intra'):
            by_dataset.setdefault((head, dataset_mode[:-len('_intra')]), {})['intra'] = (mean, n)
        elif dataset_mode.endswith('_inter'):
            by_dataset.setdefault((head, dataset_mode[:-len('_inter')]), {})['inter'] = (mean, n)

    gap = {}
    for key, d in by_dataset.items():
        if 'intra' in d and 'inter' in d:
            (m_intra, n_intra), (m_inter, n_inter) = d['intra'], d['inter']
            if not (np.isnan(m_intra) or np.isnan(m_inter)):
                gap[key] = (m_intra - m_inter, min(n_intra, n_inter))
    return gap
