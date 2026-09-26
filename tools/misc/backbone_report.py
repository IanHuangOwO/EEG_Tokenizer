"""
One report for a backbone comparison (e.g. base / A / B / AB): downstream scores on the frozen
finetune protocols (configs/sweeps/frozen_protocols.json), the backbone's own
held-out reconstruction and embedding use (tools/misc/backbone_eval.py via compare_backbones),
and a verdict per hypothesis. The hypotheses and thresholds below were fixed before the A/B/AB
results (experiment plan 2026-09-25); changing them after seeing numbers defeats the point.

Single pretrain seed per backbone: a difference inside +-3 balanced-accuracy points is a tie,
and p-values are over subjects (Holm-corrected across every group x cell test), not seeds.

    python -m tools.misc.backbone_report --ref base=mesae_tiny_static16_base_s1 \\
        --group A=mesae_tiny_static16_groupA_s1 --group B=mesae_tiny_static16_groupB_s1 \\
        --group AB=mesae_tiny_static16_groupAB_s1
Writes output/reports/backbone_report_<ref+groups>.md and prints it.
"""
import argparse
import contextlib
import io
import json
import os

import numpy as np
from scipy import stats

from tools.misc.compare_backbones import backbone as backbone_tables
from tools.misc.summarize_runs import load

CELLS = ['BNCI2014001_loso', 'BNCI2014001_fewshot', 'BNCI2014004_loso', 'BNCI2014004_fewshot',
         'BNCI2014008_loso', 'BNCI2014008_fewshot']
KEY_CELLS = ['BNCI2014001_loso', 'BNCI2014004_loso', 'BNCI2014004_fewshot']   # where a win must show
SPARSE_CELLS = ['BNCI2014004_loso', 'BNCI2014004_fewshot']                     # 3-channel cap
WIN, TIE = 2.0, 3.0            # points: a clear gain / the largest drop still counted as a tie
BIAS_RHO, BIAS_MAG = 0.5, 0.5  # a spatial-bias block is "used": closeness rho and mean |bias| above these
MSE_REL = 0.05                 # an MSE is "better" only when >= 5% lower (vs the reference, or vs IDW)


def scores(bb, head, cell):
    return load(f'output/{bb}/finetune/frozen_{head}/{cell}', 'tail') or {}


def paired(a, b):
    """a, b: {subject: score} -> (mean diff in points, 95% CI half-width, p, n) or None."""
    d = [100 * (a[s] - b[s]) for s in a if s in b]
    if len(d) < 3:
        return None
    return float(np.mean(d)), float(stats.sem(d) * stats.t.ppf(0.975, len(d) - 1)), \
        float(stats.ttest_1samp(d, 0).pvalue), len(d)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def backbone_eval(bb):
    p = f'output/{bb}/pretrain/analysis/backbone_eval.json'
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True, help='name=backbone')
    ap.add_argument('--group', action='append', required=True, help='name=backbone')
    args = ap.parse_args()
    ref_name, ref = args.ref.split('=')
    groups = dict(g.split('=') for g in args.group)
    allg = {ref_name: ref, **groups}
    out = []
    w = out.append

    w(f'# Backbone comparison: {", ".join(allg)}  (reference: {ref_name})\n')
    w(f'Single pretrain seed each. Tie band +-{TIE:g} points; "clear" gain >= +{WIN:g}. '
      'Downstream = tail balanced accuracy (%), mean over subjects, frozen protocols '
      '(mi_loso / mi_fewshot / p300_loso / p300_fewshot, tuned on DEV sets only).\n')

    # ---------- 1. downstream ----------
    s = {(g, h, c): scores(bb, h, c) for g, bb in allg.items() for h in ('learned',) for c in CELLS}
    missing = [f'{g}/{h}/{c}' for (g, h, c), v in s.items() if not v]
    w('## 1. Downstream (learned head)\n')
    w('| cell | ' + ' | '.join(allg) + ' |')
    w('|---|' + '---|' * len(allg))
    for c in CELLS:
        w(f'| {c} | ' + ' | '.join(f'{100 * np.mean(list(v.values())):.1f}' if (v := s[(g, "learned", c)]) else '-'
                                   for g in allg) + ' |')
    tests = [(g, c, paired(s[(g, 'learned', c)], s[(ref_name, 'learned', c)])) for g in groups for c in CELLS]
    tests = [t for t in tests if t[2]]
    adj = holm([t[2][2] for t in tests])
    delta = {(g, c): r[0] for (g, c, r) in tests}
    w(f'\nDifference vs {ref_name} (points, paired over subjects; p Holm-corrected over {len(tests)} tests):\n')
    w('| group | cell | diff | 95% CI | p | p (Holm) | n |')
    w('|---|---|---|---|---|---|---|')
    for (g, c, (m, ci, p, n)), pa in zip(tests, adj):
        w(f'| {g} | {c} | {m:+.1f} | [{m - ci:+.1f}, {m + ci:+.1f}] | {p:.3f} | {pa:.3f} | {n} |')

    # ---------- 3. backbone mechanism ----------
    evs = {g: backbone_eval(bb) for g, bb in allg.items()}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        backbone_tables({g: [bb] for g, bb in allg.items() if evs[g]})
    w('\n## 2. Backbone: held-out masked reconstruction and embedding use\n')
    w('Masked MSE on 512 held-out pretrain windows, identical masks for every backbone; '
      '"x baseline" = model / best model-free interpolation (<1 beats it). Ablation = MSE change '
      'when an embedding is destroyed (large = the backbone relies on it).\n')
    w('```' + buf.getvalue().rstrip() + '\n```')

    # ---------- 4. verdicts ----------
    def ev(g, key):
        e = evs.get(g)
        return e['test_masks'].get(key) if e else None

    def verdict(ok):
        return 'n/a' if ok is None else ('SUPPORTED' if ok else 'not supported')

    w('\n## 3. Hypotheses (fixed before the results)\n')
    w('| group | hypothesis | evidence | verdict |')
    w('|---|---|---|---|')
    summary = {}
    for g in groups:
        ds = [delta.get((g, c)) for c in CELLS]
        known = [d for d in ds if d is not None]
        key = [delta[(g, c)] for c in KEY_CELLS if (g, c) in delta]
        win = (max(key) >= WIN and min(known) >= -TIE) if key and known else None
        summary[g] = (win, float(np.mean(known)) if known else None)
        w(f'| {g} | wins overall: >= +{WIN:g} on a key cell ({", ".join(KEY_CELLS)}), no cell below -{TIE:g} | '
          f'best key {max(key):+.1f}, worst cell {min(known):+.1f}, mean {np.mean(known):+.1f} | {verdict(win)} |'
          if key and known else f'| {g} | wins overall | missing cells | n/a |')

        sp = [delta[(g, c)] for c in SPARSE_CELLS if (g, c) in delta]
        w(f'| {g} | helps the 3-channel cap (BNCI2014004) downstream | '
          + (' / '.join(f'{d:+.1f}' for d in sp) if sp else '-') + f' | {verdict(max(sp) >= WIN if sp else None)} |')

        ratios = {k: (ev(g, f'{k}|all|model') / ev(g, f'{k}|all|idw')) if ev(g, f'{k}|all|model') and ev(g, f'{k}|all|idw') else None
                  for k in ('random_channel', 'channel_cluster')}
        ok = None if None in ratios.values() else all(r <= 1 - MSE_REL for r in ratios.values())
        w(f'| {g} | beats IDW interpolation on whole-channel masks by >= {MSE_REL:.0%} | '
          + ', '.join(f'{k} {r:.2f}x' for k, r in ratios.items() if r) + f' | {verdict(ok)} |')

        m3, m3r = ev(g, 'motor3_to_bci22|all|model'), ev(ref_name, 'motor3_to_bci22|all|model')
        w(f'| {g} | sparse-to-dense imputation (motor3_to_bci22) >= {MSE_REL:.0%} better than {ref_name} | '
          + (f'{m3:.3f} vs {m3r:.3f}' if m3 and m3r else '-') + f' | {verdict(m3 <= (1 - MSE_REL) * m3r if m3 and m3r else None)} |')

        sparse = [(ev(g, f'{k}|sparse|model'), ev(ref_name, f'{k}|sparse|model')) for k in ('random_channel', 'channel_cluster')]
        ok = None if any(a is None or b is None for a, b in sparse) else all(a <= (1 - MSE_REL) * b for a, b in sparse)
        w(f'| {g} | channel masks on sparse caps >= {MSE_REL:.0%} better than {ref_name} | '
          + ', '.join(f'{a:.3f} vs {b:.3f}' for a, b in sparse if a and b) + f' | {verdict(ok)} |')

        sb = (evs.get(g) or {}).get('structure', {}).get('spatial_bias_per_block')
        if sb:
            used = [i for i, d in enumerate(sb) if d['closeness_spearman'] > BIAS_RHO and d['mean_abs'] > BIAS_MAG]
            w(f'| {g} | spatial bias is used (block rho > {BIAS_RHO:g} and mean abs > {BIAS_MAG:g}) | '
              f'{len(used)}/{len(sb)} blocks: {used} | {verdict(bool(used))} |')

        e = (evs.get(g) or {}).get('ablation_by_mask', {}).get('time_block')
        er = (evs.get(ref_name) or {}).get('ablation_by_mask', {}).get('time_block')
        if e and er:
            t, tr = e['time_shuffle'] / e['baseline'] - 1, er['time_shuffle'] / er['baseline'] - 1
            w(f'| {g} | (diagnostic) uses time under time_block masks | time_shuffle {t:+.0%} vs {ref_name} {tr:+.0%} | - |')

    if 'AB' in groups and 'A' in groups and 'B' in groups and all(summary[x][1] is not None for x in ('A', 'B', 'AB')):
        a, b, ab = summary['A'][1], summary['B'][1], summary['AB'][1]
        w(f'| AB | stacks: mean diff >= max(A, B) and AB wins overall | AB {ab:+.1f} vs A {a:+.1f} / B {b:+.1f} | '
          f'{verdict(ab >= max(a, b) and bool(summary["AB"][0]))} |')

    ranked = sorted(((m, g) for g, (_, m) in summary.items() if m is not None), reverse=True)
    w('\n## 4. Ranking by mean downstream difference vs ' + ref_name + '\n')
    for i, (m, g) in enumerate(ranked, 1):
        w(f'{i}. {g}: {m:+.1f} points, wins overall: {verdict(summary[g][0])}')
    if missing:
        w(f'\nMissing runs ({len(missing)}): ' + ', '.join(missing))

    text = '\n'.join(out) + '\n'
    os.makedirs('output/reports', exist_ok=True)
    path = f'output/reports/backbone_report_{"_".join(allg)}.md'
    open(path, 'w').write(text)
    print(text)
    print(f'-> {path}')


if __name__ == '__main__':
    main()
