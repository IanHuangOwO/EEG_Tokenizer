"""
The backbone comparison report (analysis_finetune.py's `report` panel): downstream scores on the frozen
finetune protocols, the backbones' own held-out reconstruction and embedding use (backbone_eval.json,
written by analysis_pretrain.py's `backbone_eval` panel), and a verdict per hypothesis. The hypotheses
and thresholds below were fixed before the A/B/AB results; changing them after seeing numbers defeats
the point. Single pretrain seed per backbone: a difference inside +-3 balanced-accuracy points is a tie,
and p-values are over subjects (Holm-corrected across every group x cell test), not seeds.
"""
import contextlib
import io
import json
import os

import numpy as np
from scipy import stats

from tools.analysis.summarize_runs import load
from tools.analysis import QTOME_OUTPUT

CELLS = ['BNCI2014001_loso', 'BNCI2014001_fewshot', 'BNCI2014004_loso', 'BNCI2014004_fewshot',
         'BNCI2014008_loso', 'BNCI2014008_fewshot']
KEY_CELLS = ['BNCI2014001_loso', 'BNCI2014004_loso', 'BNCI2014004_fewshot']   # where a win must show
SPARSE_CELLS = ['BNCI2014004_loso', 'BNCI2014004_fewshot']                     # 3-channel cap
WIN, TIE = 2.0, 3.0            # points: a clear gain / the largest drop still counted as a tie
BIAS_RHO, BIAS_MAG = 0.5, 0.5  # a spatial-bias block is "used": closeness rho and mean |bias| above these
MSE_REL = 0.05                 # an MSE is "better" only when >= 5% lower (vs the reference, or vs IDW)


def scores(bb, head, cell):
    return load(f'{QTOME_OUTPUT}/{bb}/finetune/{head}/{cell}', 'tail') or {}


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
    p = f'{QTOME_OUTPUT}/{bb}/pretrain/analysis/backbone_eval.json'
    return json.load(open(p)) if os.path.exists(p) else None


def report(allg, ref_name, head='frozen_learned', tables=None):
    """allg: {group name: backbone}, ref_name one of them; head: the finetune label compared
    (output/<backbone>/finetune/<head>/<cell>). -> the report as markdown text. A `tables` dict gets the
    numbers as CSV-ready row lists: report_tests (paired differences) and report_verdicts."""
    groups = {g: bb for g, bb in allg.items() if g != ref_name}
    out = []
    w = out.append

    w(f'# Backbone comparison: {", ".join(allg)}  (reference: {ref_name})\n')
    w(f'Single pretrain seed each. Tie band +-{TIE:g} points; "clear" gain >= +{WIN:g}. '
      'Downstream = tail balanced accuracy (%), mean over subjects, frozen protocols '
      '(mi_loso / mi_fewshot / p300_loso / p300_fewshot, tuned on DEV sets only).\n')

    # ---------- 1. downstream ----------
    s = {(g, h, c): scores(bb, h, c) for g, bb in allg.items() for h in (head,) for c in CELLS}
    missing = [f'{g}/{h}/{c}' for (g, h, c), v in s.items() if not v]
    w(f'## 1. Downstream ({head})\n')
    w('| cell | ' + ' | '.join(allg) + ' |')
    w('|---|' + '---|' * len(allg))
    for c in CELLS:
        w(f'| {c} | ' + ' | '.join(f'{100 * np.mean(list(v.values())):.1f}' if (v := s[(g, head, c)]) else '-'
                                   for g in allg) + ' |')
    tests = [(g, c, paired(s[(g, head, c)], s[(ref_name, head, c)])) for g in groups for c in CELLS]
    tests = [t for t in tests if t[2]]
    adj = holm([t[2][2] for t in tests])
    delta = {(g, c): r[0] for (g, c, r) in tests}
    if tables is not None:
        tables['report_tests'] = [{'group': g, 'ref': ref_name, 'head': head, 'cell': c, 'diff': m, 'ci_lo': m - ci,
                                   'ci_hi': m + ci, 'p': p, 'p_holm': pa, 'n': n}
                                  for (g, c, (m, ci, p, n)), pa in zip(tests, adj)]
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

    if tables is not None:
        sec = out[out.index('\n## 3. Hypotheses (fixed before the results)\n') + 3:]
        tables['report_verdicts'] = [dict(zip(('group', 'hypothesis', 'evidence', 'verdict'),
                                              (x.strip() for x in line.strip('|').split(' | '))))
                                     for line in sec if line.startswith('| ') and not line.startswith('| group |')]
    return '\n'.join(out) + '\n'


def backbone_tables(groups):
    """Held-out reconstruction (test masks vs interpolation baselines), per-mask embedding ablations and
    structure, per group of backbones (averaged over a group's seeds), from each backbone_eval.json."""
    print('\n## Backbone eval (held-out windows, identical masks; lower MSE is better)')
    evals = {g: [json.load(open(p)) for b in bbs
                 if os.path.exists(p := f'{QTOME_OUTPUT}/{b}/pretrain/analysis/backbone_eval.json')] for g, bbs in groups.items()}
    kinds = ['token_runs', 'random_channel', 'channel_cluster', 'time_block', 'motor3_to_bci22']
    print(f'  {"test mask":17} {"windows":7}' + ''.join(f' | {g:>22}' for g in evals))
    for k in kinds:
        for grp in ('all', 'sparse', 'dense'):
            row, any_ms = f'  {k if grp == "all" else "":17} {grp:7}', False
            for g, ev in evals.items():
                ms = [m for e in ev if (m := e['test_masks'].get(f'{k}|{grp}|model')) is not None]
                base = [min(v for p in ('idw', 'linear') if (v := e['test_masks'].get(f'{k}|{grp}|{p}')) is not None)
                        for e in ev if any(e['test_masks'].get(f'{k}|{grp}|{p}') is not None for p in ('idw', 'linear'))]
                any_ms |= bool(ms)
                row += (f' | {np.mean(ms):.3f} ({np.mean(ms) / np.mean(base):.2f}x baseline)' if ms and base else
                        f' | {np.mean(ms):>22.3f}' if ms else f' | {"-":>22}')
            if any_ms:
                print(row)
    for a in ('coords_shuffle', 'coords_mean', 'time_shuffle', 'time_const'):
        print(f'  ablation {a:15}' + ''.join(
            f' | {np.mean([e["ablation_masked_mse"][a] / e["ablation_masked_mse"]["baseline"] - 1 for e in ev]):>+21.0%}'
            if ev else f' | {"-":>22}' for ev in evals.values()))
    if all(ev and 'ablation_by_mask' in ev[0] for ev in evals.values()):
        print('\n## Embedding ablation, masked MSE change vs intact embeddings, per test mask')
        for a in ('coords_shuffle', 'coords_mean', 'time_shuffle', 'time_const'):
            print(f'  {a}')
            for k in kinds:
                print(f'    {k:17}' + ''.join(
                    f' | {np.mean([e["ablation_by_mask"][k][a] / e["ablation_by_mask"][k]["baseline"] - 1 for e in ev]):>+21.0%}'
                    for ev in evals.values()))
    print('  coord sim ~ closeness   ' + ''.join(
        f' | {np.mean(v):>22.3f}' if (v := [e['structure']['coord_sim_vs_closeness_spearman'] for e in ev
                                            if 'coord_sim_vs_closeness_spearman' in e['structure']]) else f' | {"-":>22}'
        for ev in evals.values()))
    for g, ev in evals.items():
        sb = [e['structure'].get('spatial_bias_per_block') for e in ev if e['structure'].get('spatial_bias_per_block')]
        if sb:
            rho = np.mean([[d['closeness_spearman'] for d in s] for s in sb], 0)
            mag = np.mean([[d['mean_abs'] for d in s] for s in sb], 0)
            print(f'  {g} spatial bias per block (closeness rho / mean|b|): '
                  + ' '.join(f'{r:+.2f}/{m:.2f}' for r, m in zip(rho, mag)))




def backbone_eval_rows(allg):
    """Each group's backbone_eval.json as CSV-ready rows: (test-mask MSEs: group, test_mask, windows,
    predictor, mse; ablations: group, test_mask, ablation, mse, change vs intact; structure: group, key,
    value, with the spatial bias one row per block)."""
    masks, abl, struct = [], [], []
    for g, bb in allg.items():
        e = backbone_eval(bb)
        if not e:
            continue
        for k, v in e['test_masks'].items():
            kind, win, pred = k.split('|')
            masks.append({'group': g, 'backbone': bb, 'test_mask': kind, 'windows': win, 'predictor': pred, 'mse': v})
        for kind, d in e.get('ablation_by_mask', {'all': e['ablation_masked_mse']}).items():
            for a, v in d.items():
                abl.append({'group': g, 'backbone': bb, 'test_mask': kind, 'ablation': a, 'mse': v,
                            'change': v / d['baseline'] - 1})
        for k, v in e['structure'].items():
            if isinstance(v, list):
                struct += [{'group': g, 'backbone': bb, 'key': f'{k}[{i}].{kk}', 'value': vv}
                           for i, d in enumerate(v) for kk, vv in d.items()]
            else:
                struct.append({'group': g, 'backbone': bb, 'key': k, 'value': v})
    return masks, abl, struct
