"""Overnight step 2: pick the tiny winner, finest-skip vs graded drop-path (the no-trial run), by the
mechanism card set on 2026-09-28 before the finest run. Spread = graded vs rankfix (near-identical
runs). Writes output/reports/overnight/decision.md and prints the winner's run name on the last line.

Card: keep finest-only skips if
  1. channel masks (random_channel, channel_cluster; all 5 bands): mean relative band error change
     vs no-trial < -max(3 x spread, 5%)          (imputation recovers)
  2. unmasked (all bands): the same rule            (visible detail recovers)
  3. time_block masked MSE change < +max(3 x spread, 3%)   (the deep path's time-gap gain is kept)
"""
import json
import os

A = 'output/{}/pretrain/analysis/backbone_eval.json'
F, N, G, R = 'mesae_tiny_finestskip_s1', 'mesae_tiny_notrial_s1', 'mesae_tiny_skipdrop_graded_s1', 'mesae_tiny_rankfix_s1'
BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ev = {r: json.load(open(A.format(r))) for r in (F, N, G, R)}


def band_change(a, b, masks):
    """mean over masks x bands of error_ratio(a) / error_ratio(b) - 1"""
    ch = [ev[a]['masked_spectrum'][m]['error_ratio'][x] / ev[b]['masked_spectrum'][m]['error_ratio'][x] - 1
          for m in masks for x in BANDS]
    return sum(ch) / len(ch), sum(abs(c) for c in ch) / len(ch)


def mse(r, k):
    return ev[r]['test_masks'][f'{k}|all|model']


chan, unm = ['random_channel', 'channel_cluster'], ['unmasked']
c_f, _ = band_change(F, N, chan)
_, s_c = band_change(G, R, chan)
u_f, _ = band_change(F, N, unm)
_, s_u = band_change(G, R, unm)
t_f = mse(F, 'time_block') / mse(N, 'time_block') - 1
s_t = abs(mse(G, 'time_block') / mse(R, 'time_block') - 1)
tests = [
    ('channel-mask band error vs no-trial', c_f, f'< {-max(3 * s_c, 0.05):+.1%}', c_f < -max(3 * s_c, 0.05)),
    ('unmasked band error vs no-trial', u_f, f'< {-max(3 * s_u, 0.05):+.1%}', u_f < -max(3 * s_u, 0.05)),
    ('time-block masked MSE vs no-trial', t_f, f'< {max(3 * s_t, 0.03):+.1%}', t_f < max(3 * s_t, 0.03)),
]
winner = F if all(t[3] for t in tests) else N
os.makedirs('output/reports/overnight', exist_ok=True)
lines = ['# Overnight decision: finest-only skips vs graded skip drop-path (tiny corpus)', '',
         'Card set 2026-09-28 before the finest run (docs/adr/0020). Spread = graded vs rankfix.', '',
         '| Test | Change | Threshold | Pass |', '|---|---|---|---|']
lines += [f'| {n} | {c:+.1%} | {th} | {"yes" if ok else "no"} |' for n, c, th, ok in tests]
lines += ['', f'Spreads: channel band error {s_c:.1%}, unmasked band error {s_u:.1%}, time-block MSE {s_t:.1%}.', '',
          '| Run | masked MSE token / channel / cluster / time | seam | ridge 004 / 001 / 008 |', '|---|---|---|---|']
for r in (F, N, G, R, 'archive/mesae_tiny_baseline_s1'):
    p = A.format(r)
    if not os.path.exists(p):
        continue
    e = json.load(open(p))
    rp = f'output/{r}/pretrain/analysis/ridge_probe.json'
    ridge = ' / '.join(f'{v["mean"] * 100:.1f}' for v in json.load(open(rp)).values()) if os.path.exists(rp) else '-'
    lines.append(f'| {r} | ' + ' / '.join(f"{e['test_masks'][f'{k}|all|model']:.3f}" for k in
                 ('token_runs', 'random_channel', 'channel_cluster', 'time_block'))
                 + f" | {e['seam_disagreement']:.3f} | {ridge} |")
lines += ['', f'**Winner: `{winner}`** ({"all three tests pass" if winner == F else "not all tests pass: graded drop-path stays"}).']
open('output/reports/overnight/decision.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(winner)
