"""Cohen's kappa (mean of the last 10 epochs, `kappa_tail`) of the three heads on the small corpus (3 pretrain seeds x
3 finetune seeds) against EEG-FM-Compass's kappa tables (appendix XVI / XVIII: BNCI2014001 / 004; BNCI2014008 has
none) -> table on stdout and kappa_small.png. Kappa in %, like Compass."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BB = ['mesae_small_p50_s16_s1', 'mesae_small_p50_s16_s2', 'mesae_small_p50_s16_s3']
DS = ['BNCI2014004', 'BNCI2014001', 'BNCI2014008']
HEADS = [('combined', 'Combined head', '#2a78d6'), ('patch_probe', 'z probe', '#eb6834'), ('cw_stamp', 'Stamp head', '#1baf7a')]
COMPASS = json.load(open('docs/reports/2026-09-29-combined-head/compass_appendix.json'))
REF = [('fm_linear', 'Best FM linear probe', '#e87ba4'), ('fm_full', 'Best FM full fine-tune', '#4a3aa7'),
       ('specialist', 'Best specialist', '#eda100')]

def ours(head, cell, key='kappa_tail'):
    per_bb = []
    for b in BB:
        s = []
        for seed in (1, 2, 3):
            d = json.load(open(f'output/{b}/finetune/{head}/{cell}_seed{seed}/artifacts/group_eval.json'))
            s.append(np.mean([v[key] for f in d.values() for g in f['groups'].values() for v in g['subjects'].values()]))
        per_bb.append(np.mean(s))
    v = np.array(per_bb)
    return v.mean() * 100, v.std(ddof=1) / np.sqrt(len(v)) * 100

SPLITS = (('loso', 'Cross-subject (loso)'), ('fewshot', 'Within-subject few-shot'))
res = {sp: {ds: {h: ours(h, f'{ds}_{sp}') for h, _, _ in HEADS} for ds in DS} for sp, _ in SPLITS}
for sp, name in SPLITS:
    print(f"\n{name}, Cohen's kappa\n")
    print('| Dataset | ' + ' | '.join(l for _, l, _ in HEADS + REF) + ' |\n|---|---|---|---|---|---|---|')
    for ds in DS:
        ref = COMPASS.get(ds, {}).get('kappa', {}).get(sp, {})
        print(f'| {ds} | ' + ' | '.join(f'{res[sp][ds][h][0]:.1f} ± {res[sp][ds][h][1]:.1f}' for h, _, _ in HEADS) + ' | '
              + ' | '.join(f'{ref[g][0]:.2f} ({ref[g][1]})' if g in ref else 'n/a' for g, _, _ in REF) + ' |')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x, w = np.arange(len(DS)), 0.13
SERIES = [(h, l, c, True) for h, l, c in HEADS] + [(g, l, c, False) for g, l, c in REF]
for ax, (sp, name) in zip(axes, SPLITS):
    for i, (h, label, col, own) in enumerate(SERIES):
        xs = x + (i - (len(SERIES) - 1) / 2) * w
        if own:
            vals = [res[sp][ds][h][0] for ds in DS]; err = [res[sp][ds][h][1] for ds in DS]
        else:
            ref = [COMPASS.get(ds, {}).get('kappa', {}).get(sp, {}).get(h) for ds in DS]
            keep = [j for j, r in enumerate(ref) if r]
            xs, vals, err = xs[keep], [ref[j][0] for j in keep], None
        ax.bar(xs, vals, w, color=col, edgecolor='#fcfcfb', linewidth=1.5, label=label, zorder=2)
        if err is not None:
            ax.errorbar(xs, vals, yerr=err, fmt='none', ecolor='#0b0b0b', elinewidth=1, capsize=2, zorder=3)
        for xi, v, e in zip(xs, vals, err if err is not None else [0] * len(vals)):
            ax.text(xi, v + e + 1.0, f'{v:.1f}', ha='center', va='bottom', fontsize=5.8, color='#0b0b0b', rotation=90)
    ax.set_xticks(x, DS)
    ax.set_ylim(0, 70)
    ax.set_ylabel("Cohen's kappa (%)", color='#52514e')
    ax.set_title(name, fontsize=11)
    ax.grid(axis='y', color='#e6e5e0', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#b5b4ad')
    ax.tick_params(colors='#52514e')
    ax.set_facecolor('#fcfcfb')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False, fontsize=8.5)
fig.text(0.5, 0.005, 'Frozen small-corpus backbones: mean ± SE over 3 pretrain seeds (each the mean of 3 finetune seeds; mean of '
         'last 10 epochs). 0 = chance. Compass: appendix Tables XVI / XVIII (last epoch); BNCI2014008 has no Compass kappa.', ha='center', fontsize=8, color='#52514e')
fig.patch.set_facecolor('#fcfcfb')
fig.tight_layout(rect=(0, 0.03, 1, 0.92))
fig.savefig('docs/reports/2026-09-29-combined-head/kappa_small.png', dpi=130)
