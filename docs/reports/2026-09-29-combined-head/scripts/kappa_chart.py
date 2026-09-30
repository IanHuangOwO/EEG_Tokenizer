"""Cohen's kappa (mean of the last 10 epochs, `kappa_tail`) of the three heads on the small corpus (3 pretrain seeds x
3 finetune seeds) -> table on stdout and kappa_small.png. EEG-FM-Compass reports balanced accuracy only, so there is no
Compass kappa to compare against."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BB = ['mesae_small_p50_s16_s1', 'mesae_small_p50_s16_s2', 'mesae_small_p50_s16_s3']
DS = ['BNCI2014004', 'BNCI2014001', 'BNCI2014008']
HEADS = [('combined', 'Combined head', '#2a78d6'), ('patch_probe', 'z probe', '#eb6834'), ('cw_stamp', 'Stamp head', '#1baf7a')]

def ours(head, cell, key='kappa_tail'):
    per_bb = []
    for b in BB:
        s = []
        for seed in (1, 2, 3):
            d = json.load(open(f'output/{b}/finetune/{head}/{cell}_seed{seed}/artifacts/group_eval.json'))
            s.append(np.mean([v[key] for f in d.values() for g in f['groups'].values() for v in g['subjects'].values()]))
        per_bb.append(np.mean(s))
    v = np.array(per_bb)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

SPLITS = (('loso', 'Cross-subject (loso)'), ('fewshot', 'Within-subject few-shot'))
res = {sp: {ds: {h: ours(h, f'{ds}_{sp}') for h, _, _ in HEADS} for ds in DS} for sp, _ in SPLITS}
for sp, name in SPLITS:
    print(f"\n{name}, Cohen's kappa\n")
    print('| Dataset | ' + ' | '.join(l for _, l, _ in HEADS) + ' |\n|---|---|---|---|')
    for ds in DS:
        print(f'| {ds} | ' + ' | '.join(f'{res[sp][ds][h][0]:.3f} ± {res[sp][ds][h][1]:.3f}' for h, _, _ in HEADS) + ' |')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
x, w = np.arange(len(DS)), 0.26
for ax, (sp, name) in zip(axes, SPLITS):
    for i, (h, label, col) in enumerate(HEADS):
        vals = [res[sp][ds][h][0] for ds in DS]; err = [res[sp][ds][h][1] for ds in DS]
        xs = x + (i - 1) * w
        ax.bar(xs, vals, w, color=col, edgecolor='#fcfcfb', linewidth=2, label=label, zorder=2)
        ax.errorbar(xs, vals, yerr=err, fmt='none', ecolor='#0b0b0b', elinewidth=1, capsize=3, zorder=3)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=7.5, color='#0b0b0b')
    ax.set_xticks(x, DS)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Cohen's kappa", color='#52514e')
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
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=9)
fig.text(0.5, 0.005, 'Frozen small-corpus backbones: mean ± SE over 3 pretrain seeds (each the mean of 3 finetune seeds; mean of '
         'last 10 epochs). 0 = chance. EEG-FM-Compass reports no kappa.', ha='center', fontsize=8, color='#52514e')
fig.patch.set_facecolor('#fcfcfb')
fig.tight_layout(rect=(0, 0.03, 1, 0.92))
fig.savefig('docs/reports/2026-09-29-combined-head/kappa_small.png', dpi=130)
