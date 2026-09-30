"""Small-corpus heads (3 pretrain seeds x 3 finetune seeds) vs EEG-FM-Compass best entries (Tables V-VI) -> table on
stdout and compass_small.png (grouped bars, loso and few-shot)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BB = ['mesae_small_p50_s16_s1', 'mesae_small_p50_s16_s2', 'mesae_small_p50_s16_s3']
DS = ['BNCI2014004', 'BNCI2014001', 'BNCI2014008']
COMPASS = {  # (best FM linear probe, best FM full fine-tune, best specialist), docs/papers/2601.17883_EEG-FM-Compass.md
    'loso': {'BNCI2014004': ((75.57, 'Neuro-GPT'), (77.70, 'Neuro-GPT'), (76.38, 'EEGNet')),
             'BNCI2014001': ((48.24, 'Neuro-GPT'), (53.03, 'CBraMod'), (46.80, 'LMDA')),
             'BNCI2014008': ((67.11, 'BENDR'), (69.91, 'CBraMod'), (72.29, 'EEGNet'))},
    'fewshot': {'BNCI2014004': ((76.69, 'Neuro-GPT'), (77.39, 'CBraMod'), (80.17, 'Conformer')),
                'BNCI2014001': ((49.82, 'Neuro-GPT'), (50.34, 'CBraMod'), (60.62, 'classical ML')),
                'BNCI2014008': ((61.45, 'EEGMamba'), (61.61, 'Neuro-GPT'), (70.91, 'EEGNet'))}}

def ours(head, cell):
    per_bb = []
    for b in BB:
        s = []
        for seed in (1, 2, 3):
            d = json.load(open(f'output/{b}/finetune/{head}/{cell}_seed{seed}/artifacts/group_eval.json'))
            s.append(np.mean([v['tail'] for f in d.values() for g in f['groups'].values() for v in g['subjects'].values()]) * 100)
        per_bb.append(np.mean(s))
    v = np.array(per_bb)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

res = {sp: {ds: {h: ours(h, f'{ds}_{sp}') for h in ('combined', 'patch_probe', 'cw_stamp')} for ds in DS} for sp in COMPASS}
for sp, name in (('loso', 'Cross-subject (loso)'), ('fewshot', 'Within-subject few-shot')):
    print(f'\n{name}, balanced accuracy (%)\n')
    print('| Dataset | Ours: combined | Ours: z probe | Ours: stamp head | Best FM linear probe | Best FM full fine-tune | Best specialist |')
    print('|---|---|---|---|---|---|---|')
    for ds in DS:
        o = res[sp][ds]
        c = COMPASS[sp][ds]
        print(f'| {ds} | {o["combined"][0]:.1f} ± {o["combined"][1]:.1f} | {o["patch_probe"][0]:.1f} ± {o["patch_probe"][1]:.1f} | '
              f'{o["cw_stamp"][0]:.1f} ± {o["cw_stamp"][1]:.1f} | ' + ' | '.join(f'{v:.2f} ({m})' for v, m in c) + ' |')

SERIES = [('Ours: combined head (frozen, small corpus)', '#2a78d6'), ('Best FM linear probe', '#eb6834'),
          ('Best FM full fine-tune', '#1baf7a'), ('Best specialist', '#eda100')]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
x, w = np.arange(len(DS)), 0.2
for ax, (sp, name) in zip(axes, (('loso', 'Cross-subject (loso)'), ('fewshot', 'Within-subject few-shot'))):
    for i, (label, col) in enumerate(SERIES):
        if i == 0:
            vals = [res[sp][ds]['combined'][0] for ds in DS]; err = [res[sp][ds]['combined'][1] for ds in DS]
        else:
            vals = [COMPASS[sp][ds][i - 1][0] for ds in DS]; err = None
        xs = x + (i - 1.5) * w
        ax.bar(xs, vals, w, color=col, edgecolor='#fcfcfb', linewidth=2, label=label, zorder=2)
        if err is not None:
            ax.errorbar(xs, vals, yerr=err, fmt='none', ecolor='#0b0b0b', elinewidth=1, capsize=3, zorder=3)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.6, f'{v:.1f}', ha='center', va='bottom', fontsize=7.5, color='#0b0b0b')
    ax.set_xticks(x, DS)
    ax.set_ylim(30, 85)
    ax.set_ylabel('balanced accuracy (%)', color='#52514e')
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
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize=9)
fig.text(0.5, 0.005, 'Ours: mean ± SE over 3 small-corpus pretrain seeds (each the mean of 3 finetune seeds; mean of last 10 epochs). '
         'Compass: EEG-FM-Compass Tables V-VI (last epoch). y-axis starts at 30%.', ha='center', fontsize=8, color='#52514e')
fig.patch.set_facecolor('#fcfcfb')
fig.tight_layout(rect=(0, 0.03, 1, 0.92))
fig.savefig('docs/reports/2026-09-29-combined-head/compass_small.png', dpi=130)
