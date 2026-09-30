"""Small-corpus heads (3 pretrain seeds x 3 finetune seeds) vs EEG-FM-Compass best entries (appendix XV / XVII, Tables V-VI for 008) -> table on
stdout and accuracy_small.png (grouped bars, loso and few-shot)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BB = ['mesae_small_p50_s16_s1', 'mesae_small_p50_s16_s2', 'mesae_small_p50_s16_s3']
DS = ['BNCI2014004', 'BNCI2014001', 'BNCI2014008']
COMPASS = {  # (best FM linear probe, best FM full fine-tune, best specialist). BNCI2014001 / 004: the per-dataset appendix
    # tables (XV, XVII; compass_appendix.py), which include MIRepNet, an MI-specific FM absent from the Table V-VI summary.
    # BNCI2014008: Tables V-VI (docs/papers/2601.17883_EEG-FM-Compass.md).
    'loso': {'BNCI2014004': ((79.40, 'MIRepNet'), (78.41, 'MIRepNet'), (76.38, 'EEGNet')),
             'BNCI2014001': ((50.48, 'MIRepNet'), (54.21, 'MIRepNet'), (46.80, 'LMDA')),
             'BNCI2014008': ((67.11, 'BENDR'), (69.91, 'CBraMod'), (72.29, 'EEGNet'))},
    'fewshot': {'BNCI2014004': ((76.82, 'MIRepNet'), (81.10, 'MIRepNet'), (80.17, 'Conformer')),
                'BNCI2014001': ((49.82, 'Neuro-GPT'), (63.27, 'MIRepNet'), (60.62, 'CSP+LDA')),
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

OWN = [('combined', 'Combined head', '#2a78d6'), ('patch_probe', 'z probe', '#eb6834'), ('cw_stamp', 'Stamp head', '#1baf7a')]
REF = [('Best FM linear probe', '#e87ba4'), ('Best FM full fine-tune', '#4a3aa7'), ('Best specialist', '#eda100')]
SERIES = [(h, l, c, True) for h, l, c in OWN] + [(j, l, c, False) for j, (l, c) in enumerate(REF)]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
x, w = np.arange(len(DS)), 0.13
for ax, (sp, name) in zip(axes, (('loso', 'Cross-subject (loso)'), ('fewshot', 'Within-subject few-shot'))):
    for i, (key, label, col, own) in enumerate(SERIES):
        xs = x + (i - (len(SERIES) - 1) / 2) * w
        if own:
            vals = [res[sp][ds][key][0] for ds in DS]; err = [res[sp][ds][key][1] for ds in DS]
        else:
            vals = [COMPASS[sp][ds][key][0] for ds in DS]; err = None
        ax.bar(xs, vals, w, color=col, edgecolor='#fcfcfb', linewidth=1.5, label=label, zorder=2)
        if err is not None:
            ax.errorbar(xs, vals, yerr=err, fmt='none', ecolor='#0b0b0b', elinewidth=1, capsize=2, zorder=3)
        for xi, v, e in zip(xs, vals, err if err is not None else [0] * len(vals)):
            ax.text(xi, v + e + 1.0, f'{v:.1f}', ha='center', va='bottom', fontsize=5.8, color='#0b0b0b', rotation=90)
    ax.set_xticks(x, DS)
    ax.set_ylim(30, 85)
    ax.set_ylabel('balanced accuracy (%)', color='#52514e')
    ax.set_title(name, fontsize=11)
    ax.grid(axis='y', color='#e6e5e0', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sd in ('top', 'right'):
        ax.spines[sd].set_visible(False)
    for sd in ('left', 'bottom'):
        ax.spines[sd].set_color('#b5b4ad')
    ax.tick_params(colors='#52514e')
    ax.set_facecolor('#fcfcfb')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False, fontsize=8.5)
fig.text(0.5, 0.005, 'Frozen small-corpus backbones: mean ± SE over 3 pretrain seeds (each the mean of 3 finetune seeds; mean of last 10 epochs). y-axis starts at 30%.\n'
         'Compass (last epoch): appendix Tables XV / XVII for BNCI2014001 / 004 (incl. MIRepNet), Tables V-VI for BNCI2014008.',
         ha='center', fontsize=8, color='#52514e')
fig.patch.set_facecolor('#fcfcfb')
fig.tight_layout(rect=(0, 0.06, 1, 0.92))
fig.savefig('docs/reports/2026-09-29-combined-head/accuracy_small.png', dpi=130)
