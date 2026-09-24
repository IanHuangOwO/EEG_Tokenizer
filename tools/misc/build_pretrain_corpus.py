"""
Builds dataset_params.pretrain for the balanced pretrain corpus (2026-09-24 design): motor
(MI + motor execution) ~50%, the other paradigms ~12.5% each, whole subjects only.

ALLOC below is the full corpus in hours per dataset. For --fraction f, each dataset gets
f * its allocation. Subjects are taken in one seeded order per dataset, adding whole
subjects while that brings the total closer to the target, so a smaller fraction is always a
prefix (a subset) of a larger one. A dataset whose target is under half a subject is left
out of that fraction.

    python -m tools.misc.build_pretrain_corpus --fraction 1.0  --out configs/pretrain.template.json
    python -m tools.misc.build_pretrain_corpus --fraction 0.05 --out configs/pretrain_tiny5.template.json

--out must already exist (a pretrain config); only its dataset_params.pretrain is replaced.
"""
import argparse
import glob
import json
import os
import random

import numpy as np

# paradigm -> {dataset: full-corpus hours}; None = every compiled subject
ALLOC = {
    'motor imagery': {'Lee2019_MI': 25, 'Dreyer2023': 25, 'Cho2017': None, 'Weibo2014': None},
    'motor execution': {'Schirrmeister2017': None, 'GraspAndLift_Train': None, 'GraspAndLift_Test': None},
    'SSVEP': {'Lee2019_SSVEP': 9, 'BETA_4s': 4.5, 'BETA_3s': 1.5, 'Wang2016': 6, 'Liu2022EldBETA': 6},
    'ERP': {'ERP_Longitudinal': None, 'Inria_Train': None, 'Inria_Test': None},
    'rest / clinical': {'SRM_RestingState': None, 'MDD_Mumtaz': 7, 'Neonatal_Helsinki': 7,
                        'UCSD_PD': None, 'SPIS': None},
    'cognitive / affective': {'BCMI_MusicEmotion': 18, 'BCIC2020-3': None, 'STEW': None},
}
SEED = 42
SUFFIX = 'fs200_bp0.5-100.0_pre1_post4'


def subject_hours(ds):
    out = {}
    for f in glob.glob(f'datas/pretrain/{ds}/cache/*_{SUFFIX}.npz'):
        z = np.load(f)
        out[os.path.basename(f).split(f'_{SUFFIX}')[0]] = (z['valid_end'] - z['valid_start']).sum() / 200 / 3600
    return out


def pick(ds, hours, fraction):
    h = subject_hours(ds)
    if not h:
        raise SystemExit(f'{ds}: no compiled cache')
    order = sorted(h, key=lambda s: (len(s), s))
    random.Random(SEED).shuffle(order)
    target = fraction * (sum(h.values()) if hours is None else hours)
    if target < 0.5 * np.mean(list(h.values())):
        return [], 0.0
    chosen, t = [], 0.0
    for s in order:
        if chosen and t + h[s] / 2 > target:   # stop at the subject count closest to the target
            break
        chosen.append(s)
        t += h[s]
    return sorted(chosen, key=lambda s: (len(s), s)), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fraction', type=float, default=1.0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.out))
    dp, total, per = {}, 0.0, {}
    for paradigm, sets in ALLOC.items():
        for ds, hours in sets.items():
            subs, t = pick(ds, hours, args.fraction)
            if subs:
                dp[ds] = {'dataset_path': f'datas/pretrain/{ds}', 'subject_to_use': subs, 'channels_to_use': ['all']}
            per[paradigm] = per.get(paradigm, 0.0) + t
            total += t
            print(f'  {paradigm:22} {ds:20} {len(subs):3} subjects {t:6.1f} h')
    print(f'total {total:.1f} h | ' + ', '.join(f'{p} {h / total:.0%}' for p, h in per.items()))
    cfg['dataset_params']['pretrain'] = dp
    with open(args.out, 'w') as f:
        f.write(json.dumps(cfg, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
