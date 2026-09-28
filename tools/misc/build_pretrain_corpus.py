"""
Builds dataset_params.pretrain for the balanced pretrain corpus (2026-09-24 design): motor
(MI + motor execution) ~50%, the other paradigms ~12.5% each.

ALLOC below is the full corpus in hours per dataset. Every compiled subject of a dataset is used; a
dataset over its allocation gets a per-dataset window_fraction = allocation / its hours (IO/dataset.py
keeps that fraction of every subject's windows), so balancing never drops subjects (2026-09-29; the
model has to see cross-subject variation). The corpus sizes multiply in on top:
configs/pretrain_tiny.template.json sets preprocess_params.window_fraction = 0.05.

    python -m tools.misc.build_pretrain_corpus --out configs/pretrain.template.json

--out must already exist (a pretrain config); only its dataset_params.pretrain is replaced.
"""
import argparse
import glob
import json
import os

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
SUFFIX = 'fs200_bp0.5-100.0_pre1_post4'


def subject_hours(ds):
    out = {}
    for f in glob.glob(f'datas/pretrain/{ds}/cache/*_{SUFFIX}.npz'):
        z = np.load(f)
        out[os.path.basename(f).split(f'_{SUFFIX}')[0]] = (z['valid_end'] - z['valid_start']).sum() / 200 / 3600
    return out


def pick(ds, hours):
    """-> (every compiled subject, window fraction or None, hours after the fraction)."""
    h = subject_hours(ds)
    if not h:
        raise SystemExit(f'{ds}: no compiled cache')
    total = sum(h.values())
    frac = None if hours is None or hours >= total else round(hours / total, 4)
    return sorted(h, key=lambda s: (len(s), s)), frac, total * (frac or 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.out))
    dp, total, per = {}, 0.0, {}
    for paradigm, sets in ALLOC.items():
        for ds, hours in sets.items():
            subs, frac, t = pick(ds, hours)
            dp[ds] = {'dataset_path': f'datas/pretrain/{ds}', 'subject_to_use': subs, 'channels_to_use': ['all']}
            if frac is not None:
                dp[ds]['window_fraction'] = frac
            per[paradigm] = per.get(paradigm, 0.0) + t
            total += t
            print(f'  {paradigm:22} {ds:20} {len(subs):3} subjects {t:6.1f} h' + (f' (window_fraction {frac})' if frac else ''))
    print(f'total {total:.1f} h | ' + ', '.join(f'{p} {h / total:.0%}' for p, h in per.items()))
    cfg['dataset_params']['pretrain'] = dp
    with open(args.out, 'w') as f:
        f.write(json.dumps(cfg, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
