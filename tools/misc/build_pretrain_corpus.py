"""
Builds dataset_params.pretrain for the balanced pretrain corpus (design 2026-09-30, event-free caches 2026-10-05):
motor (MI + motor execution) 25%, SSVEP 25%, ERP 25%, and the other paradigms share the last 25% evenly
(rest / clinical 12.5%, cognitive / affective 12.5%).

TOTAL_HOURS is the full corpus. Each group's budget is split over its datasets by water-filling: every dataset gets
the same cap, a dataset under the cap is used whole, one over it gets a per-dataset window_fraction = cap / its hours
(IO/dataset.py keeps that fraction of every subject's windows). Balancing never drops subjects (2026-09-29: the model
has to see cross-subject variation), except the MAX_SUBJECTS cap below. Hours are seconds of real signal (valid samples) in the _cont5 caches. The
corpus sizes multiply in on top: configs/pretrain_tiny.template.json sets preprocess_params.window_fraction = 0.05.

    python -m tools.misc.build_pretrain_corpus --out configs/pretrain.template.json

--out must already exist (a pretrain config); only its dataset_params.pretrain is replaced.
"""
import argparse
import glob
import json
import os
import random

import numpy as np

TOTAL_HOURS = 210.0
# group -> (share of the full corpus, datasets). Left out 2026-10-05: Stieger2021, Yang2025, LEMON, Nieuwland2018_N400,
# GuttmannFlury2025_* (raw > 100 GB, disk), Ma2020 (download fails), Liu2024 (figshare 403), in-house sets.
GROUPS = {
    'motor': (0.25, ['Lee2019_MI', 'Dreyer2023', 'Cho2017', 'Weibo2014', 'Schirrmeister2017', 'GraspAndLift_Train',
                     'GraspAndLift_Test', 'Chang2025']),
    'SSVEP': (0.25, ['Lee2019_SSVEP', 'BETA_4s', 'BETA_3s', 'Wang2016', 'Liu2022EldBETA', 'Kim2025BetaRange']),
    'ERP': (0.25, ['ERP_Longitudinal', 'Inria_Train', 'Inria_Test', 'BI2014a', 'BI2014b', 'BI2015a', 'BI2015b',
                   *[f'ErpCore2021_{t}' for t in ('ERN', 'MMN', 'N170', 'N2pc', 'N400', 'P3')],
                   *[f'PURSUE_{t}' for t in ('LRP_ERN', 'MMN', 'N170', 'N2pc', 'N400', 'P300')]]),
    'rest / clinical': (0.125, ['SRM_RestingState', 'MDD_Mumtaz', 'Neonatal_Helsinki', 'UCSD_PD', 'SPIS',
                                'NMT_Clinical', 'LSE_IndiaTanzania', 'PD_Rest', 'Rest_PrePostCognitive']),
    'cognitive / affective': (0.125, ['BCMI_MusicEmotion', 'BCIC2020-3', 'STEW', 'Gao2026']),
}
SUFFIX = 'fs200_bp0.5-100.0_cont5'
# Datasets with more subjects keep a fixed seeded pick of this many (user, 2026-10-05): with one window per subject as
# the floor, NMT_Clinical (2417) / LSE_IndiaTanzania (2000) / Rest_PrePostCognitive (608) would fill the tiny corpus
# (5%) by the floor alone (rest/clinical 42% instead of 12.5%). The pick is the same for every corpus size (nested).
MAX_SUBJECTS = 300


def subject_hours(ds):
    out = {}
    for f in glob.glob(f'datas/pretrain/{ds}/cache/*_{SUFFIX}.npz'):
        with np.load(f) as z:
            out[os.path.basename(f).split(f'_{SUFFIX}')[0]] = (z['valid_end'] - z['valid_start']).sum() / 200 / 3600
    if not out:
        raise SystemExit(f'{ds}: no compiled {SUFFIX} cache')
    if len(out) > MAX_SUBJECTS:
        keep = random.Random(f'subjects/{ds}').sample(sorted(out), MAX_SUBJECTS)
        out = {s: out[s] for s in keep}
    return out


def water_fill(hours, budget):
    """{ds: hours} -> cap such that sum(min(h, cap)) = budget (inf if the group has less than its budget)."""
    if sum(hours.values()) <= budget:
        return float('inf')
    lo, hi = 0.0, max(hours.values())
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sum(min(h, mid) for h in hours.values()) < budget else (lo, mid)
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.out))
    dp, total = {}, 0.0
    for group, (share, sets) in GROUPS.items():
        subj = {ds: subject_hours(ds) for ds in sets}
        hours = {ds: sum(h.values()) for ds, h in subj.items()}
        cap = water_fill(hours, share * TOTAL_HOURS)
        used = 0.0
        for ds in sets:
            frac = None if hours[ds] <= cap else round(cap / hours[ds], 4)
            dp[ds] = {'dataset_path': f'datas/pretrain/{ds}', 'subject_to_use': sorted(subj[ds], key=lambda s: (len(s), s)),
                      'channels_to_use': ['all']}
            if frac is not None:
                dp[ds]['window_fraction'] = frac
            t = hours[ds] * (frac or 1.0)
            used += t
            print(f'  {group:22} {ds:24} {len(subj[ds]):5} subjects {hours[ds]:7.1f} h -> {t:5.1f} h'
                  + (f' (window_fraction {frac})' if frac else ''))
        print(f'{group}: {used:.1f} h of {share * TOTAL_HOURS:.1f} h budget')
        total += used
    print(f'total {total:.1f} h')
    cfg['dataset_params']['pretrain'] = dp
    with open(args.out, 'w') as f:
        f.write(json.dumps(cfg, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
