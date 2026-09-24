"""
Regenerates datas/DATASETS.md: one row per datas/<split>/<Name>/ with paradigm, subjects,
channels, native rate, classes, compiled hours and status. Re-run after adding, compiling
or migrating a dataset (docs/agents/adding-a-dataset.md Step 9):

    python -m tools.misc.dataset_inventory

Hours = real (non-padded) samples summed over every compiled trial, from the cache that
configs/compile.json currently selects. Trials that overlap in time (P300 flashes ~0.25 s
apart with 1 s windows) are counted once per trial, so those rows overstate the recording.
"""
import glob
import json
import os

import numpy as np

from IO.preprocessing import cache_suffix

PARADIGM = {
    # finetune
    'BNCI2014001': 'Motor imagery (4-class)', 'BNCI2014004': 'Motor imagery (L/R hand)',
    'BNCI2014008': 'P300 speller (ALS)', 'BNCI2014009': 'P300 speller',
    'BNCI2015001': 'Motor imagery (hand/feet)', 'CHB_MIT': 'Seizure detection (pediatric)',
    'EEGMAT': 'Mental workload (arithmetic)', 'Nakanishi2015': 'SSVEP (12-class)',
    'PhysionetMI': 'Motor imagery (5-class)', 'SEED': 'Emotion (3-class)',
    'SEED_V': 'Emotion (5-class)', 'SEED_VII': 'Emotion (7-class)',
    'SEED_VIG': 'Vigilance (PERCLOS regression)', 'Siena': 'Seizure detection (adult)',
    'Sleep_EDFx': 'Sleep staging', 'Things_EEG2': 'Visual decoding (images)',
    'TUAB': 'Abnormal EEG (clinical)', 'TUEV': 'Event classification (clinical)',
    'TUSL': 'Slowing classification (clinical)',
    # pretrain
    'AAD_KUL': 'Auditory attention', 'BCIC2020-3': 'Imagined speech',
    'BCICIV1_Test': 'Motor imagery (uncued)', 'BCICIV1_Train': 'Motor imagery',
    'BCMI_MusicEmotion': 'Emotion (music)', 'BETA_3s': 'SSVEP (40-class)',
    'BETA_4s': 'SSVEP (40-class)', 'DEAP': 'Emotion (music video)',
    'EmotionVideo': 'Emotion (video)', 'ERP_Longitudinal': 'RSVP oddball ERP',
    'GraspAndLift_Test': 'Motor execution (grasp-and-lift)',
    'GraspAndLift_Train': 'Motor execution (grasp-and-lift)',
    'Inria_Test': 'Error-related potential', 'Inria_Train': 'Error-related potential',
    'Lee2019_MI': 'Motor imagery (L/R hand)', 'MDD_Mumtaz': 'Clinical: depression (rest + P300)',
    'Neonatal_Helsinki': 'Clinical: neonatal (NICU)',
    'SPIS': 'Resting state (eyes open/closed)', 'SRM_RestingState': 'Resting state',
    'STEW': 'Mental workload (multitasking)',
    'Weibo2014': 'Motor imagery (7-class)', 'Cho2017': 'Motor imagery (L/R hand)',
    'Schirrmeister2017': 'Motor execution (4-class)', 'Dreyer2023': 'Motor imagery (L/R hand)',
    'Wang2016': 'SSVEP (40-class)', 'Lee2019_SSVEP': 'SSVEP (4-class)',
    'Liu2022EldBETA': 'SSVEP (9-class, elderly)', 'Lee2019_ERP': 'P300 speller (continuous)', 'UCSD_PD': "Clinical: Parkinson's (rest)",
}

# Which external finetune benchmark lists each dataset: B = EEG-FM-Bench (arXiv 2508.17742),
# C = EEG-FM-Compass (arXiv 2601.17883). Notes on both: docs/papers/*.md (local, git-ignored).
BENCH = {
    'BNCI2014001': 'B, C', 'BNCI2014004': 'C', 'BNCI2014008': 'C', 'BNCI2014009': 'C',
    'BNCI2015001': 'C', 'CHB_MIT': 'C', 'EEGMAT': 'B, C', 'Nakanishi2015': 'C', 'PhysionetMI': 'B',
    'SEED': 'B, C', 'SEED_V': 'B', 'SEED_VII': 'B', 'SEED_VIG': 'C', 'Siena': 'B',
    'Sleep_EDFx': 'C', 'Things_EEG2': 'B, C', 'TUAB': 'B, C', 'TUEV': 'B', 'TUSL': 'B',
}
# Benchmark datasets with no datas/ folder yet (all open access).
MISSING = [
    ('Mimul-11', 'Motor imagery (upper limb, 3-class)', 'B', 'open (GigaDB, Jeong 2020), not fetched'),
    ('HMC', 'Sleep staging', 'B', 'open (PhysioNet hmc-sleep-staging), not fetched'),
    ('ADFTD', "Clinical: Alzheimer's / FTD", 'B', 'open (OpenNeuro ds004504), not fetched'),
]


def row(split, path, suffix):
    name = os.path.basename(path)
    notes = os.path.join(path, 'NOTES.md')
    meta_path = os.path.join(path, 'metadata.json')
    if not os.path.exists(meta_path):
        status = open(notes).readline().split('--', 1)[-1].strip() if os.path.exists(notes) else 'no metadata'
        return [name, split, PARADIGM.get(name, '?'), BENCH.get(name, ''), '', '', '', '', '', '', status], 0.0
    m = json.load(open(meta_path))
    d = m['data_metadata']
    n_subj = len(m['data_structure'])
    n_cls = d.get('targets', {}).get('count', '')
    if d.get('targets', {}).get('type') in ('pretrain_dummy', 'unlabeled', 'unlabelled'):
        n_cls = 'none'
    caches = glob.glob(os.path.join(path, 'cache', f'*_{suffix}.npz'))
    samples = 0
    for c in caches:
        z = np.load(c)
        samples += int((z['valid_end'] - z['valid_start']).sum()) if 'valid_end' in z.files \
            else z['data'].shape[0] * z['data'].shape[2]
    hours = samples / SAMPLE_FREQ / 3600
    # event position inside each compiled trial (what time-axis plots draw as the event line,
    # tools.analysis.lookup_event_onset_sample): seconds from trial start, or '—' = no event
    if d.get('event_onset_seconds') is not None:
        event = f"{d['event_onset_seconds']:g} s"
    elif d.get('event_onset_sample') is not None:
        event = f"{d['event_onset_sample'] / SAMPLE_FREQ:g} s"
    else:
        event = '—'
    status = 'compiled' if len(caches) >= n_subj else (f'partial ({len(caches)}/{n_subj})' if caches else 'not compiled')
    via = ' (MOABB)' if 'moabb' in d else ''
    return [name, split, PARADIGM.get(name, d.get('dataset_info', {}).get('task_type', '?')),
            BENCH.get(name, ''), str(n_subj),
            str(d['channels']['count']), f"{float(d['acquisition']['sample_frequency']):g}", str(n_cls), event,
            f'{hours:.1f}' if caches else '', status + via], hours


cp = json.load(open('configs/compile.json'))['compile_params']
SAMPLE_FREQ = cp['sample_freq']
suffix = cache_suffix(cp['sample_freq'], cp['bandpass_filter'], cp.get('pre_event_seconds', 0.0),
                      cp.get('post_event_seconds', 0.0))
lines = ['# Datasets', '',
         'Generated by `python -m tools.misc.dataset_inventory` -- re-run it whenever a dataset is added,',
         'compiled or migrated (docs/agents/adding-a-dataset.md Step 9). Hand-maintained part: the',
         '`PARADIGM` table in that script.', '',
         f'Hours = real (non-padded) samples over every compiled trial in the current cache (`*_{suffix}.npz`).',
         'P300 rows overstate the recording: their 1 s windows around flashes ~0.25 s apart overlap.',
         'Benchmark: B = EEG-FM-Bench (arXiv 2508.17742), C = EEG-FM-Compass (arXiv 2601.17883).',
         'Event: where the event (cue / flash / stimulus) sits inside each compiled trial, seconds from the',
         'trial start -- the line time-axis plots draw. — = no event (windows cut from continuous recordings).', '']
for split in ('finetune', 'pretrain'):
    rows, total = [], 0.0
    for p in sorted(glob.glob(f'datas/{split}/*/'), key=str.lower):
        r, h = row(split, p.rstrip('/'), suffix)
        rows.append(r)
        total += h
    lines += [f'## {split} ({len(rows)} datasets, {total:.1f} h compiled)', '',
              '| Dataset | Paradigm | Benchmark | Subjects | Ch | Native Hz | Classes | Event | Hours | Status |',
              '|---|---|---|---|---|---|---|---|---|---|']
    lines += ['| ' + ' | '.join([r[0]] + r[2:]) + ' |' for r in rows]
    if split == 'finetune':
        lines += [f'| {n} | {p} | {b} |  |  |  |  |  |  | {st} |' for n, p, b, st in MISSING]
    lines.append('')
open('datas/DATASETS.md', 'w').write('\n'.join(lines))
print('\n'.join(lines))
