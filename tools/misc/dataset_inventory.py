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
    'SEED_IV': 'Emotion (4-class)', 'SEED_V': 'Emotion (5-class)', 'SEED_VII': 'Emotion (7-class)',
    'SEED_VIG': 'Vigilance (PERCLOS regression)', 'Siena': 'Seizure detection (adult)',
    'Sleep_EDFx': 'Sleep staging', 'Things_EEG2': 'Visual decoding (images)',
    'TUAB': 'Abnormal EEG (clinical)', 'TUEV': 'Event classification (clinical)',
    'TUSL': 'Slowing classification (clinical)', 'ADFTD': "Clinical: Alzheimer's / FTD / healthy (rest)",
    'Lee2019_ERP': 'P300 speller (continuous)',
    'ErpCore2021': 'raw only (read by the ErpCore2021_<task> folders)',
    'GuttmannFlury2025': 'raw only (read by the GuttmannFlury2025_<task> folders)',
    'GoNoGo_Delorme(inhouse)': 'Go / No-go (in-house hold)',
    'Lane_Keeping(inhouse)': 'Lane keeping (in-house hold)',
    'BI2014a': 'P300 (Brain Invaders)',
    'BI2014b': 'P300 (Brain Invaders)',
    'BI2015a': 'P300 (Brain Invaders)',
    'BI2015b': 'P300 (Brain Invaders)',
    'Chang2025': 'Motor imagery (rehabilitation)',
    'Gao2026': 'Visual imagery',
    'Kim2025BetaRange': 'SSVEP (beta range)',
    'Yang2025': 'Motor imagery (multi-day)',
    'Stieger2021': 'Motor imagery (cursor control)',
    'Liu2024': 'Motor imagery (stroke)',
    'Ma2020': 'Motor imagery (multi-session)',
    'ErpCore2021_ERN': 'ERP (ERN)',
    'ErpCore2021_MMN': 'ERP (MMN)',
    'ErpCore2021_N170': 'ERP (N170)',
    'ErpCore2021_N2pc': 'ERP (N2pc)',
    'ErpCore2021_N400': 'ERP (N400)',
    'ErpCore2021_P3': 'ERP (P3)',
    'GuttmannFlury2025_ME': 'Motor execution',
    'GuttmannFlury2025_MI': 'Motor imagery',
    'GuttmannFlury2025_P300': 'P300 speller',
    'GuttmannFlury2025_SSVEP': 'SSVEP',
    'PURSUE_LRP_ERN': 'ERP (LRP_ERN)',
    'PURSUE_MMN': 'ERP (MMN)',
    'PURSUE_N170': 'ERP (N170)',
    'PURSUE_N2pc': 'ERP (N2pc)',
    'PURSUE_N400': 'ERP (N400)',
    'PURSUE_P300': 'ERP (P300)',
    'NMT_Clinical': 'Clinical (routine EEG)',
    'LSE_IndiaTanzania': 'Resting state / task (consumer headsets)',
    'PD_Rest': "Resting state (Parkinson's)",
    'Rest_PrePostCognitive': 'Resting state (pre / post task)',
    'LEMON': 'Resting state',
    'Nieuwland2018_N400': 'ERP (N400, sentence reading)',
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
    'Liu2022EldBETA': 'SSVEP (9-class, elderly)', 'UCSD_PD': "Clinical: Parkinson's (rest)",
}

# Which external finetune benchmark lists each dataset: B = EEG-FM-Bench (arXiv 2508.17742),
# C = EEG-FM-Compass (arXiv 2601.17883). Notes on both: docs/references/*.md (local, git-ignored).
BENCH = {
    'BNCI2014001': 'B, C', 'BNCI2014004': 'C', 'BNCI2014008': 'C', 'BNCI2014009': 'C',
    'BNCI2015001': 'C', 'CHB_MIT': 'C', 'EEGMAT': 'B, C', 'Nakanishi2015': 'C', 'PhysionetMI': 'B',
    'SEED': 'B, C', 'SEED_V': 'B', 'SEED_VII': 'B', 'SEED_VIG': 'C', 'Siena': 'B',
    'Sleep_EDFx': 'C', 'Things_EEG2': 'B, C', 'TUAB': 'B, C', 'TUEV': 'B', 'TUSL': 'B', 'ADFTD': 'B',
}
# Benchmark datasets with no datas/ folder yet (all open access).
MISSING = [
    ('Mimul-11', 'Motor imagery (upper limb, 3-class)', 'B', 'open (GigaDB, Jeong 2020), not fetched'),
    ('HMC', 'Sleep staging', 'B', 'open (PhysioNet hmc-sleep-staging), not fetched'),
]

# Pretrain candidates recorded but on hold (TB scale, 2026-09-30): (name, source, subjects / channels, size, note)
ON_HOLD = [
    ('HBN (Healthy Brain Network), releases 1-11', 'NEMAR on005505-on005516 (+ nm000103)', '~3000 children, 129ch EGI HydroCel',
     '~2 TB', 'pediatric, EGI net, 5 tasks + rest; EEG Foundation Challenge 2025 data'),
    ('PEERS', 'NEMAR on004395', '364, 125ch EGI', '9.6 TB', 'memory encoding / free recall, many sessions'),
    # gated SJTU BCMI sets (application form + license, bcmi.sjtu.edu.cn/home/seed/): likely the same NeuroScan 62ch
    # cap as SEED/IV/V (more subjects, no new montage); film/driving tasks count toward the 'rest/other' share
    ('SEED-SD', 'BCMI (gated)', '40, 62ch NeuroScan?', '?', 'emotion under sleep deprivation / recovery / normal'),
    ('SEED-DV', 'BCMI (gated)', '20, 62ch NeuroScan?', '?', 'watching 1400 video clips (visual decoding)'),
    ('SEED-VLA / SEED-VRW', 'BCMI (gated)', '?', '?', 'fatigue: lab-simulated and real-world driving'),
    ('SEED-VII', 'BCMI (gated)', '?', '?', '7 emotions; may be wanted as finetune instead'),
    ('SEED-FRA / SEED-GER', 'BCMI (gated)', '8 + 8', '?', '3 emotions, French / German subjects; small'),
]

RAW_SHARED_IN_CORPUS = {'ErpCore2021'}   # raw/ read by the compiled ErpCore2021_<task> folders

# Why a downloaded pretrain dataset is not compiled (it is listed under the candidates, not the corpus table)
NOT_COMPILED = {
    **{d: 'raw > 100 GB, left out of corpus v2 for disk space (2026-10-05)' for d in (
        'Stieger2021', 'Yang2025', 'LEMON', 'Nieuwland2018_N400', 'GuttmannFlury2025', 'GuttmannFlury2025_ME',
        'GuttmannFlury2025_MI', 'GuttmannFlury2025_P300', 'GuttmannFlury2025_SSVEP')},
    'Ma2020': 'download fails (dataverse 400 Bad Request)',
    'Liu2024': 'MOABB metadata fails (figshare 403 on the electrodes file)',
    'GoNoGo_Delorme(inhouse)': 'in-house hold', 'Lane_Keeping(inhouse)': 'in-house hold',
}

# Why each datas/archive/ dataset was dropped from pretraining (2026-09-24 rebalance).
ARCHIVED = {
    'AAD_KUL': 'Public release (Zenodo 4004271) is downsampled to 128 Hz, 0.5 Hz high-passed and '
               'MWF artifact-removed -- no >64 Hz content; raw 8192 Hz data only on request from '
               'KU Leuven (Bertrand / Francart)',
    'BCICIV1_Train': 'Subjects c, d, e of BCI Competition IV ds1 are artificially generated, not real EEG; '
                     'the 4 real subjects are ~3.5 h, not worth keeping',
    'BCICIV1_Test': 'Same subjects as BCICIV1_Train (3 of 7 synthetic)',
    'DEAP': 'Only the data_preprocessed_python release is on disk: downsampled to 128 Hz, band-passed '
            '4-45 Hz, EOG-removed and re-referenced -- no delta or >45 Hz content, unlike the 0.5-100 Hz '
            'corpus. Revisit with the raw 512 Hz BDF release (data_original, same EULA)',
    'Things_EEG2': 'Finetune benchmark (B, C) dropped 2026-10-05: image decoding / 200-way retrieval is not a field '
                   'we compete in, and 241.5 GB; never downloaded',
    'EmotionVideo': 'g.tec Unicorn, 8 dry electrodes: sparse montage (mostly zero-padded on the 10-10 '
                    'grid) and dry-electrode artifacts, little spatial signal',
}


def row(split, path, suffix):
    name = os.path.basename(path)
    notes = os.path.join(path, 'NOTES.md')
    meta_path = os.path.join(path, 'metadata.json')
    fetch = os.path.join(path, 'FETCH_STATUS.json')
    if not os.path.exists(meta_path):
        if os.path.exists(fetch):   # being fetched by tools/misc/fetch_datasets.py, not added yet
            f = json.load(open(fetch))
            prog = (f"{f.get('bytes_done', 0) / 1e9:.0f}/{f.get('bytes_total', 0) / 1e9:.0f} GB" if f.get('source') == 'nemar'
                    else f"{f.get('subjects_done', 0)}/{f.get('subjects_total', '?')} subjects")
            status = f"raw {f.get('state')} ({f.get('source')} {f.get('id')}, {prog}), not added"
        else:
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
# pretrain datasets compile to continuous windows (compile_params.continuous_seconds), finetune ones to event windows
suffix_of = {split: cache_suffix(cp['sample_freq'], cp['bandpass_filter'], cp.get('pre_event_seconds', 0.0),
                                 cp.get('post_event_seconds', 0.0),
                                 cp.get('continuous_seconds', 0.0) if split == 'pretrain' else 0.0)
             for split in ('finetune', 'pretrain')}
lines = ['# Datasets', '',
         'Generated by `python -m tools.misc.dataset_inventory` -- re-run it whenever a dataset is added,',
         'compiled or migrated (docs/agents/adding-a-dataset.md Step 9). Hand-maintained part: the',
         '`PARADIGM` table in that script.', '',
         f"Hours = real (non-padded) samples over every compiled trial in the current cache "
         f"(finetune `*_{suffix_of['finetune']}.npz`, pretrain `*_{suffix_of['pretrain']}.npz`).",
         'A pretrain dataset not listed in configs/compile.json (left out of the corpus) shows no hours.',
         'P300 rows overstate the recording: their 1 s windows around flashes ~0.25 s apart overlap.',
         'Benchmark: B = EEG-FM-Bench (arXiv 2508.17742), C = EEG-FM-Compass (arXiv 2601.17883).',
         'Event: where the event (cue / flash / stimulus) sits inside each compiled trial, seconds from the',
         'trial start -- the line time-axis plots draw. — = no event (windows cut from continuous recordings).', '']
candidates = []   # downloaded pretrain datasets without a compiled cache
for split in ('finetune', 'pretrain'):
    rows, total = [], 0.0
    for p in sorted(glob.glob(f'datas/{split}/*/'), key=str.lower):
        r, h = row(split, p.rstrip('/'), suffix_of[split])
        # a raw-only folder whose task folders are compiled (moabb raw_root) stays with the corpus
        if split == 'pretrain' and not h and os.path.basename(p.rstrip('/')) not in RAW_SHARED_IN_CORPUS:
            candidates.append(r)
            continue
        rows.append(r)
        total += h
    lines += [f'## {split} ({len(rows)} datasets, {total:.1f} h compiled)', '',
              '| Dataset | Paradigm | Benchmark | Subjects | Ch | Native Hz | Classes | Event | Hours | Status |',
              '|---|---|---|---|---|---|---|---|---|---|']
    lines += ['| ' + ' | '.join([r[0]] + r[2:]) + ' |' for r in rows]
    if split == 'finetune':
        lines += [f'| {n} | {p} | {b} |  |  |  |  |  |  | {st} |' for n, p, b, st in MISSING]
    lines.append('')
lines += ['## pretrain candidates', '',
          f'### Downloaded, not compiled ({len(candidates)} folders in `datas/pretrain/`, not in the corpus)', '',
          '| Dataset | Paradigm | Subjects | Ch | Native Hz | Classes | Status | Why not compiled |',
          '|---|---|---|---|---|---|---|---|']
lines += ['| ' + ' | '.join([r[0], r[2], r[4], r[5], r[6], r[7], r[10], NOT_COMPILED.get(r[0], '')]) + ' |' for r in candidates]
lines += ['', '### On hold, not downloaded (TB scale or gated)', '', '| Dataset | Source | Subjects / channels | Size | Note |',
          '|---|---|---|---|---|'] + [f'| {" | ".join(r)} |' for r in ON_HOLD] + ['']
archived = sorted(os.path.basename(p.rstrip('/')) for p in glob.glob('datas/archive/*/'))
lines += [f'## archive ({len(archived)} datasets, not compiled into any run)', '',
          'Moved out of `datas/pretrain/` / `datas/finetune/` and `configs/compile.json`; loaders, metadata and cache kept.', '',
          '| Dataset | Paradigm | Reason |', '|---|---|---|']
lines += [f"| {n} | {PARADIGM.get(n, '?')} | {ARCHIVED.get(n, '?')} |" for n in archived]
lines.append('')
open('datas/DATASETS.md', 'w').write('\n'.join(lines))
print('\n'.join(lines))
