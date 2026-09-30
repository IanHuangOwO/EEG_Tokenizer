"""
SEED-V (SJTU multimodal emotion dataset, five emotions, BCMI Lab) -- metadata.json generator. See
docs/agents/adding-a-dataset.md Step 3.

Source verified (2026-09-30) from the download (raw/Read me.txt, EEG_raw/*.ipynb, trial_start_end_timestamp.txt,
emotion_label_and_stimuli_order.xlsx, Channel Order.xlsx): 16 subjects x 3 sessions, files
EEG_raw/<subject>_<session>_<yyyymmdd>.cnt (session = the stimulus set watched, not recording order), 1000 Hz
NeuroScan .cnt with 66 channels: the 62 EEG channels of Channel Order.xlsx plus M1, M2, VEO, HEO (dropped, as the
authors' notebook does; picked by name). 15 clips per session; clip onsets/offsets in seconds from the recording start
(trial_start_end_timestamp.txt), the same for every subject. Emotion per clip and label ints from the xlsx:
0 disgust, 1 fear, 2 sad, 3 neutral, 4 happy. Not filtered by the authors.

Windows: non-overlapping 10 s within each clip, as EEG-FM-Bench (arXiv 2508.17742) cuts SEED-V.
"""
import json
import os
import re

ROOT = os.path.dirname(__file__)
RAW = os.path.join(ROOT, 'raw', 'EEG_raw')

DATASET_INFO = {
    "source_url": "https://bcmi.sjtu.edu.cn/home/seed/seed-v.html",
    "file_format": "NeuroScan (.cnt)",
    "description": ("16 subjects x 3 sessions, 15 film clips per session (happy / sad / fear / disgust / neutral), "
                    "62ch NeuroScan, 1000 Hz; classify the clip's emotion."),
    "task_type": "emotion",
    "reference": ("Liu W, Qiu J-L, Zheng W-L, Lu B-L (2022). Comparing Recognition Performance and Robustness of "
                  "Multimodal Deep Learning Models for Multimodal Emotion Recognition. IEEE TCDS 14(2):715-729."),
    "notes": ("EEG_raw .cnt, M1/M2/VEO/HEO dropped. 10 s non-overlapping windows within each clip (EEG-FM-Bench). "
              "Session = stimulus set (file name), not recording order. Eye data not staged. Access is gated."),
}

LABEL = {'Disgust': 0, 'Fear': 1, 'Sad': 2, 'Neutral': 3, 'Happy': 4}
# emotion_label_and_stimuli_order.xlsx, rows "Session 1..3"
SESSION_EMOTIONS = [
    ['Happy', 'Fear', 'Neutral', 'Sad', 'Disgust'] * 3,
    ['Sad', 'Fear', 'Neutral', 'Disgust', 'Happy', 'Happy', 'Disgust', 'Neutral', 'Sad', 'Fear', 'Neutral', 'Happy',
     'Fear', 'Sad', 'Disgust'],
    ['Sad', 'Fear', 'Neutral', 'Disgust', 'Happy', 'Happy', 'Disgust', 'Neutral', 'Sad', 'Fear', 'Neutral', 'Happy',
     'Fear', 'Sad', 'Disgust'],
]

NAMES = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3',
         'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5',
         'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7',
         'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
# CB1/CB2 (NeuroScan cerebellar sites) have no MNE position: polar fallback as BETA_4s/gen_metadata.py
POLAR = {'CB1': -170.0, 'CB2': 170.0}
CHANNELS = {str(i + 1): ({"label": n, "coordinates": {"polar_angle_deg": POLAR[n], "polar_radius": 0.52}}
                         if n in POLAR else {"label": n}) for i, n in enumerate(NAMES)}

TARGETS = {"count": 5, "type": "emotion", **{str(i): {"label": k.lower()} for k, i in LABEL.items()}}


def main():
    txt = open(os.path.join(ROOT, 'raw', 'trial_start_end_timestamp.txt')).read()
    starts = [json.loads(s) for s in re.findall(r'start_second:\s*(\[[^\]]*\])', txt)]
    ends = [json.loads(s) for s in re.findall(r'end_second:\s*(\[[^\]]*\])', txt)]
    assert len(starts) == len(ends) == 3 and all(len(x) == 15 for x in starts + ends)
    clips = [[[a, b, LABEL[e]] for a, b, e in zip(starts[s], ends[s], SESSION_EMOTIONS[s])] for s in range(3)]
    by_sub = {}
    for f in os.listdir(RAW):
        m = re.fullmatch(r'(\d+)_(\d)_\d{8}\.cnt', f)
        if m:
            by_sub.setdefault(int(m.group(1)), {})[int(m.group(2))] = f"raw/EEG_raw/{f}"
    structure = {str(s): {"sessions": [v[k] for k in sorted(v)]} for s, v in sorted(by_sub.items())}
    assert all(len(v['sessions']) == 3 for v in structure.values())
    meta = {
        "data_metadata": {
            "dataset_name": "SEED_V",
            "dataset_info": DATASET_INFO,
            "acquisition": {"sample_frequency": 1000, "window_size_seconds": 10.0, "num_subjects": len(structure),
                            "num_sessions_per_subject": 3},
            "targets": TARGETS,
            "session_clips": clips,
            "channels": {"count": len(CHANNELS), "system": "10-20 (ESI NeuroScan 62ch)", **CHANNELS},
        },
        "data_structure": structure,
    }
    out = os.path.join(ROOT, 'metadata.json')
    json.dump(meta, open(out, 'w'), indent=4)
    print(f"wrote {out}: {len(structure)} subjects")


if __name__ == '__main__':
    main()
