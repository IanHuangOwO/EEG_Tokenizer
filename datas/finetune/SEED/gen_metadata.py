"""
SEED (SJTU Emotion EEG Dataset, BCMI Lab) -- metadata.json generator. See docs/agents/adding-a-dataset.md Step 3.

Source verified (2026-09-30) from the download (raw/Preprocessed_EEG/readme.txt, label.mat, channel-order.xlsx,
subject-id-gender-seed.txt): 15 subjects x 3 sessions (files <subject>_<yyyymmdd>.mat, sessions ordered by date),
15 film clips per session, one array per clip (<initials>_eeg1..15, [62, T]). The released "Preprocessed_EEG" is
downsampled to 200 Hz and 0-75 Hz band-passed by the authors (the 1000 Hz originals are not in this download).
Clip labels (label.mat, the same for every session): 1 positive, 0 neutral, -1 negative -> 0 negative, 1 neutral,
2 positive. 62ch ESI NeuroScan 10-20 cap, channel order from channel-order.xlsx.

Windows: non-overlapping 1 s within each clip, as EEG-FM-Compass cuts SEED (50,910 windows over 45 sessions).
"""
import json
import os
import re

import scipy.io as sio

ROOT = os.path.dirname(__file__)
RAW = os.path.join(ROOT, 'raw', 'Preprocessed_EEG')

DATASET_INFO = {
    "source_url": "https://bcmi.sjtu.edu.cn/home/seed/seed.html",
    "file_format": "MATLAB (.mat)",
    "description": ("15 subjects x 3 sessions, 15 film clips per session (5 positive / 5 neutral / 5 negative), "
                    "62ch NeuroScan, released at 200 Hz (0-75 Hz band-passed); classify the clip's emotion."),
    "task_type": "emotion",
    "reference": ("Zheng W-L, Lu B-L (2015). Investigating Critical Frequency Bands and Channels for EEG-based "
                  "Emotion Recognition with Deep Neural Networks. IEEE TAMD 7(3):162-175."),
    "notes": ("Preprocessed_EEG (200 Hz) is the only EEG in the download. 1 s non-overlapping windows within each "
              "clip (EEG-FM-Compass). Session = recording date order. Access is gated (license agreement)."),
}

NAMES = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3',
         'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5',
         'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7',
         'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
# CB1/CB2 (NeuroScan cerebellar sites) have no MNE position: polar fallback as BETA_4s/gen_metadata.py
POLAR = {'CB1': -170.0, 'CB2': 170.0}
CHANNELS = {str(i + 1): ({"label": n, "coordinates": {"polar_angle_deg": POLAR[n], "polar_radius": 0.52}}
                         if n in POLAR else {"label": n}) for i, n in enumerate(NAMES)}

TARGETS = {"count": 3, "type": "emotion", "0": {"label": "negative"}, "1": {"label": "neutral"},
           "2": {"label": "positive"}}


def main():
    raw_labels = sio.loadmat(os.path.join(RAW, 'label.mat'))['label'].ravel().tolist()
    clip_labels = [{-1: 0, 0: 1, 1: 2}[v] for v in raw_labels]
    by_sub = {}
    for f in os.listdir(RAW):
        m = re.fullmatch(r'(\d+)_(\d{8})\.mat', f)
        if m:
            by_sub.setdefault(int(m.group(1)), []).append((m.group(2), f"raw/Preprocessed_EEG/{f}"))
    structure = {str(s): {"sessions": [p for _, p in sorted(v)]} for s, v in sorted(by_sub.items())}
    meta = {
        "data_metadata": {
            "dataset_name": "SEED",
            "dataset_info": DATASET_INFO,
            "acquisition": {"sample_frequency": 200, "window_size_seconds": 1.0, "num_subjects": len(structure),
                            "num_sessions_per_subject": 3},
            "targets": TARGETS,
            "clip_labels": clip_labels,
            "channels": {"count": len(CHANNELS), "system": "10-20 (ESI NeuroScan 62ch)", **CHANNELS},
        },
        "data_structure": structure,
    }
    out = os.path.join(ROOT, 'metadata.json')
    json.dump(meta, open(out, 'w'), indent=4)
    print(f"wrote {out}: {len(structure)} subjects, {sum(len(v['sessions']) for v in structure.values())} sessions")


if __name__ == '__main__':
    main()
