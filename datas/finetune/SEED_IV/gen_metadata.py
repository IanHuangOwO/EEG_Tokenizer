"""
SEED-IV (SJTU Emotion EEG Dataset for Four Emotions, BCMI Lab) -- metadata.json generator. See
docs/agents/adding-a-dataset.md Step 3.

Source verified (2026-09-30) from the download (raw/ReadMe.txt, Channel Order.xlsx, raw/eeg_raw_data/<session>/):
15 subjects x 3 sessions (folders 1/2/3, files <subject>_<yyyymmdd>.mat), 24 film clips per session, one array per
clip (<initials>_eeg1..24, [62, T]) at 200 Hz (T / 200 matches the ~2-4 min clips). Clip labels per session from
ReadMe.txt: 0 neutral, 1 sad, 2 fear, 3 happy (kept as is). 62ch ESI NeuroScan, same cap and order as SEED.

Windows: non-overlapping 4 s within each clip (the ReadMe's feature window; no FM benchmark cuts SEED-IV).
"""
import json
import os
import re

ROOT = os.path.dirname(__file__)
RAW = os.path.join(ROOT, 'raw', 'eeg_raw_data')

DATASET_INFO = {
    "source_url": "https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html",
    "file_format": "MATLAB (.mat)",
    "description": ("15 subjects x 3 sessions, 24 film clips per session (neutral / sad / fear / happy), 62ch "
                    "NeuroScan, 200 Hz; classify the clip's emotion."),
    "task_type": "emotion",
    "reference": ("Zheng W-L, Liu W, Lu Y, Lu B-L, Cichocki A (2018). EmotionMeter: A Multimodal Framework for "
                  "Recognizing Human Emotions. IEEE Trans Cybernetics 49(3):1110-1122."),
    "notes": ("eeg_raw_data (200 Hz). 4 s non-overlapping windows within each clip. Session = the dataset's session "
              "folder (1-3). Eye-tracking and feature folders are not staged. Access is gated (license agreement)."),
}

# ReadMe.txt
SESSION_LABELS = [
    [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
]

NAMES = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3',
         'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5',
         'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7',
         'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
# CB1/CB2 (NeuroScan cerebellar sites) have no MNE position: polar fallback as BETA_4s/gen_metadata.py
POLAR = {'CB1': -170.0, 'CB2': 170.0}
CHANNELS = {str(i + 1): ({"label": n, "coordinates": {"polar_angle_deg": POLAR[n], "polar_radius": 0.52}}
                         if n in POLAR else {"label": n}) for i, n in enumerate(NAMES)}

TARGETS = {"count": 4, "type": "emotion", "0": {"label": "neutral"}, "1": {"label": "sad"}, "2": {"label": "fear"},
           "3": {"label": "happy"}}


def main():
    by_sub = {}
    for sess in ('1', '2', '3'):
        for f in os.listdir(os.path.join(RAW, sess)):
            m = re.fullmatch(r'(\d+)_\d{8}\.mat', f)
            if m:
                by_sub.setdefault(int(m.group(1)), {})[int(sess)] = f"raw/eeg_raw_data/{sess}/{f}"
    structure = {str(s): {"sessions": [v[k] for k in sorted(v)]} for s, v in sorted(by_sub.items())}
    assert all(len(v['sessions']) == 3 for v in structure.values())
    meta = {
        "data_metadata": {
            "dataset_name": "SEED_IV",
            "dataset_info": DATASET_INFO,
            "acquisition": {"sample_frequency": 200, "window_size_seconds": 4.0, "num_subjects": len(structure),
                            "num_sessions_per_subject": 3},
            "targets": TARGETS,
            "session_clip_labels": SESSION_LABELS,
            "channels": {"count": len(CHANNELS), "system": "10-20 (ESI NeuroScan 62ch)", **CHANNELS},
        },
        "data_structure": structure,
    }
    out = os.path.join(ROOT, 'metadata.json')
    json.dump(meta, open(out, 'w'), indent=4)
    print(f"wrote {out}: {len(structure)} subjects")


if __name__ == '__main__':
    main()
