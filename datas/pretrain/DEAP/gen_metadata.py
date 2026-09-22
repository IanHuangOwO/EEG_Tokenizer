"""
DEAP (Database for Emotion Analysis using Physiological signals) --
metadata.json generator. See docs/agents/adding-a-dataset.md Step 3.

Source: `data_preprocessed_python/sNN.dat`, the official preprocessed release
(Koelstra et al. 2012) -- pickled dict {'data': [40 trials, 40 channels,
8064 samples], 'labels': [40 trials, 4] continuous 1-9 ratings (valence,
arousal, dominance, liking)}. Verified directly against s01.dat (2026-09-22):
shape/dtype match, and 8064 samples / 128 Hz = 63.0 s exactly, matching the
documented 3 s pre-trial baseline + 60 s trial -- confirms the 128 Hz sample
rate independently rather than trusting the paper alone.

Only the first 32 (EEG) of the 40 channels are used; the last 8 are
peripheral (hEOG, vEOG, zEMG, tEMG, GSR, respiration, plethysmograph,
temperature) -- a different modality entirely, not an EEG montage channel,
and this repo's IO/dataset.py NON_EEG_CHANNELS is an exact-match set that
would not catch labels like "zEMG"/"GSR" anyway. loader.py drops them at
read time; they are never listed here.

Channel order is the DEAP-published 32-channel Biosemi 10-20 order,
replicated identically across the paper's supplementary material and every
open DEAP loader -- NOT independently re-verified against a channel list
shipped in this particular archive (none was present). All 32 names resolve
via MNE's standard_1020 montage (checked 2026-09-22), so per-channel
"coordinates" are omitted here -- same precedent as PhysionetMI's midline
channels (see adding-a-dataset.md Step 3 notes).
"""
import json
import os
import re

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "https://www.eecs.qmul.ac.uk/mmv/datasets/deap/",
    "file_format": "Python pickle (.dat, data_preprocessed_python release)",
    "description": (
        "32 participants, EEG (+ peripheral) recorded while watching 40 one-minute "
        "music video excerpts, rating each on valence/arousal/dominance/liking "
        "(1-9 continuous scale) afterwards. This release: 512 Hz original, "
        "downsampled to 128 Hz, EOG artifact removed, bandpass 4.0-45.0 Hz, "
        "average referenced, segmented to 3 s pre-trial baseline + 60 s trial."
    ),
    "task_type": "emotion_recognition",
    "reference": "Koelstra et al., 'DEAP: A Database for Emotion Analysis using "
                  "Physiological Signals', IEEE Trans. Affective Computing, 2012.",
    "notes": (
        "Labels here are a derived 4-class valence x arousal quadrant (midpoint-5 "
        "split on each dimension; user decision 2026-09-22), computed in loader.py "
        "from the raw continuous ratings -- not the dataset's native label. "
        "dominance/liking ratings are read but unused. Samples 0:384 (3 s) of each "
        "8064-sample trial are the pre-trial baseline, kept as part of the trial "
        "(not padding) rather than discarded -- see loader.py."
    ),
}

# DEAP-published 32-channel order (Biosemi ActiveTwo, 10-20 layout).
CHANNELS = {
    str(i + 1): {"label": label} for i, label in enumerate([
        "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1",
        "P3", "P7", "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4",
        "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8",
        "PO4", "O2",
    ])
}

TARGETS = {
    "count": 4,
    "type": "emotion_recognition",
    "0": {"label": "low valence, low arousal"},
    "1": {"label": "low valence, high arousal"},
    "2": {"label": "high valence, low arousal"},
    "3": {"label": "high valence, high arousal"},
}


def build_data_structure(raw_dir):
    structure = {}
    for fname in sorted(os.listdir(raw_dir)):
        m = re.match(r"s(\d+)\.dat$", fname)
        if not m:
            continue
        structure[str(int(m.group(1)))] = {"file": f"raw/data_preprocessed_python/{fname}"}
    return structure


def main():
    raw_dir = os.path.join(ROOT, "raw", "data_preprocessed_python")
    structure = build_data_structure(raw_dir)

    meta = {
        "data_metadata": {
            "dataset_name": "DEAP",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 128,
                "window_size_seconds": 63.0,
                "num_subjects": len(structure),
                "num_trials_per_subject": 40,
            },
            "targets": TARGETS,
            "channels": {"count": len(CHANNELS), "system": "10-20 International System", **CHANNELS},
        },
        "data_structure": structure,
    }

    out_path = os.path.join(ROOT, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"wrote {out_path}: {len(structure)} subjects")


if __name__ == "__main__":
    main()
