"""
EEGMAT (PhysioNet "EEG During Mental Arithmetic Tasks", Zyma et al. 2019) --
metadata.json generator. See docs/agents/adding-a-dataset.md Step 3.

Source verified directly (2026-09-22) via physionet.org/content/eegmat/1.0.0/
(README.txt, subject-info.csv, one subject's real EDF header). 36 subjects
(Subject00-Subject35), Neurocom EEG 23-channel system, 19 real EEG channels
(10-20 system, ear reference) + 2 non-EEG columns (A2-A1 reference,
ECG -- both dropped), 500 Hz (verified via EDF header, not stated in
README.txt). Two files/subject: "_1" = 60s-nominal resting baseline before
the task (actual header duration 182s, per-file, not all exactly 60s),
"_2" = same duration during the mental arithmetic task (serial subtraction).

Real 2-class label from subject-info.csv's "Count quality" column (0=Group B
"bad" performer, mean 7 correct subtractions/4min; 1=Group G "good"
performer, mean 21/4min) -- matches Table 14's "low, high" workload
classification. Label is per-subject (task performance trait), applied only
to "_2" (task) windows -- "_1" (baseline, no task running) is NOT given a
workload label since the label describes task performance, not the resting
state; "_1" files are downloaded but unused by loader.py.
"""
import csv
import json
import os
import re

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "https://physionet.org/content/eegmat/1.0.0/",
    "file_format": "EDF",
    "description": (
        "36 subjects, 19ch EEG (10-20 system), resting baseline (_1) and "
        "mental-arithmetic task (_2) recordings, serial-subtraction task "
        "used to classify workload/performance quality."
    ),
    "task_type": "mental_workload",
    "reference": "Zyma I, Tukaev S, Seleznov I, Kiyono K, Popov A, Chernykh M, "
                 "Shpenkov O (2019). Electroencephalograms during Mental "
                 "Arithmetic Task Performance. Data 4(1):14. "
                 "doi:10.3390/data4010014",
    "notes": (
        "Label (0=low/Group B, 1=high/Group G workload-handling quality) "
        "comes from subject-info.csv's 'Count quality' column, applied only "
        "to _2 (task) recordings -- _1 (baseline) is present on disk but "
        "unlabelled/unused, see this file's module docstring. 500 Hz "
        "sample rate confirmed directly from an EDF header (not stated in "
        "the PhysioNet README). Channels 'A2-A1' (ear reference) and 'ECG' "
        "dropped, only the 19 'EEG <name>'-prefixed columns used."
    ),
}

CHANNELS = {
    str(i + 1): {"label": label} for i, label in enumerate([
        "Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "C4",
        "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz",
    ])
}

TARGETS = {
    "count": 2,
    "type": "mental_workload",
    "0": {"label": "low workload-handling quality (Group B, 'bad' counters)"},
    "1": {"label": "high workload-handling quality (Group G, 'good' counters)"},
}


def load_labels(csv_path):
    labels = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            m = re.match(r"Subject(\d+)", row["Subject"])
            labels[int(m.group(1))] = int(row["Count quality"])
    return labels


def build_data_structure(raw_dir, labels):
    structure = {}
    for dname in sorted(os.listdir(raw_dir)):
        m = re.match(r"Subject(\d+)_2\.edf$", dname)
        if not m:
            continue
        n = int(m.group(1))
        if n not in labels:
            continue
        structure[str(n)] = {"file": f"raw/Subject{n:02d}_2.edf", "label": labels[n]}
    return structure


def main():
    raw_dir = os.path.join(ROOT, "raw")
    labels = load_labels(os.path.join(raw_dir, "subject-info.csv"))
    structure = build_data_structure(raw_dir, labels)

    meta = {
        "data_metadata": {
            "dataset_name": "EEGMAT",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 500,
                "window_size_seconds": 4.0,
                "num_subjects": len(structure),
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
