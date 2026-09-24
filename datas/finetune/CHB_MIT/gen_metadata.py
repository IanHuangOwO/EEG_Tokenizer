"""
CHB-MIT Scalp EEG Database (PhysioNet chbmit 1.0.0) -- metadata.json generator.
See docs/agents/adding-a-dataset.md Step 3.

Source verified directly (2026-09-24) from the downloaded files: RECORDS (686
EDF recordings, authoritative list), per-case chbNN-summary.txt (256 Hz,
channel list, per-file seizure start/end in seconds -- both the "Seizure Start
Time" and "Seizure N Start Time" spellings occur), EDF headers.

Task: binary seizure detection, ictal (1) vs interictal (0) 5 s windows,
BALANCED per subject (decided 2026-09-24): every non-overlapping 5 s window fully
inside a seizure, plus the same number of interictal windows sampled at random
(fixed per-subject seed) from recordings that contain no seizure and are not
directly before/after a seizure recording (recordings are ~1 h, so this keeps
interictal windows roughly >= 1 h away from seizures). Raw class ratio is ~250:1;
this subsample is 1:1. Seizure intervals are parsed here into data_structure so
loader.py never re-parses the summaries.

Channels: recorded as bipolar 'double banana' derivations. Each is labelled by
its FIRST electrode (FP1-F7 -> Fp1) so it maps into the 10-10 channel space
(decided 2026-09-24); the signal stays the bipolar difference. Of the 23
derivations, duplicate first electrodes (FP1-F3, FP2-F8, P7-T7, T7-FT9, second
T8-P8) are dropped and FT9-FT10 / FT10-T8 are dropped (FT9/FT10 are not in the
64-channel 10-10 set), leaving 16. Recordings with a different montage are
picked by name; a missing derivation is zero-filled.

24 cases (chb01..chb24; chb21 is chb01 recorded 1.5 years later, kept as its own
subject as PhysioNet lists it). Subject key = case number ('chb05' -> '5').
"""
import json
import os
import re
from collections import OrderedDict

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "https://physionet.org/content/chbmit/1.0.0/",
    "file_format": "EDF",
    "description": (
        "23 pediatric patients with intractable seizures (24 cases), Children's "
        "Hospital Boston, 256 Hz scalp EEG in bipolar double-banana montage, "
        "198 annotated seizures. Binary ictal vs interictal detection, 5 s "
        "windows, class-balanced per subject."
    ),
    "task_type": "seizure_detection",
    "reference": "Shoeb A. Application of Machine Learning to Epileptic Seizure Onset "
                 "Detection and Treatment. PhD Thesis, MIT, 2009. Goldberger et al. "
                 "(2000) PhysioBank, PhysioToolkit, and PhysioNet. Circulation "
                 "101(23):e215-e220.",
    "notes": (
        "Balanced 1:1 subsample: all non-overlapping 5 s windows fully inside "
        "seizures + equal count of interictal windows (fixed seed per subject) "
        "from seizure-free recordings not adjacent to a seizure recording. "
        "Bipolar derivations labelled by first electrode; 16 channels kept."
    ),
}

# (10-10 label of the first electrode, bipolar derivation in the EDF)
_DERIV = [
    ("Fp1", "FP1-F7"), ("F7", "F7-T7"), ("T7", "T7-P7"), ("P7", "P7-O1"),
    ("F3", "F3-C3"), ("C3", "C3-P3"), ("P3", "P3-O1"),
    ("Fp2", "FP2-F4"), ("F4", "F4-C4"), ("C4", "C4-P4"), ("P4", "P4-O2"),
    ("F8", "F8-T8"), ("T8", "T8-P8"), ("P8", "P8-O2"),
    ("Fz", "FZ-CZ"), ("Cz", "CZ-PZ"),
]
CHANNELS = OrderedDict((str(i + 1), {"label": lab, "original_label": deriv})
                       for i, (lab, deriv) in enumerate(_DERIV))

TARGETS = {
    "count": 2,
    "type": "seizure_detection",
    "0": {"label": "interictal"},
    "1": {"label": "ictal (seizure)"},
}

_FILE = re.compile(r"File Name:\s*(\S+)")
_START = re.compile(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds")
_END = re.compile(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds")


def parse_summary(path):
    """-> {edf_filename: [[start_s, end_s], ...]} for every file the summary lists."""
    seizures, cur, starts = {}, None, []
    with open(path, errors="replace") as f:
        for line in f:
            if m := _FILE.search(line):
                cur = m.group(1)
                seizures.setdefault(cur, [])
            elif (m := _START.search(line)) and cur:
                starts.append(int(m.group(1)))
            elif (m := _END.search(line)) and cur and starts:
                seizures[cur].append([starts.pop(0), int(m.group(1))])
    return seizures


def build_data_structure(raw_dir):
    with open(os.path.join(raw_dir, "RECORDS")) as f:
        records = [ln.strip() for ln in f if ln.strip().endswith(".edf")]
    by_case = OrderedDict()
    for rec in records:
        case, fname = rec.split("/")
        by_case.setdefault(case, []).append(fname)
    structure = OrderedDict()
    for case, files in by_case.items():
        summary = os.path.join(raw_dir, case, f"{case}-summary.txt")
        if not os.path.exists(summary):
            print(f"  [Warning] {case}: no summary file yet, skipping")
            continue
        seiz = parse_summary(summary)
        structure[str(int(case[3:]))] = {
            "folder": f"raw/{case}",
            "runs": [{"file": fn, "seizures": seiz.get(fn, [])} for fn in sorted(files)],
        }
    return structure


def main():
    raw_dir = os.path.join(ROOT, "raw")
    structure = build_data_structure(raw_dir)
    n_seiz = sum(len(r["seizures"]) for v in structure.values() for r in v["runs"])
    meta = {
        "data_metadata": {
            "dataset_name": "CHB_MIT",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 256,
                "window_size_seconds": 5.0,
                "num_subjects": len(structure),
                "num_recordings": sum(len(v["runs"]) for v in structure.values()),
                "num_seizures": n_seiz,
            },
            "targets": TARGETS,
            "channels": {"count": len(CHANNELS), "system": "bipolar double banana, first-electrode labels", **CHANNELS},
        },
        "data_structure": structure,
    }
    out_path = os.path.join(ROOT, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"wrote {out_path}: {len(structure)} subjects, "
          f"{meta['data_metadata']['acquisition']['num_recordings']} recordings, {n_seiz} seizures")


if __name__ == "__main__":
    main()
