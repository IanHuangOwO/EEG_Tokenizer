"""
STEW: Simultaneous Task EEG Workload dataset (IEEE DataPort, open access) --
metadata.json generator. See docs/agents/adding-a-dataset.md Step 3.

Source verified (2026-09-24) against the dataset page
(ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset) and the
extracted files: 48 subjects, Emotiv EPOC, 128 Hz, 14 channels in the column order
AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4. Per subject two plain-text files,
subNN_lo.txt (2.5 min at rest) and subNN_hi.txt (2.5 min during the SIMKAP
multitasking test): 19200 rows (150 s x 128 Hz) x 14 whitespace-separated columns,
raw Emotiv values with the device's ~4000 DC offset (removed by the compile-time
bandpass).

Added for pretraining. Each recording is cut into non-overlapping 5 s windows; the
label is the real condition (0 = rest 'lo', 1 = multitask 'hi') rather than a dummy,
so the dataset can also serve a rest-vs-workload task later. ratings.txt (per-subject
self-reported workload, 1-9, for rest and test; missing for subjects 5, 24, 42) is not
used.
"""
import json
import os
import re
from collections import OrderedDict

ROOT = os.path.dirname(__file__)
DATA_DIR = "STEW Dataset"

DATASET_INFO = {
    "source_url": "https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset",
    "file_format": "plain text (whitespace-separated, 14 columns)",
    "description": (
        "48 subjects, Emotiv EPOC 14-channel EEG at 128 Hz: 2.5 min at rest and "
        "2.5 min during the SIMKAP multitasking workload test."
    ),
    "task_type": "mental_workload",
    "reference": "Lim WL, Sourina O, Wang LP. STEW: Simultaneous Task EEG Workload Data Set. "
                 "IEEE Trans Neural Syst Rehabil Eng 2018;26(11):2106-2114. "
                 "doi:10.1109/TNSRE.2018.2872924",
    "notes": (
        "5 s non-overlapping windows; label 0 = rest (lo), 1 = multitask test (hi). "
        "ratings.txt (self-reported workload 1-9) not used; ratings missing for "
        "subjects 5, 24, 42. Values are raw Emotiv units with a ~4000 DC offset."
    ),
}

_CH = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
CHANNELS = OrderedDict((str(i + 1), {"label": c}) for i, c in enumerate(_CH))

TARGETS = {
    "count": 2,
    "type": "mental_workload",
    "0": {"label": "rest (lo)"},
    "1": {"label": "multitask test (hi)"},
}


def build_data_structure(raw_dir):
    structure = OrderedDict()
    for fname in sorted(os.listdir(raw_dir)):
        m = re.match(r"sub(\d+)_(lo|hi)\.txt$", fname)
        if not m:
            continue
        key = str(int(m.group(1)))
        structure.setdefault(key, {"files": []})["files"].append(
            {"file": f"raw/{DATA_DIR}/{fname}", "label": 0 if m.group(2) == "lo" else 1})
    return structure


def main():
    structure = build_data_structure(os.path.join(ROOT, "raw", DATA_DIR))
    meta = {
        "data_metadata": {
            "dataset_name": "STEW",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 128,
                "window_size_seconds": 5.0,
                "num_subjects": len(structure),
            },
            "targets": TARGETS,
            "channels": {"count": len(CHANNELS), "system": "10-20 (Emotiv EPOC)", **CHANNELS},
        },
        "data_structure": structure,
    }
    out_path = os.path.join(ROOT, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"wrote {out_path}: {len(structure)} subjects, "
          f"{sum(len(v['files']) for v in structure.values())} recordings")


if __name__ == "__main__":
    main()
