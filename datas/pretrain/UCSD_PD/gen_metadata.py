"""
UC San Diego resting-state EEG, Parkinson's disease vs healthy controls (Rockhill,
Jackson, Swann et al.), OpenNeuro ds002778 (BIDS) -- metadata.json generator.
See docs/agents/adding-a-dataset.md Step 3.

Source verified (2026-09-24) against the README, participants.tsv, *_eeg.json and
*_channels.tsv: 31 subjects -- 16 healthy (sub-hcN, one session 'hc') and 15 PD
(sub-pdN, sessions 'off' and 'on' medication), 46 BioSemi .bdf recordings, all 512 Hz,
~3 min eyes-open rest each (~151 min total). 32 scalp channels (10-20 names, all in
the 10-10 set) plus EXG1-8 and Status, which are left out. Subject numbers are unique
across hc/pd, so the key is just the number.

Added for pretraining: each recording is cut into non-overlapping 5 s windows, label =
0 healthy, 1 PD off medication, 2 PD on medication. BDF from BioSemi has no reference
(CMS/DRL), so the loader re-references to the average of the 32 scalp channels.
"""
import glob
import json
import os
import re
from collections import OrderedDict

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "https://openneuro.org/datasets/ds002778",
    "file_format": "BIDS (BDF)",
    "description": (
        "31 subjects (16 healthy controls, 15 Parkinson's disease recorded off and on "
        "dopaminergic medication), 32-channel BioSemi EEG at 512 Hz, ~3 min eyes-open rest."
    ),
    "task_type": "clinical_pd",
    "reference": "Rockhill AP, Jackson N, George J, Aron A, Swann NC (2020). UC San Diego "
                 "Resting State EEG Data from Patients with Parkinson's Disease. OpenNeuro "
                 "ds002778. doi:10.18112/openneuro.ds002778. Jackson N et al. (2019) eNeuro "
                 "6(3); Swann NC et al. (2015) Ann Neurol 78(5):742-50.",
    "notes": (
        "5 s non-overlapping windows, label 0 healthy / 1 PD off meds / 2 PD on meds. "
        "Average-referenced over the 32 scalp channels in the loader. Authors ask that "
        "PD-vs-HC classification not be reported from this dataset alone (see README)."
    ),
}

CHANNEL_LABELS = [
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3", "CP1", "CP5", "P7", "P3", "Pz",
    "PO3", "O1", "Oz", "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8", "FC6", "FC2",
    "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
]

TARGETS = {
    "count": 3,
    "type": "clinical_group",
    "0": {"label": "healthy control"},
    "1": {"label": "Parkinson's disease, off medication"},
    "2": {"label": "Parkinson's disease, on medication"},
}

SESSION_LABEL = {"hc": 0, "off": 1, "on": 2}


def build_data_structure(raw_dir):
    structure = {}
    for bdf in glob.glob(os.path.join(raw_dir, "sub-*", "ses-*", "eeg", "*_eeg.bdf")):
        m = re.match(r"sub-(?:hc|pd)(\d+)_ses-(hc|off|on)_", os.path.basename(bdf))
        structure.setdefault(str(int(m.group(1))), {"runs": []})["runs"].append(
            {"file": "raw/" + os.path.relpath(bdf, raw_dir), "label": SESSION_LABEL[m.group(2)]})
    for v in structure.values():
        v["runs"].sort(key=lambda r: r["file"])
    return OrderedDict(sorted(structure.items(), key=lambda kv: int(kv[0])))


def main():
    structure = build_data_structure(os.path.join(ROOT, "raw"))
    channels = OrderedDict((str(i + 1), {"label": c}) for i, c in enumerate(CHANNEL_LABELS))
    meta = {
        "data_metadata": {
            "dataset_name": "UCSD_PD",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 512,
                "window_size_seconds": 5.0,
                "num_subjects": len(structure),
            },
            "targets": TARGETS,
            "channels": {"count": len(channels), "system": "BioSemi 32 (10-20)", **channels},
        },
        "data_structure": structure,
    }
    out_path = os.path.join(ROOT, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"wrote {out_path}: {len(structure)} subjects, "
          f"{sum(len(v['runs']) for v in structure.values())} recordings")


if __name__ == "__main__":
    main()
