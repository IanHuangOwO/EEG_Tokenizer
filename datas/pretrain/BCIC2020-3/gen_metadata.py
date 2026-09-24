"""
2020 BCI Competition, Track 3 (imagined speech), NEMAR nm000113 v1.0.0 (BIDS) --
metadata.json generator. See docs/agents/adding-a-dataset.md Step 3.

Source verified (2026-09-24) against nemar.org/dataset/nm000113, the dataset README
and the extracted BIDS files: 15 subjects, 64 channels (extended 10-20, same list in
all 45 channels.tsv), 256 Hz, 5 imagined-speech classes (Hello / Help me / Stop /
Thank you / Yes; events.tsv value 1..5). Three runs per subject: run-00 training
(300 trials), run-01 validation (50), run-02 test (50; labels from the competition's
answer sheet). The original epoched .mat data was concatenated into continuous EDF:
every trial is a 3.10546875 s (795-sample) epoch starting at events.tsv 'sample',
placed back to back. Units are Volts (the BIDS conversion scaled by 1e-6).

Added for pretraining: one window per trial with its real class label (value - 1),
all three runs. channels.tsv/events.tsv carry a UTF-8 BOM (read with utf-8-sig).
TP9/TP10/PO9/PO10/FT9/FT10 are kept here but are not in the 64-channel 10-10 set,
so channel unification drops them at train time.
"""
import csv
import glob
import json
import os
from collections import OrderedDict

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "https://nemar.org/dataset/nm000113",
    "file_format": "BIDS (EDF + events.tsv)",
    "description": (
        "15 subjects, 64-channel EEG at 256 Hz, imagined speech of 5 commands "
        "(Hello, Help me, Stop, Thank you, Yes); 300 training + 50 validation + "
        "50 test trials per subject, 3.1 s each."
    ),
    "task_type": "imagined_speech",
    "reference": "Lee S, Mueller K-R, Millan JdR (2026). 2020 BCI competition, track 3 "
                 "(v1.0.0). NEMAR. doi:10.82901/nemar.nm000113. Competition: "
                 "https://osf.io/pq7vb/",
    "notes": (
        "One window per trial (795 samples, 3.1 s), real 5-class labels, all three "
        "runs (train/validation/test) used. EDF units are Volts."
    ),
}

TARGETS = {
    "count": 5,
    "type": "imagined_speech",
    "0": {"label": "Hello"},
    "1": {"label": "Help me"},
    "2": {"label": "Stop"},
    "3": {"label": "Thank you"},
    "4": {"label": "Yes"},
}


def read_channels(raw_dir):
    path = sorted(glob.glob(os.path.join(raw_dir, "sub-*", "eeg", "*_channels.tsv")))[0]
    with open(path, encoding="utf-8-sig") as f:
        return [r["name"] for r in csv.DictReader(f, delimiter="\t")]


def build_data_structure(raw_dir):
    structure = OrderedDict()
    for sub_dir in sorted(glob.glob(os.path.join(raw_dir, "sub-*"))):
        sub = os.path.basename(sub_dir)
        runs = []
        for edf in sorted(glob.glob(os.path.join(sub_dir, "eeg", "*_eeg.edf"))):
            base = edf[: -len("_eeg.edf")]
            rel = lambda p: "raw/" + os.path.relpath(p, raw_dir)
            runs.append({"edf": rel(edf), "events": rel(base + "_events.tsv")})
        structure[str(int(sub.split("-")[1]))] = {"runs": runs}
    return structure


def main():
    raw_dir = os.path.join(ROOT, "raw")
    ch = read_channels(raw_dir)
    channels = OrderedDict((str(i + 1), {"label": c}) for i, c in enumerate(ch))
    structure = build_data_structure(raw_dir)
    meta = {
        "data_metadata": {
            "dataset_name": "BCIC2020-3",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 256,
                "window_size_seconds": 795 / 256,
                "num_subjects": len(structure),
            },
            "targets": TARGETS,
            "channels": {"count": len(channels), "system": "extended 10-20", **channels},
        },
        "data_structure": structure,
    }
    out_path = os.path.join(ROOT, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"wrote {out_path}: {len(structure)} subjects, "
          f"{sum(len(v['runs']) for v in structure.values())} runs, {len(channels)} channels")


if __name__ == "__main__":
    main()
