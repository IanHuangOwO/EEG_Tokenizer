"""
Auditory Attention Detection Dataset KULeuven (Das, Francart & Bertrand), Zenodo
record 3997352 -- metadata.json generator. See docs/agents/adding-a-dataset.md Step 3.

Source verified (2026-09-24) against raw/README.txt.txt and the files: 16 subjects,
one MATLAB v7.3 (HDF5) file per subject, 'trials' = 20 trials each (8 x ~6.5 min
from experiments 1-2, 12 x ~2 min repetitions from experiment 3; ~77 min/subject).
Each trial: RawData.EegData (64 channels x samples), already 0.5 Hz high-passed and
downsampled to 128 Hz, units uV. Channel labels (FileHeader.Channels.Label) are the
BioSemi 64 order, all already 10-10 names.

Added for pretraining: each trial is cut into non-overlapping 5 s windows, label =
attended ear (0 = left, 1 = right). The authors warn that trial identity leaks into
short windows (trial-specific patterns, eye-gaze bias), so this label is not a valid
AAD benchmark under window-level splits -- it's only kept as a placeholder label here.
"""
import glob
import json
import os
import re
from collections import OrderedDict

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "https://zenodo.org/records/3997352",
    "file_format": "MATLAB v7.3 (HDF5)",
    "description": (
        "16 normal-hearing subjects, 64-channel BioSemi EEG (downsampled to 128 Hz, "
        "0.5 Hz high-pass), listening to one of two competing Dutch stories presented "
        "dichotically or HRTF-filtered; 20 trials (~77 min) per subject."
    ),
    "task_type": "auditory_attention",
    "reference": "Das N, Francart T, Bertrand A (2020). Auditory Attention Detection "
                 "Dataset KULeuven. Zenodo. doi:10.5281/zenodo.3997352. Biesmans W et al. "
                 "(2016) IEEE TNSRE 25(5):402-412.",
    "notes": (
        "5 s non-overlapping windows, label = attended ear (0 left, 1 right). Units uV. "
        "Trial identity leaks into short windows (see README), so don't use this label "
        "as a finetune benchmark without trial-level splits."
    ),
}

CHANNEL_LABELS = [
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1", "C1", "C3",
    "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9", "PO7", "PO3",
    "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2",
    "F4", "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8",
    "TP8", "CP6", "CP4", "CP2", "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
]

TARGETS = {
    "count": 2,
    "type": "attended_ear",
    "0": {"label": "left"},
    "1": {"label": "right"},
}


def main():
    files = glob.glob(os.path.join(ROOT, "raw", "S*.mat"))
    structure = OrderedDict(
        (str(int(re.match(r"S(\d+)\.mat$", os.path.basename(p)).group(1))),
         {"file": "raw/" + os.path.basename(p)})
        for p in sorted(files, key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0])))
    channels = OrderedDict((str(i + 1), {"label": c}) for i, c in enumerate(CHANNEL_LABELS))
    meta = {
        "data_metadata": {
            "dataset_name": "AAD_KUL",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 128,
                "window_size_seconds": 5.0,
                "num_subjects": len(structure),
            },
            "targets": TARGETS,
            "channels": {"count": len(channels), "system": "BioSemi 64 (10-10)", **channels},
        },
        "data_structure": structure,
    }
    out_path = os.path.join(ROOT, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"wrote {out_path}: {len(structure)} subjects, {len(channels)} channels")


if __name__ == "__main__":
    main()
