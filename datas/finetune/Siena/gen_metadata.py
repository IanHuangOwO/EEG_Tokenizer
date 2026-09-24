"""
Siena Scalp EEG Database (PhysioNet siena-scalp-eeg 1.0.0) -- metadata.json
generator. See docs/agents/adding-a-dataset.md Step 3.

Source verified directly (2026-09-24) from the downloaded files: RECORDS (41
EDF recordings, authoritative file list), subject_info.csv (14 patients,
PN00..PN17 with gaps), each patient's Seizures-list-PNxx.txt header (512 Hz,
channel list) and the EDF headers themselves (mne.io.read_raw_edf).

Used here for SELF-SUPERVISED PRETRAINING ONLY: every recording is chopped
into non-overlapping 5 s windows with dummy label 0 (same approach as
GraspAndLift_Train). The seizure annotations are not read; a supervised
seizure-detection loader would need writing before this is usable for
finetune/eval.

29 referential scalp EEG channels (10-20 plus Fc/Cp/F9/F10). The EDF channel
ORDER and CASE differ between files ('EEG Fp2' vs 'EEG FP2', extra P9/P10 or
polygraphy channels in some), so loader.py picks channels by name, not by
index. PN10 was recorded with only 20 EEG channels (subject_info.csv
eeg_channel=20) -- its missing channels are zero-filled.
"""
import json
import os
from collections import OrderedDict

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "https://physionet.org/content/siena-scalp-eeg/1.0.0/",
    "file_format": "EDF",
    "description": (
        "14 adult epilepsy patients (Unit of Neurology and Neurophysiology, "
        "University of Siena), 41 continuous video-EEG recordings (~128 h), "
        "512 Hz, 29 scalp EEG channels, 47 annotated seizures."
    ),
    "task_type": "pretrain_continuous",
    "reference": "Detti P. (2020) Siena Scalp EEG Database (version 1.0.0). PhysioNet. "
                 "Detti P, Vatti G, Zabalo Manrique de Lara G. EEG Synchronization "
                 "Analysis for Seizure Prediction: A Study on Data of Noninvasive "
                 "Recordings. Processes 2020;8(7):846.",
    "notes": (
        "Pretraining only: 5 s non-overlapping windows, dummy label 0; seizure "
        "annotations are not used. Channels picked by name per file (order/case "
        "vary between files). PN10 has only 20 EEG channels; the rest are "
        "zero-filled for that subject. Subject keys are the PN number without "
        "leading zeros (PN00 -> '0')."
    ),
}

# (standard 10-10 label, raw EDF label without the 'EEG ' prefix)
_EEG = [
    ("Fp1", "Fp1"), ("F3", "F3"), ("C3", "C3"), ("P3", "P3"), ("O1", "O1"),
    ("F7", "F7"), ("T7", "T3"), ("P7", "T5"), ("FC1", "Fc1"), ("FC5", "Fc5"),
    ("CP1", "Cp1"), ("CP5", "Cp5"), ("F9", "F9"), ("Fz", "Fz"), ("Cz", "Cz"),
    ("Pz", "Pz"), ("Fp2", "Fp2"), ("F4", "F4"), ("C4", "C4"), ("P4", "P4"),
    ("O2", "O2"), ("F8", "F8"), ("T8", "T4"), ("P8", "T6"), ("FC2", "Fc2"),
    ("FC6", "Fc6"), ("CP2", "Cp2"), ("CP6", "Cp6"), ("F10", "F10"),
]
CHANNELS = OrderedDict((str(i + 1), {"label": lab, "original_label": raw})
                       for i, (lab, raw) in enumerate(_EEG))

TARGETS = {
    "count": 1,
    "type": "pretrain_dummy",
    "0": {"label": "dummy (continuous EEG window, no task label)"},
}


def build_data_structure(raw_dir):
    """Groups RECORDS (e.g. 'PN00/PN00-1.edf') by patient folder."""
    structure = OrderedDict()
    with open(os.path.join(raw_dir, "RECORDS")) as f:
        records = [ln.strip() for ln in f if ln.strip().endswith(".edf")]
    for rec in records:
        folder, fname = rec.split("/")
        key = str(int(folder[2:]))
        structure.setdefault(key, {"folder": f"raw/{folder}", "runs": []})["runs"].append(fname)
    return structure


def main():
    raw_dir = os.path.join(ROOT, "raw")
    structure = build_data_structure(raw_dir)
    meta = {
        "data_metadata": {
            "dataset_name": "Siena",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 512,
                "window_size_seconds": 5.0,
                "num_subjects": len(structure),
                "num_recordings": sum(len(v["runs"]) for v in structure.values()),
            },
            "targets": TARGETS,
            "channels": {"count": len(CHANNELS), "system": "10-20 extended (Fc/Cp/F9/F10)", **CHANNELS},
        },
        "data_structure": structure,
    }
    out_path = os.path.join(ROOT, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"wrote {out_path}: {len(structure)} subjects, "
          f"{meta['data_metadata']['acquisition']['num_recordings']} recordings")


if __name__ == "__main__":
    main()
