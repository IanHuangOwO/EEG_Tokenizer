"""
SRM Resting-state EEG (OpenNeuro ds003775, BIDS) -- metadata.json generator.
See docs/agents/adding-a-dataset.md Step 3.

Source verified directly (2026-09-22) via the dataset's own BIDS sidecar files
(dataset_description.json, README, one subject's *_eeg.json/*_channels.tsv,
participants.tsv) fetched straight from the public S3 bucket
(s3://openneuro.org/ds003775/, unsigned/no-auth) -- not guessed from memory.

111 healthy-control subjects, BioSemi ActiveTwo, 64 channels, extended 10-20
(10-10) layout, 1024 Hz, 4 minutes eyes-closed continuous resting-state per
session. Some subjects have a second session ("ses-t2") at a later date; both
are used, one "trial" per available session (shape C: multiple files per
subject, see loader.py). No events/conditions of any kind -- this is
unlabelled continuous data (targets.count=1, dummy label 0 throughout);
self-supervised pretraining only, never supervised finetune/eval.

Channel order/list and the 1024 Hz sample rate come straight from one
subject's real *_channels.tsv/*_eeg.json (identical across subjects per BIDS
convention for one recording device/montage). All 64 names resolve via MNE's
standard_1020 montage (checked 2026-09-22), so per-channel "coordinates" are
omitted, same precedent as PhysionetMI/DEAP.
"""
import json
import os
import re

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "https://openneuro.org/datasets/ds003775/versions/1.2.1",
    "file_format": "EDF (BIDS)",
    "description": (
        "111 healthy control subjects, 4 minutes of continuous eyes-closed "
        "resting-state EEG, extracted from the Stimulus-Selective Response "
        "Modulation (SRM) project. BioSemi ActiveTwo, 64 channels, average "
        "referenced, unfiltered (50 Hz line noise, European recording). Some "
        "subjects have a second session ('ses-t2') at a later date."
    ),
    "task_type": "resting_state",
    "reference": "Rygvold T, Hatlestad-Hall C et al. (2021), "
                  "http://dx.doi.org/10.1111/ejn.14964",
    "contact": "Christoffer Hatlestad-Hall, Trine Waage Rygvold, Stein Andersson "
               "(Dept. of Psychology, University of Oslo)",
    "notes": (
        "No events/conditions -- the whole continuous segment is resting-state "
        "data (README: 'The files contain no events'). targets.count is a "
        "placeholder (1 class, dummy label 0 from loader.py), not a real task -- "
        "self-supervised pretraining only, see adding-a-dataset.md's "
        "'Continuous multi-label event annotations' bullet for the same pattern "
        "(GraspAndLift_Train). License: CC0. Paper abstract also mentions "
        "single-session recordings from 52 additional individuals -- NOT present "
        "in this deposit (only the 111 ses-t1/ses-t2 subjects are here), possibly "
        "a separate OpenNeuro/figshare item; not investigated further as of "
        "2026-09-22. sub-041's ses-t1 EDF has a malformed header timestamp "
        "('second must be in 0..59', rejected by MNE) -- loader.py skips that one "
        "session; subject 41 has zero usable sessions and is absent from the "
        "compiled cache (110/111 subjects, 152 sessions total)."
    ),
}

# BIDS channels.tsv order, verified against sub-001 (2026-09-22).
CHANNELS = {
    str(i + 1): {"label": label} for i, label in enumerate([
        "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3",
        "FC1", "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1",
        "P3", "P5", "P7", "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz",
        "Pz", "CPz", "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4",
        "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCz", "Cz", "C2", "C4",
        "C6", "T8", "TP8", "CP6", "CP4", "CP2", "P2", "P4", "P6", "P8",
        "P10", "PO8", "PO4", "O2",
    ])
}

TARGETS = {
    "count": 1,
    "type": "resting_state",
    "0": {"label": "resting-state (no real classes; pretrain-only, see dataset_info.notes)"},
}

SES_PATTERN = "sub-{sub}_ses-{ses}_task-resteyesc_eeg.edf"


def build_data_structure(raw_dir):
    """One entry per subject (sub-NNN dirs), one file per session that actually
    exists (ses-t1 always, ses-t2 if present) -- shape C, see Step 4."""
    structure = {}
    for dname in sorted(os.listdir(raw_dir)):
        m = re.match(r"sub-(\d+)$", dname)
        if not m:
            continue
        sub = m.group(1)
        files = []
        for ses in ("t1", "t2"):
            rel = f"raw/sub-{sub}/ses-{ses}/eeg/{SES_PATTERN.format(sub=sub, ses=ses)}"
            if os.path.exists(os.path.join(ROOT, rel)):
                files.append(rel)
        if files:
            structure[str(int(sub))] = {"files": files}
    return structure


def main():
    raw_dir = os.path.join(ROOT, "raw")
    structure = build_data_structure(raw_dir)

    meta = {
        "data_metadata": {
            "dataset_name": "SRM_RestingState",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 1024,
                "window_size_seconds": 240.0,
                "num_subjects": len(structure),
            },
            "targets": TARGETS,
            "channels": {"count": len(CHANNELS), "system": "10-20 International System (10-10)", **CHANNELS},
        },
        "data_structure": structure,
    }

    out_path = os.path.join(ROOT, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    n_sessions = sum(len(v["files"]) for v in structure.values())
    print(f"wrote {out_path}: {len(structure)} subjects, {n_sessions} sessions")


if __name__ == "__main__":
    main()
