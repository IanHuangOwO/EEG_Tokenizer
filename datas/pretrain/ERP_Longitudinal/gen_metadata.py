"""
ERP_Longitudinal (figshare 27201003, Nature Scientific Data companion) --
metadata.json generator. See docs/agents/adding-a-dataset.md Step 3.

Source verified directly (2026-09-22) via figshare's public API
(api.figshare.com/v2/articles/27201003, no auth) plus three small support
files fetched from it: Subject.txt (15 subjects, gender/age), Trigger.txt
(trigger code table), ChannelPosition.locs (57 EEG channel names/polar
coords, standard 10-10 names -- resolve via MNE standard_1020, same
precedent as SRM_RestingState/DEAP, coordinates omitted).

Paradigm: RSVP (rapid serial visual presentation) oddball at 10 Hz. Each
subject has 4 longitudinal session files (Day_1/Day_7/Day_80/Day_200), each
a (4, 1) cell array of 4 blocks; each block is one (T, 58) float64 matrix,
columns 1-57 = EEG, column 58 = trigger (verified directly with h5py on
S1.mat): "1"=target stimulus (client photo), "2"=non-target (generated
photo), "4"=sequence start, "9"=sequence end, 8 sequences/block x 200
stimuli/sequence = 1600 target+non-target events/block (matches exactly:
S1.mat block 'b' has 1600 trigger-1/2 events). Consecutive-stimulus sample
gap of 100 at 10 Hz -> sample_frequency=1000 (not documented anywhere,
derived directly from the trigger timing).

S1.mat also has a "GroupB" cell array (24 blocks) not covered by
Subject.txt or Trigger.txt or the article description fetched so far --
meaning unconfirmed, SKIPPED entirely (not included in data_structure)
rather than guessed. Revisit if the paper text turns up an explanation.

Files format: MATLAB v7.3 (real HDF5, unlike SPIS) -- each Day_N key is an
HDF5 object-reference array into the file's own #refs# group; loader.py
dereferences them directly since scipy.io.loadmat can't read v7.3.

Only S1.mat is downloaded as of 2026-09-22 (S2-S15 queued, ~3.2-3.9GB each,
~48GB total remaining) -- data_structure lists whichever raw/S*.mat files
actually exist on disk at gen_metadata.py run time, so subjects appear
automatically as more files land; re-run this script after each download.

Known scope gap (flagged to user 2026-09-22, unresolved): the paper
abstract describes 52 additional single-session subjects not present in
this figshare deposit (15 subjects only, Subject.txt confirms).
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "https://doi.org/10.6084/m9.figshare.27201003",
    "file_format": "MATLAB v7.3 (HDF5)",
    "description": (
        "15 subjects, longitudinal RSVP (rapid serial visual presentation) "
        "oddball ERP task at four timepoints (Day 1/7/80/200 post-enrollment). "
        "57-channel EEG, 1000 Hz (derived from trigger timing, not stated in "
        "any fetched doc). Each session = 4 blocks of 8 sequences x 200 "
        "stimuli (10 Hz), 4-6 target ('client photo') stimuli randomly placed "
        "per sequence among non-target ('generated photo') stimuli."
    ),
    "task_type": "rsvp_oddball_erp",
    "reference": "figshare 27201003 (companion Nature Scientific Data article; "
                  "exact citation not yet confirmed, article page redirected to "
                  "an auth-gated URL, not followed -- see notes)",
    "notes": (
        "Trigger semantics (column 58 of each block, verified via h5py on "
        "S1.mat): 1=target ('client photo'), 2=non-target ('generated "
        "photo'), 4=sequence start, 9=sequence end; other observed codes "
        "(3,13-16,24-26) undocumented, ignored by loader.py. GroupB cell "
        "array present in S1.mat (24 blocks) is SKIPPED -- not described in "
        "Subject.txt/Trigger.txt/the article, meaning unconfirmed as of "
        "2026-09-22. Paper abstract also describes 52 additional "
        "single-session subjects NOT present in this figshare deposit (only "
        "these 15 numbered subjects are here) -- flagged to user, unresolved. "
        "Epoch window hardcoded in loader.py to -0.2s/+0.8s (literature "
        "P300/RSVP standard), NOT compile.json's global pre_event_seconds/ "
        "post_event_seconds (those are tuned for ~5s motor-imagery trials, "
        "wrong scale for 10 Hz RSVP -- same opt-out precedent as "
        "PhysionetMI's loader)."
    ),
}

# ChannelPosition.locs order (57 channels), names normalized (trailing '.'/case
# stripped) -- all standard 10-10 names, resolve via MNE standard_1020.
CHANNELS = {
    str(i + 1): {"label": label} for i, label in enumerate([
        "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "Fz", "F1", "F2",
        "F3", "F4", "F5", "F6", "F7", "F8", "FCz", "FC1", "FC2", "FC3",
        "FC4", "FC5", "FC6", "FT7", "FT8", "Cz", "C1", "C2", "C3", "C4",
        "C5", "C6", "T7", "T8", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6",
        "TP7", "TP8", "Pz", "P3", "P4", "P5", "P6", "P7", "P8", "POz",
        "PO3", "PO4", "PO7", "PO8", "Oz", "O1", "O2",
    ])
}

TARGETS = {
    "count": 2,
    "type": "rsvp_oddball",
    "0": {"label": "non-target (generated photo, trigger 2)"},
    "1": {"label": "target (client photo, trigger 1)"},
}

SESSIONS = ("Day_1", "Day_7", "Day_80", "Day_200")


def build_data_structure(raw_dir):
    """One entry per subject with an S{n}.mat file actually on disk -- shape A
    (single file per subject), but the file itself contains multiple named
    block groups the loader dereferences internally (see Step 4 note above:
    the shape is a contract with the loader, not enforced by the pipeline)."""
    structure = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "S*.mat"))):
        m = re.match(r"S(\d+)\.mat$", os.path.basename(path))
        if not m:
            continue
        sub = m.group(1)
        structure[sub] = {"file": f"raw/S{sub}.mat"}
    return structure


def main():
    raw_dir = os.path.join(ROOT, "raw")
    structure = build_data_structure(raw_dir)

    meta = {
        "data_metadata": {
            "dataset_name": "ERP_Longitudinal",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 1000,
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
    print(f"wrote {out_path}: {len(structure)} subjects ({sorted(structure.keys())})")


if __name__ == "__main__":
    main()
