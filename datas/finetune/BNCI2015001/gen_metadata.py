"""
BNCI2015001 ("Autocalibration and recurrent adaptation: Towards a plug and
play online ERD-BCI", BNCI Horizon 2020 001-2015) -- metadata.json
generator. See docs/agents/adding-a-dataset.md Step 3.

Source verified directly (2026-09-22) via the dataset's own description.pdf
(bnci-horizon-2020.eu/database/data-sets/001-2015/description.pdf, redirects
to lampx.tugraz.at/~bci/database/001-2015/) plus a real .mat file's struct
fields (scipy.io.loadmat, struct_as_record=False) and an HTTP existence
probe of every subject/session combination.

12 subjects, 13ch (10-10 system, Laplacian-derivation-adjacent electrodes
around C3/Cz/C4: FC3/FCz/FC4/C5/C3/C1/Cz/C2/C4/C6/CP3/CPz/CP4), 512 Hz
(matches Table 14 exactly). One .mat file PER SESSION (not per subject):
subjects S01-S07 and S12 have 2 sessions (A, B); S08-S11 have a 3rd (C,
recorded for subjects who didn't reach the 70% accuracy criterion in 2
sessions, per description.pdf section 3.4) -- verified via real HTTP HEAD/
GET probes of every S{01-12}{A,B,C}.mat combination, not assumed. 28 files
total.

Real 2-class label from each file's `y` array (1=right hand -> 0, 2=both
feet -> 1), one label per trial (200 trials/session: 5 runs x 40 trials).
Epoch window hardcoded in loader.py to the paradigm's cue-to-end-of-imagery
window (description.pdf Figure 1: cue at second 3, imagery period ends at
second 8) -- offset 3.0s, duration 5.0s, matching Table 14's 5s trial
length exactly.
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "http://bnci-horizon-2020.eu/database/data-sets/001-2015",
    "file_format": "MATLAB (.mat)",
    "description": (
        "12 subjects, 13ch EEG, 2-class motor imagery (right hand vs both "
        "feet), synchronous Graz BCI paradigm, 2-3 sessions/subject on "
        "different days."
    ),
    "task_type": "motor_imagery",
    "reference": "Faller J, Vidaurre C, Solis-Escalante T, Neuper C, Scherer "
                 "R (2012). Autocalibration and recurrent adaptation: "
                 "Towards a plug and play online ERD-BCI. IEEE Trans Neural "
                 "Syst Rehabil Eng. 20(3):313-319. "
                 "doi:10.1109/tnsre.2012.2189584",
    "contact": "Reinhold Scherer (reini.scherer@gmail.com), Josef Faller "
               "(josef.faller@gmx.at)",
    "notes": (
        "2 sessions/subject (S01-S07, S12) or 3 (S08-S11, a 3rd was "
        "recorded for subjects who hadn't reached the 70% criterion after "
        "2) -- confirmed via real HTTP probes of every subject/session "
        "combination, not assumed from the paper text alone. Real trial "
        "label from each file's `y` array (1=right hand, 2=both feet). "
        "Epoch window hardcoded in loader.py to offset 3.0s/duration 5.0s "
        "post-trial-onset (the cue-to-end-of-imagery window, description."
        "pdf Figure 1), NOT compile.json's global pre/post_event_seconds."
    ),
}

CHANNELS = {
    str(i + 1): {"label": label} for i, label in enumerate([
        "FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CPz", "CP4",
    ])
}

TARGETS = {
    "count": 2,
    "type": "motor_imagery",
    "0": {"label": "right hand"},
    "1": {"label": "both feet"},
}

# Verified 2026-09-22 via HTTP probe of every S{01-12}{A,B,C}.mat -- not assumed.
SESSIONS_PER_SUBJECT = {
    "1": "AB", "2": "AB", "3": "AB", "4": "AB", "5": "AB", "6": "AB", "7": "AB",
    "8": "ABC", "9": "ABC", "10": "ABC", "11": "ABC",
    "12": "AB",
}


def build_data_structure(raw_dir):
    structure = {}
    for sub, sessions in SESSIONS_PER_SUBJECT.items():
        files = [f"raw/S{int(sub):02d}{ses}.mat" for ses in sessions
                 if os.path.exists(os.path.join(raw_dir, f"S{int(sub):02d}{ses}.mat"))]
        if files:
            structure[sub] = {"files": files}
    return structure


def main():
    raw_dir = os.path.join(ROOT, "raw")
    structure = build_data_structure(raw_dir)

    meta = {
        "data_metadata": {
            "dataset_name": "BNCI2015001",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 512,
                "window_size_seconds": 5.0,
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
    n_files = sum(len(v["files"]) for v in structure.values())
    print(f"wrote {out_path}: {len(structure)} subjects, {n_files} sessions")


if __name__ == "__main__":
    main()
