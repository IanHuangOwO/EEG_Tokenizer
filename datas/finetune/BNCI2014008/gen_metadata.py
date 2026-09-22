"""
BNCI2014008 ("P300 Speller with patients with ALS", BNCI Horizon 2020
008-2014) -- metadata.json generator. See docs/agents/adding-a-dataset.md
Step 3.

Source verified directly (2026-09-22) via the dataset's own description.pdf
(bnci-horizon-2020.eu/database/data-sets/008-2014/description.pdf, redirects
to lampx.tugraz.at/~bci/database/008-2014/) plus a real .mat file's struct
fields (scipy.io.loadmat, struct_as_record=False).

8 subjects (all diagnosed with ALS, per description.pdf's Table I -- real
demographic fact, not filler), 8ch (10-10 system: Fz/Cz/Pz/Oz/P3/P4/PO7/
PO8), 256 Hz (stated in description.pdf, not present as a struct field in
the .mat itself). Each A0N.mat has ONE top-level `data` struct (all 7 runs
already concatenated into one continuous X, unlike BNCI2014009's per-run
array) with `X` ([T,8]), `y` (per-sample sparse trigger, 0=none,
1=NonTarget, 2=Target), `channels`, plus per-subject clinical fields
(age/gender/ALSfrs/onsetALS) not used here. Verified directly: exactly 4200
rising-edge stimulus events/subject (33600 across all 8), matching Table
14's reported total exactly.
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "http://bnci-horizon-2020.eu/database/data-sets/008-2014",
    "file_format": "MATLAB (.mat)",
    "description": (
        "8 subjects with ALS, 8ch EEG, P300 matrix speller (6x6, "
        "row/column intensification) target-vs-non-target classification."
    ),
    "task_type": "P300 (Event-Related Potential)",
    "reference": "Riccio A, Simione L, Schettini F, Pizzimenti A, Inghilleri "
                 "M, Belardinelli MO, Mattia D, Cincotti F. Attention and "
                 "P300-based BCI performance in people with amyotrophic "
                 "lateral sclerosis. Front Hum Neurosci. 2013;7:732.",
    "contact": "BNCI Horizon 2020, http://bnci-horizon-2020.eu",
    "notes": (
        "Real per-flash target/non-target label from `y`'s rising edges "
        "(see this file's module docstring). Epoch window hardcoded in "
        "loader.py to -0.2s/+0.8s (P300 scale, matches Table 14's 1.0s "
        "trial length exactly), not compile.json's global pre/"
        "post_event_seconds (tuned for ~5s motor-imagery trials). Clinical "
        "metadata (age/gender/ALSfrs-R/onset site) present in the raw .mat "
        "but not carried into this pipeline."
    ),
}

CHANNELS = {
    str(i + 1): {"label": label} for i, label in enumerate([
        "Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8",
    ])
}

TARGETS = {
    "count": 2,
    "type": "p300",
    "0": {"label": "non-target"},
    "1": {"label": "target"},
}


def build_data_structure(raw_dir):
    structure = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "A*.mat"))):
        m = re.match(r"A(\d+)\.mat$", os.path.basename(path))
        if not m:
            continue
        n = int(m.group(1))
        structure[str(n)] = {"file": f"raw/A{n:02d}.mat"}
    return structure


def main():
    raw_dir = os.path.join(ROOT, "raw")
    structure = build_data_structure(raw_dir)

    meta = {
        "data_metadata": {
            "dataset_name": "BNCI2014008",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 256,
                "window_size_seconds": 1.0,
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
    print(f"wrote {out_path}: {len(structure)} subjects")


if __name__ == "__main__":
    main()
