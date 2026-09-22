"""
BNCI2014009 ("Covert and Overt ERP-based BCI", BNCI Horizon 2020 009-2014,
P300 Speller / overt-attention subset only) -- metadata.json generator. See
docs/agents/adding-a-dataset.md Step 3.

Source verified directly (2026-09-22) via the dataset's own description.pdf
(bnci-horizon-2020.eu/database/data-sets/009-2014/description.pdf, redirects
to lampx.tugraz.at/~bci/database/009-2014/) plus a real .mat file's struct
fields (scipy.io.loadmat, struct_as_record=False).

10 subjects (all female, per description.pdf's Table I -- a real
demographic fact, not a data-entry omission), 16ch (10-10 system:
Fz/Cz/Pz/Oz/P3/P4/PO7/PO8/F3/F4/FCz/C3/C4/CP3/CPz/CP4), 256 Hz. Two
paradigms recorded per subject: P300 Speller (overt attention, files
"A0NS.mat") and GeoSpell (covert attention, files "A0NG.mat") -- this
dataset uses ONLY the P300 Speller ("S") files, matching MOABB's canonical
BNCI2014009 definition; GeoSpell ("G") files are a distinct paradigm not
fetched here.

Each A0NS.mat has a top-level `data` struct array of 3 runs; each run has
`X` (continuous [T,16] signal), `fs` (256), `channels`, and `y` -- a
per-SAMPLE sparse label array (0=no stimulus, 1=NonTarget, 2=Target) that
stays nonzero for a stimulus's ~125ms intensification, verified directly by
counting rising edges: exactly 576 stimulus onsets/run x 3 runs = 1728
target+non-target events/subject (17280 across all 10 subjects) --
noticeably more than Table 14's reported 5760 total (576/subject), which
this repo cannot reproduce exactly without the benchmark paper's own
loading code; using the full real trigger data here rather than guessing
which subset the paper counted.
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "http://bnci-horizon-2020.eu/database/data-sets/009-2014",
    "file_format": "MATLAB (.mat)",
    "description": (
        "10 subjects, 16ch EEG, P300 Speller (overt attention, 6x6 matrix "
        "row/column intensification) target-vs-non-target classification."
    ),
    "task_type": "P300 (Event-Related Potential)",
    "reference": "Aricò P, Aloise F, Schettini F, Salinari S, Mattia D, "
                 "Cincotti F. Influence of P300 latency jitter on event "
                 "related potential-based brain-computer interface "
                 "performance. J Neural Eng. 2014;11(3):035008.",
    "contact": "BNCI Horizon 2020, http://bnci-horizon-2020.eu",
    "notes": (
        "GeoSpell (covert attention, 'G' files) is a second paradigm this "
        "dataset ships but is NOT fetched/used here -- only the P300 "
        "Speller ('S') files, matching MOABB's canonical BNCI2014009. "
        "Real per-flash target/non-target label from each run's `y` array, "
        "rising-edge-detected (see this file's module docstring); epoch "
        "window hardcoded in loader.py to -0.2s/+0.8s (P300 literature "
        "scale), not compile.json's global pre/post_event_seconds (tuned "
        "for ~5s motor-imagery trials)."
    ),
}

CHANNELS = {
    str(i + 1): {"label": label} for i, label in enumerate([
        "Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8",
        "F3", "F4", "FCz", "C3", "C4", "CP3", "CPz", "CP4",
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
    for path in sorted(glob.glob(os.path.join(raw_dir, "A*S.mat"))):
        m = re.match(r"A(\d+)S\.mat$", os.path.basename(path))
        if not m:
            continue
        n = int(m.group(1))
        structure[str(n)] = {"file": f"raw/A{n:02d}S.mat"}
    return structure


def main():
    raw_dir = os.path.join(ROOT, "raw")
    structure = build_data_structure(raw_dir)

    meta = {
        "data_metadata": {
            "dataset_name": "BNCI2014009",
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
