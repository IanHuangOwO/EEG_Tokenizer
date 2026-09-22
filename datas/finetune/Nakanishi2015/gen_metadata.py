import json
import os
import re

ROOT = os.path.dirname(__file__)

# 12-class Joint Frequency-Phase Modulated (JFPM) SSVEP dataset -- 8 occipital
# channels, 256 Hz, 12-target speller (4x3 grid, frequency+phase coded).
# Renamed from "Dial" 2026-09-22 (that name carried no information and the
# old dataset_info was wrong -- "Hybrid BCI dataset (SSVEP + Motor Imagery)"
# was stale/incorrect, this is SSVEP-only, no motor imagery).
#
# Two sources, cross-checked 2026-09-22:
# - Actual data downloaded from: github.com/mnakanishi/12JFPM_SSVEP (10
#   subjects, DataSub_N.mat/LabSub_N.mat pairs).
# - Fuller metadata (channel montage, task/timing detail, citation) from the
#   NEMAR/OpenNeuro BIDS curation of the same recordings:
#   github.com/nemarDatasets/nm000118 (DOI 10.82901/nemar.nm000118) -- that
#   BIDS release only ships 9 of the 10 subjects (reason not stated); this
#   repo keeps all 10 since the raw github source has all 10 and they load
#   fine.
#
# Channel names (PO7/PO3/POz/PO4/PO8/O1/Oz/O2) are standard 10-20 occipital
# sites per NEMAR's "montage: standard_1020" -- coordinates omitted, resolved
# via MNE's standard_1020 montage at load time (IO/loader.py's
# get_standard_coords, takes precedence over metadata coordinates anyway;
# the old hand-typed polar coordinates here were dead weight, same
# resolution path, just harder to read).
DATASET_INFO = {
    "source_url": "https://github.com/mnakanishi/12JFPM_SSVEP",
    "file_format": "MATLAB (.mat)",
    "description": "12-class Joint Frequency-Phase Modulated (JFPM) SSVEP dataset",
    "task_type": "SSVEP (Steady-State Visual Evoked Potential)",
    "reference": "Nakanishi M, Wang Y, Wang Y-T, Jung T-P (2015). A Comparison "
                 "Study of Canonical Correlation Analysis Based Methods for "
                 "Detecting Steady-State Visual Evoked Potentials. PLoS ONE "
                 "10(10):e0140703. doi:10.1371/journal.pone.0140703",
    "contact": "wangyj@semi.ac.cn (per NEMAR nm000118)",
    "notes": (
        "Biosemi ActiveTwo, CMS/DRL reference, originally recorded at 2048 Hz "
        "then downsampled to 256 Hz (per NEMAR nm000118). Task: 4x3 visual "
        "grid, 1s cue then 4s of flickering stimuli at 12 frequencies "
        "(9.25-14.75 Hz, 0.5 Hz spacing) with joint frequency+phase coding, "
        "15 blocks x 12 trials/subject (180 trials/subject). Fuller metadata "
        "at github.com/nemarDatasets/nm000118 -- that BIDS curation lists "
        "only 9/10 subjects; this repo uses all 10 from the raw source."
    ),
}

CHANNELS = {
    str(i + 1): {"label": label}
    for i, label in enumerate(["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"])
}

# 12 SSVEP targets on a 4x3 stimulus-frequency grid (0.5 Hz spacing within a
# row, 1.0 Hz spacing between rows).
_FREQS = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]


def build_targets():
    targets = {"count": len(_FREQS), "type": "ssvep"}
    for i, f in enumerate(_FREQS):
        targets[str(i)] = {"label": f"{f} Hz", "stimulus_frequency_hz": f}
    return targets


def build_data_structure(signals_dir, labels_dir):
    entries = []
    for fname in os.listdir(signals_dir):
        m = re.match(r"DataSub_(\d+)\.mat", fname)
        if not m:
            continue
        n = int(m.group(1))
        label_fname = f"LabSub_{n}.mat"
        if not os.path.exists(os.path.join(labels_dir, label_fname)):
            continue
        entries.append((n, fname, label_fname))
    return {
        str(n): {"signals": f"raw/signals/{fname}", "labels": f"raw/labels/{label_fname}"}
        for n, fname, label_fname in sorted(entries)
    }


def main():
    signals_dir = os.path.join(ROOT, "raw", "signals")
    labels_dir = os.path.join(ROOT, "raw", "labels")
    structure = build_data_structure(signals_dir, labels_dir)

    meta = {
        "data_metadata": {
            "dataset_name": "Nakanishi2015",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 256,
                "window_size_seconds": 4.0,
                "num_subjects": len(structure),
                "num_trials_per_subject": 15,
            },
            "channels": {"count": len(CHANNELS), "system": "10-20 International System (occipital sites)", **CHANNELS},
            "targets": build_targets(),
        },
        "data_structure": structure,
    }

    out_path = os.path.join(ROOT, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"wrote {out_path}: {len(structure)} subjects")


if __name__ == "__main__":
    main()
