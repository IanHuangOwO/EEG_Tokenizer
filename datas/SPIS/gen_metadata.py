"""
SPIS Resting-State EEG Dataset -- metadata.json generator.
See docs/agents/adding-a-dataset.md Step 3.

Source: github.com/mastaneh/SPIS-Resting-State-Dataset (zip placed by hand in
datas/SPIS/, extracted to raw/). 10 subjects (S02-S11, no S01), 64-channel
Biosemi Active 2, 10-10 layout, EC (eyes-closed) and EO (eyes-open) resting
state, 2.5 minutes each, recorded before a 105-minute sustained-attention
task. README claims .mat v7.3/HDF5, but the actual files are plain MATLAB v5
(verified 2026-09-22 via `file` + scipy.io.loadmat) -- one array `dataRest`
per file, shape (68, T): channels 1-64 EEG (10-10 names, identical list to
SRM_RestingState -- all resolve via MNE's standard_1020 montage, coordinates
omitted same precedent), 65-67 EOG, 68 trigger (200=EO, 220=EC, unused here
since EC/EO is already given by filename/directory, not read from the
trigger channel). Raw sample rate 256 Hz per README (T ~= 38400 = 150s).

Unlike SRM_RestingState (no real task), EC vs EO here IS a real 2-class
label. Each file is one long continuous block per condition -- chopped into
non-overlapping window_size_seconds windows at load time (same pattern as
GraspAndLift_Train's loader), each window inheriting its file's EC/EO label.
"""
import json
import os
import re

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "https://github.com/mastaneh/SPIS-Resting-State-Dataset",
    "file_format": "MATLAB v5 .mat (README claims v7.3, verified otherwise)",
    "description": (
        "10 subjects, 2.5 minutes eyes-closed (EC) and 2.5 minutes eyes-open "
        "(EO) resting-state EEG each, recorded before a 105-minute Sustained "
        "Attention to Response Task. Biosemi Active 2, 64 monopolar channels, "
        "10-10 layout, recorded at 2048 Hz then downsampled to 256 Hz in the "
        "provided files."
    ),
    "task_type": "eyes_open_vs_closed",
    "reference": "M. Torkamani-Azar, S. D. Kanik, S. Aydin and M. Cetin, "
                  "'Prediction of Reaction Time and Vigilance Variability From "
                  "Spatio-Spectral Features of Resting-State EEG in a Long "
                  "Sustained Attention Task,' IEEE JBHI, vol. 24, no. 9, 2020, "
                  "doi: 10.1109/JBHI.2020.2980056",
    "contact": "Mastaneh Torkamani-Azar (mastaneh.torkamani@uef.fi), "
               "Mujdat Cetin (mujdat.cetin@rochester.edu)",
    "notes": (
        "No subject S01 -- numbering starts at S02. Each subject contributes "
        "exactly one EC file and one EO file, each ~150s continuous (no "
        "trial structure) -- chopped into window_size_seconds windows in "
        "loader.py, label taken from which file the window came from (not "
        "from channel 68's trigger values, which are redundant with this). "
        "Channels 65-67 (EOG) and 68 (trigger) dropped, only the 64 EEG "
        "channels are used."
    ),
}

# Same 64-channel 10-10 list as SRM_RestingState/README.md's channelList.
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
    "count": 2,
    "type": "eyes_open_vs_closed",
    "0": {"label": "eyes-closed (EC)"},
    "1": {"label": "eyes-open (EO)"},
}

FNAME = "S{sub}_restingPre_{cond}.mat"


def build_data_structure(raw_dir):
    """One entry per subject, EC/EO file pair -- shape C variant (dict of
    named files instead of a runs list), see Step 4."""
    structure = {}
    for dname in sorted(os.listdir(raw_dir)):
        m = re.match(r"S(\d+)_restingPre_EC\.mat$", dname)
        if not m:
            continue
        sub = m.group(1)
        ec_rel = f"raw/Pre-SART EEG/{FNAME.format(sub=sub, cond='EC')}"
        eo_rel = f"raw/Pre-SART EEG/{FNAME.format(sub=sub, cond='EO')}"
        if os.path.exists(os.path.join(ROOT, eo_rel)):
            structure[str(int(sub))] = {"EC": ec_rel, "EO": eo_rel}
    return structure


def main():
    raw_dir = os.path.join(ROOT, "raw", "Pre-SART EEG")
    structure = build_data_structure(raw_dir)

    meta = {
        "data_metadata": {
            "dataset_name": "SPIS",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 256,
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
    print(f"wrote {out_path}: {len(structure)} subjects")


if __name__ == "__main__":
    main()
