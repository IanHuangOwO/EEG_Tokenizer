import json
import math
import os
import re

import scipy.io as sio

ROOT = os.path.dirname(__file__)

# BCI Competition IV, Data set 1 (Berlin BCI group -- TU Berlin / Fraunhofer FIRST
# / Charite Neurophysics). Uncued continuous motor-imagery classification task.
# Reference: Blankertz, B., Dornhege, G., Krauledat, M., Muller, K.-R., & Curio, G.
# (2007). The non-invasive Berlin Brain-Computer Interface: Fast acquisition of
# effective performance in untrained subjects. NeuroImage, 37(2), 539-550.
# Source: http://www.bbci.de/competition/iv/desc_1.html
DATASET_INFO = {
    "source_url": "http://www.bbci.de/competition/iv/desc_1.html",
    "file_format": "MATLAB (.mat)",
    "description": (
        "Uncued continuous motor imagery, 2 of {left hand, right hand, foot} "
        "per subject (subject-chosen pair). Calibration (Train) runs have cue "
        "markers (4s cued MI per trial)."
    ),
    "task_type": "motor_imagery",
    "reference": "Blankertz, B., Dornhege, G., Krauledat, M., Muller, K.-R., & Curio, G. (2007). "
                  "The non-invasive Berlin Brain-Computer Interface: Fast acquisition of effective "
                  "performance in untrained subjects. NeuroImage, 37(2), 539-550.",
    "notes": (
        "59 EEG channels, 1000 Hz. Per-subject class pair varies -- see each "
        "data_structure entry's 'classes' field (targets.0/1 below are "
        "generic placeholders since the physical MI class differs per "
        "subject). Channel labels use BBCI's extended 10-10 ring naming "
        "(CFC*, CCP*, PO1/PO2); IO/dataset.py's _LABEL_ALIASES already maps "
        "these to canonical FC*/CP*/PO3/PO4 for cross-dataset channel "
        "matching. Some subjects' recordings are artificially generated "
        "(undisclosed which, per the competition's design) -- see source_url."
    ),
}


def load_channels(calib_path):
    mat = sio.loadmat(calib_path)
    nfo = mat['nfo'][0, 0]
    clab = [str(c[0]) for c in nfo['clab'][0]]
    xpos = nfo['xpos'].flatten()
    ypos = nfo['ypos'].flatten()

    channels = {}
    for i, label in enumerate(clab):
        x, y = float(xpos[i]), float(ypos[i])
        r = math.hypot(x, y)
        angle = math.degrees(math.atan2(x, y))
        channels[str(i + 1)] = {
            "label": label,
            "coordinates": {
                "polar_angle_deg": round(angle, 4),
                "polar_radius": round(r, 4),
            },
        }
    return channels


def build_data_structure(raw_dir, pattern):
    # EEGDataset (IO/dataset.py) requires int(subject_id) -- map the raw
    # dataset's letter subject ids (a..g) to 1..7, keeping the letter as
    # 'orig_id' for traceability back to the original filenames.
    structure = {}
    for fname in sorted(os.listdir(raw_dir)):
        m = re.match(pattern, fname)
        if not m:
            continue
        letter = m.group(1)[-1]
        subject_num = ord(letter) - ord('a') + 1
        mat = sio.loadmat(os.path.join(raw_dir, fname))
        nfo = mat['nfo'][0, 0]
        classes = [str(c[0]) for c in nfo['classes'][0]]
        structure[str(subject_num)] = {"file": f"raw/{fname}", "orig_id": letter, "classes": classes}
    return structure


def main():
    raw_dir = os.path.join(ROOT, "raw")
    channels = load_channels(os.path.join(raw_dir, "BCICIV_calib_ds1a_1000Hz.mat"))
    structure = build_data_structure(raw_dir, r"(BCICIV_calib_ds1\w)_1000Hz\.mat")

    meta = {
        "data_metadata": {
            "dataset_name": "BCICIV1_Train",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 1000,
                "window_size_seconds": 4.0,
                "num_subjects": 7,
            },
            "targets": {
                "count": 2,
                "type": "motor_imagery",
                "0": {"label": "Class one",
                      "description": "y=-1 in raw mrk; physical MI class is subject-specific, see data_structure[subject].classes[0]"},
                "1": {"label": "Class two",
                      "description": "y=+1 in raw mrk; physical MI class is subject-specific, see data_structure[subject].classes[1]"},
            },
            "channels": {
                "count": len(channels),
                "system": "BBCI extended 10-10 (see notes for CFC/CCP/PO1-2 aliasing)",
                **channels,
            },
        },
        "data_structure": structure,
    }

    out_path = os.path.join(ROOT, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"wrote {out_path}: {len(structure)} subjects")


if __name__ == "__main__":
    main()
