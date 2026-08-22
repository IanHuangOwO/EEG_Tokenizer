import json
import os
import re

ROOT = os.path.dirname(__file__)

# BCI Competition IV, Data set 2b (Graz University of Technology).
# Source: https://www.bbci.de/competition/iv/#dataset2b
# Only the 3 *T (training) sessions per subject are included -- the 2 *E
# (evaluation) sessions' true labels were distributed separately as .mat
# files not present in this raw download, so E-session trials cannot be
# labeled and are excluded.

DATASET_INFO = {
    "source_url": "https://www.bbci.de/competition/iv/#dataset2b",
    "file_format": "GDF",
    "description": "BCI Competition IV Dataset 2b - 2-class motor imagery (left hand, right hand), bipolar C3/Cz/C4 montage, 3 training sessions per subject",
    "task_type": "Motor Imagery",
    "reference": "Leeb, R., Brunner, C., Muller-Putz, G., Schlogl, A., Pfurtscheller, G. (2008). BCI Competition 2008 - Graz data set B.",
    "contact": "BCI Competition IV organizers",
    "note": "Only the 3 *T (training) sessions per subject are included. The 2 *E (evaluation) sessions' true labels "
            "were distributed separately as .mat files not present in this raw download, so E-session trials cannot "
            "be labeled and are excluded. Cue-onset event codes (769/770) mark the start of a 4s motor-imagery "
            "window (verified 3.0s after the 768 trial-start marker, matching the published paradigm timing).",
}

ACQUISITION = {
    "sample_frequency": 250,
    "window_size_seconds": 4.0,
    "num_sessions_per_subject": 3,
    "trials_per_subject": 400,
}

TARGETS = {
    "count": 2,
    "type": "motor_imagery",
    "0": {"label": "Left hand", "description": "Event code 769"},
    "1": {"label": "Right hand", "description": "Event code 770"},
}

CHANNELS = {
    "1": {"label": "C3", "original_label": "EEG:C3", "coordinates": {"polar_angle_deg": -100.091205, "polar_radius": 0.663851}},
    "2": {"label": "Cz", "original_label": "EEG:Cz", "coordinates": {"polar_angle_deg": 177.495882, "polar_radius": 0.091758}},
    "3": {"label": "C4", "original_label": "EEG:C4", "coordinates": {"polar_angle_deg": 99.224598, "polar_radius": 0.679973}},
    "4": {"label": "EOG", "original_label": "EOG:ch01"},
    "5": {"label": "EOG", "original_label": "EOG:ch02"},
    "6": {"label": "EOG", "original_label": "EOG:ch03"},
}
CHANNELS_SYSTEM = "10-20 International System (central line only)"


def build_data_structure(gdf_dir):
    by_subject = {}
    for fname in sorted(os.listdir(gdf_dir)):
        m = re.match(r"B(\d\d)(\d\d)T\.gdf$", fname)
        if not m:
            continue
        subject_num = int(m.group(1))
        by_subject.setdefault(subject_num, []).append(fname)
    return {
        str(n): {"folder": "raw/BCICIV_2b_gdf", "runs": sorted(runs)}
        for n, runs in sorted(by_subject.items())
    }


def main():
    gdf_dir = os.path.join(ROOT, "raw", "BCICIV_2b_gdf")
    structure = build_data_structure(gdf_dir)

    meta = {
        "data_metadata": {
            "dataset_name": "BCICIV2b",
            "dataset_info": DATASET_INFO,
            "acquisition": {**ACQUISITION, "num_subjects": len(structure)},
            "targets": TARGETS,
            "channels": {"count": len(CHANNELS), "system": CHANNELS_SYSTEM, **CHANNELS},
        },
        "data_structure": structure,
    }

    out_path = os.path.join(ROOT, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"wrote {out_path}: {len(structure)} subjects")


if __name__ == "__main__":
    main()
