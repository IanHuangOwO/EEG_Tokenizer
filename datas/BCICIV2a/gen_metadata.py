import json
import os
import re

ROOT = os.path.dirname(__file__)

# BCI Competition IV, Data set 2a (Graz University of Technology).
# Source: https://www.bbci.de/competition/iv/#dataset2a
# Only the *T (training) session is included -- the *E (evaluation) session's
# true labels were distributed separately by the competition organizers as
# .mat files not present in this raw GDF download, so E-session trials cannot
# be labeled and are excluded (raw *E.gdf files still sit alongside *T.gdf in
# BCICIV_2a_gdf/, just never referenced by data_structure).

DATASET_INFO = {
    "source_url": "https://www.bbci.de/competition/iv/#dataset2a",
    "file_format": "GDF",
    "description": "BCI Competition IV Dataset 2a - 4-class motor imagery (left hand, right hand, feet, tongue)",
    "task_type": "Motor Imagery",
    "reference": "Brunner, C., Leeb, R., Muller-Putz, G., Schlogl, A., Pfurtscheller, G. (2008). BCI Competition 2008 - Graz data set A.",
    "contact": "BCI Competition IV organizers",
    "note": "Only the *T (training) session per subject is included. The *E (evaluation) session's true labels were "
            "distributed separately by the competition organizers as .mat files not present in this raw download, "
            "so E-session trials cannot be labeled and are excluded. Cue-onset event codes (769/770/771/772) mark "
            "the start of a 4s motor-imagery window (verified 2.0s after the 768 trial-start marker, matching the "
            "published paradigm timing).",
}

ACQUISITION = {
    "sample_frequency": 250,
    "window_size_seconds": 4.0,
    "num_sessions_per_subject": 1,
    "trials_per_subject": 288,
}

TARGETS = {
    "count": 4,
    "type": "motor_imagery",
    "0": {"label": "Left hand", "description": "Event code 769"},
    "1": {"label": "Right hand", "description": "Event code 770"},
    "2": {"label": "Feet", "description": "Event code 771"},
    "3": {"label": "Tongue", "description": "Event code 772"},
}

CHANNELS = {
    "1": {"label": "Fz", "original_label": "EEG-Fz", "coordinates": {"polar_angle_deg": 0.305708, "polar_radius": 0.585128}},
    "2": {"label": "FC3", "coordinates": {"polar_angle_deg": -69.320534, "polar_radius": 0.643264}},
    "3": {"label": "FC1", "coordinates": {"polar_angle_deg": -52.633124, "polar_radius": 0.428578}},
    "4": {"label": "FCz", "coordinates": {"polar_angle_deg": 0.786695, "polar_radius": 0.273926}},
    "5": {"label": "FC2", "coordinates": {"polar_angle_deg": 52.763095, "polar_radius": 0.436909}},
    "6": {"label": "FC4", "coordinates": {"polar_angle_deg": 69.151891, "polar_radius": 0.666573}},
    "7": {"label": "C5", "coordinates": {"polar_angle_deg": -99.725774, "polar_radius": 0.814507}},
    "8": {"label": "C3", "original_label": "EEG-C3", "coordinates": {"polar_angle_deg": -100.091205, "polar_radius": 0.663851}},
    "9": {"label": "C1", "coordinates": {"polar_angle_deg": -105.435825, "polar_radius": 0.375111}},
    "10": {"label": "Cz", "original_label": "EEG-Cz", "coordinates": {"polar_angle_deg": 177.495882, "polar_radius": 0.091758}},
    "11": {"label": "C2", "coordinates": {"polar_angle_deg": 104.330883, "polar_radius": 0.388819}},
    "12": {"label": "C4", "original_label": "EEG-C4", "coordinates": {"polar_angle_deg": 99.224598, "polar_radius": 0.679973}},
    "13": {"label": "C6", "coordinates": {"polar_angle_deg": 98.703859, "polar_radius": 0.844282}},
    "14": {"label": "CP3", "coordinates": {"polar_angle_deg": -126.488165, "polar_radius": 0.79052}},
    "15": {"label": "CP1", "coordinates": {"polar_angle_deg": -143.095865, "polar_radius": 0.591414}},
    "16": {"label": "CPz", "coordinates": {"polar_angle_deg": 179.532858, "polar_radius": 0.473196}},
    "17": {"label": "CP2", "coordinates": {"polar_angle_deg": 140.805909, "polar_radius": 0.607387}},
    "18": {"label": "CP4", "coordinates": {"polar_angle_deg": 124.997181, "polar_radius": 0.813152}},
    "19": {"label": "P1", "coordinates": {"polar_angle_deg": -160.433681, "polar_radius": 0.854598}},
    "20": {"label": "Pz", "original_label": "EEG-Pz", "coordinates": {"polar_angle_deg": 179.770649, "polar_radius": 0.811156}},
    "21": {"label": "P2", "coordinates": {"polar_angle_deg": 158.367636, "polar_radius": 0.865855}},
    "22": {"label": "POz", "coordinates": {"polar_angle_deg": 179.879104, "polar_radius": 1.021782}},
    "23": {"label": "EOG", "original_label": "EOG-left"},
    "24": {"label": "EOG", "original_label": "EOG-central"},
    "25": {"label": "EOG", "original_label": "EOG-right"},
}
CHANNELS_SYSTEM = "10-10 International System"


def build_data_structure(gdf_dir):
    structure = {}
    for fname in sorted(os.listdir(gdf_dir)):
        m = re.match(r"A(\d\d)T\.gdf$", fname)
        if not m:
            continue
        structure[str(int(m.group(1)))] = {"file": f"raw/BCICIV_2a_gdf/{fname}"}
    return structure


def main():
    gdf_dir = os.path.join(ROOT, "raw", "BCICIV_2a_gdf")
    structure = build_data_structure(gdf_dir)

    meta = {
        "data_metadata": {
            "dataset_name": "BCICIV2a",
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
