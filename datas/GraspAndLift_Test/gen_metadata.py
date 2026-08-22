import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(__file__)

# Kaggle "Grasp-and-Lift EEG Detection" (WAY-EEG-GAL recordings, Luciw et al.
# 2014, "Multi-channel EEG recordings during 3,936 grasp and lift trials with
# varying weight and friction", Scientific Data).
# Source: https://www.kaggle.com/competitions/grasp-and-lift-eeg-detection/data
DATASET_INFO = {
    "source_url": "https://www.kaggle.com/competitions/grasp-and-lift-eeg-detection/data",
    "file_format": "CSV",
    "description": (
        "Continuous 32-channel EEG during a grasp-and-lift task (varying "
        "object weight/friction). This is the competition's held-out test "
        "split (series 9-10 per subject) -- test.zip ships *_data.csv only, "
        "no *_events.csv, since predicting those events is the competition "
        "task. No ground-truth labels were ever released for this split."
    ),
    "task_type": "grasp_and_lift_event_detection",
    "reference": "Luciw, M. D., Jarocka, E., & Edin, B. B. (2014). Multi-channel EEG recordings during "
                 "3,936 grasp and lift trials with varying weight and friction. Scientific Data, 1, 140047.",
    "notes": (
        "12 subjects, 2 series each (9-10). 500 Hz. Same 32-channel montage "
        "as GraspAndLift_Train. loader.py chops each continuous series into "
        "fixed non-overlapping windows with a dummy label 0 (there are no "
        "real per-window labels to read here, unlike GraspAndLift_Train "
        "which discards its real event columns for the same reason). Fine "
        "for self-supervised pretraining only."
    ),
}

# Same 32-channel montage as GraspAndLift_Train (same raw CSV header order).
CHANNELS = {
    "1": {"label": "Fp1", "coordinates": {"polar_angle_deg": -19.330008, "polar_radius": 0.889303}},
    "2": {"label": "Fp2", "coordinates": {"polar_angle_deg": 19.385429, "polar_radius": 0.899982}},
    "3": {"label": "F7", "coordinates": {"polar_angle_deg": -58.846813, "polar_radius": 0.821032}},
    "4": {"label": "F3", "coordinates": {"polar_angle_deg": -43.410838, "polar_radius": 0.731111}},
    "5": {"label": "Fz", "coordinates": {"polar_angle_deg": 0.305708, "polar_radius": 0.585128}},
    "6": {"label": "F4", "coordinates": {"polar_angle_deg": 43.667670, "polar_radius": 0.750733}},
    "7": {"label": "F8", "coordinates": {"polar_angle_deg": 58.693816, "polar_radius": 0.854902}},
    "8": {"label": "FC5", "coordinates": {"polar_angle_deg": -76.425905, "polar_radius": 0.794337}},
    "9": {"label": "FC1", "coordinates": {"polar_angle_deg": -52.633124, "polar_radius": 0.428578}},
    "10": {"label": "FC2", "coordinates": {"polar_angle_deg": 52.763095, "polar_radius": 0.436909}},
    "11": {"label": "FC6", "coordinates": {"polar_angle_deg": 75.928386, "polar_radius": 0.819945}},
    "12": {"label": "T7", "coordinates": {"polar_angle_deg": -100.776424, "polar_radius": 0.856720}},
    "13": {"label": "C3", "coordinates": {"polar_angle_deg": -100.091205, "polar_radius": 0.663851}},
    "14": {"label": "Cz", "coordinates": {"polar_angle_deg": 177.495882, "polar_radius": 0.091758}},
    "15": {"label": "C4", "coordinates": {"polar_angle_deg": 99.224598, "polar_radius": 0.679973}},
    "16": {"label": "T8", "coordinates": {"polar_angle_deg": 100.012029, "polar_radius": 0.863956}},
    "17": {"label": "TP9"},
    "18": {"label": "CP5", "coordinates": {"polar_angle_deg": -120.321875, "polar_radius": 0.922057}},
    "19": {"label": "CP1", "coordinates": {"polar_angle_deg": -143.095865, "polar_radius": 0.591414}},
    "20": {"label": "CP2", "coordinates": {"polar_angle_deg": 140.805909, "polar_radius": 0.607387}},
    "21": {"label": "CP6", "coordinates": {"polar_angle_deg": 118.955412, "polar_radius": 0.952253}},
    "22": {"label": "TP10"},
    "23": {"label": "P7", "coordinates": {"polar_angle_deg": -135.399961, "polar_radius": 1.031602}},
    "24": {"label": "P3", "coordinates": {"polar_angle_deg": -146.067901, "polar_radius": 0.949594}},
    "25": {"label": "Pz"},
    "26": {"label": "P4", "coordinates": {"polar_angle_deg": 144.679127, "polar_radius": 0.962834}},
    "27": {"label": "P8", "coordinates": {"polar_angle_deg": 135.004941, "polar_radius": 1.033253}},
    "28": {"label": "PO9"},
    "29": {"label": "O1", "coordinates": {"polar_angle_deg": -165.341503, "polar_radius": 1.162322}},
    "30": {"label": "Oz"},
    "31": {"label": "O2", "coordinates": {"polar_angle_deg": 165.099907, "polar_radius": 1.160584}},
    "32": {"label": "PO10"},
}

TARGETS = {
    "count": 1,
    "type": "unlabeled",
    "0": {"label": "unlabeled (pretrain-only placeholder)",
          "description": "loader.py assigns this to every window -- no ground-truth labels were "
                          "ever released for this split, see dataset_info.notes."},
}


def build_data_structure(test_dir):
    by_subject = defaultdict(list)
    for fname in sorted(os.listdir(test_dir)):
        m = re.match(r"subj(\d+)_series(\d+)_data\.csv", fname)
        if not m:
            continue
        subject_num = int(m.group(1))
        by_subject[subject_num].append(f"raw/test/{fname}")

    return {
        str(subject_num): {"files": [{"data": f} for f in sorted(files)]}
        for subject_num, files in sorted(by_subject.items())
    }


def main():
    test_dir = os.path.join(ROOT, "raw", "test")
    structure = build_data_structure(test_dir)

    meta = {
        "data_metadata": {
            "dataset_name": "GraspAndLift_Test",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 500,
                "window_size_seconds": 4.0,
                "num_subjects": len(structure),
                "num_series_per_subject": 2,
            },
            "targets": TARGETS,
            "channels": {"count": len(CHANNELS), "system": "10-20/10-10 International System", **CHANNELS},
        },
        "data_structure": structure,
    }

    out_path = os.path.join(ROOT, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"wrote {out_path}: {len(structure)} subjects")


if __name__ == "__main__":
    main()
