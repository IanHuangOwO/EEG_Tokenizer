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
        "object weight/friction), annotated with 6 binary event columns "
        "(HandStart, FirstDigitTouch, BothStartLoadPhase, LiftOff, Replace, "
        "BothReleased) that overlap in time -- a multi-label sequence task, "
        "not one discrete class per trial."
    ),
    "task_type": "grasp_and_lift_event_detection",
    "reference": "Luciw, M. D., Jarocka, E., & Edin, B. B. (2014). Multi-channel EEG recordings during "
                 "3,936 grasp and lift trials with varying weight and friction. Scientific Data, 1, 140047.",
    "notes": (
        "12 subjects, 8 training series each (series 9-10 are the Kaggle "
        "competition's held-out test set, not included in train.zip -- "
        "GraspAndLift_Val is currently empty, nothing downloaded for it "
        "yet). 500 Hz. GraspAndLiftLoader (IO/loader.py) does NOT implement "
        "the real 6-column multi-label task -- it chops each continuous "
        "series into fixed non-overlapping windows with a dummy label 0, "
        "discarding the real event columns. Fine for self-supervised "
        "pretraining; do not use for supervised event-detection "
        "finetune/eval without writing a real multi-label loader first."
    ),
}

# 32-channel montage, standard 10-20/10-10 names as given in the raw CSV
# header. Coordinates reused from the shared 10-10 polar table (see
# _gen_eegmmidb_metadata.py) for channels that overlap with it; TP9/TP10/
# PO9/PO10 (and midline Pz/Oz) are left to MNE's standard_1020 montage
# fallback, same precedent as EEGMMIdb's omitted midline channels.
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
          "description": "GraspAndLiftLoader assigns this to every window; real per-sample event "
                          "columns (HandStart/FirstDigitTouch/BothStartLoadPhase/LiftOff/Replace/"
                          "BothReleased) are read from *_events.csv but discarded -- see dataset_info.notes."},
}


def build_data_structure(train_dir):
    by_subject = defaultdict(dict)
    for fname in sorted(os.listdir(train_dir)):
        m = re.match(r"subj(\d+)_series(\d+)_(data|events)\.csv", fname)
        if not m:
            continue
        subject_num, series_num, kind = int(m.group(1)), int(m.group(2)), m.group(3)
        by_subject[subject_num].setdefault(series_num, {})[kind] = f"raw/train/{fname}"

    structure = {}
    for subject_num, series_map in sorted(by_subject.items()):
        files = []
        for series_num, pair in sorted(series_map.items()):
            if "data" in pair and "events" in pair:
                files.append({"data": pair["data"], "events": pair["events"]})
        structure[str(subject_num)] = {"files": files}
    return structure


def main():
    train_dir = os.path.join(ROOT, "raw", "train")
    structure = build_data_structure(train_dir)

    meta = {
        "data_metadata": {
            "dataset_name": "GraspAndLift_Train",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 500,
                "window_size_seconds": 4.0,
                "num_subjects": len(structure),
                "num_series_per_subject": 8,
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
