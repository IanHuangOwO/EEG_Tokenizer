import csv
import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(__file__)

# Kaggle "Inria BCI Challenge" (error-related potential / P300 speller feedback).
# Reference: Margaux, P. et al. (2012), "Objective and subjective evaluation of
# online error correction during P300-based spelling", Adv. in HCI.
# Source: https://www.kaggle.com/c/inria-bci-challenge
DATASET_INFO = {
    "source_url": "https://www.kaggle.com/c/inria-bci-challenge",
    "file_format": "CSV",
    "description": (
        "P300-speller feedback recordings used to detect error-related "
        "potentials (ErrP): whether the subject perceived the speller's "
        "feedback (letter selection) as erroneous."
    ),
    "task_type": "error_related_potential",
    "reference": "Margaux, P., Emmanuel, M., Sebastien, D., Olivier, B., & Jeremie, M. (2012). "
                  "Objective and subjective evaluation of online error correction during "
                  "P300-based spelling. Advances in Human-Computer Interaction.",
    "notes": (
        "56 EEG channels (10-20/10-10) + 1 EOG channel, sampled at 200 Hz "
        "(inferred from the Time column's 0.005s step). Each subject has 5 "
        "sessions; trials are cut as a 1.3s window starting at each "
        "FeedBackEvent marker, a commonly used feedback-locked epoch length "
        "for this challenge. Channel 54 is labelled 'P08' in the raw files "
        "(typo for PO8, preserved via original_label)."
    ),
}


def load_channels():
    path = os.path.join(ROOT, "raw", "ChannelsLocation.csv")
    channels = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["Id"])
            label = row["Labels"]
            entry = {
                "label": "PO8" if label == "P08" else label,
                "coordinates": {
                    "polar_angle_deg": float(row["Phi"]),
                    "polar_radius": float(row["Radius"]),
                },
            }
            if label == "P08":
                entry["original_label"] = "P08"
            channels[str(idx)] = entry
    channels[str(len(channels) + 1)] = {"label": "EOG"}
    return channels


def build_data_structure(signals_dir):
    by_subject = defaultdict(list)
    for fname in sorted(os.listdir(signals_dir)):
        m = re.match(r"Data_S(\d+)_Sess\d+\.csv", fname)
        if not m:
            continue
        subject_num = int(m.group(1))
        by_subject[subject_num].append(f"raw/signals/{fname}")
    return {
        str(sub): {"signals": files, "labels": "raw/TrainLabels.csv"}
        for sub, files in sorted(by_subject.items())
    }


def main():
    channels = load_channels()
    signals_dir = os.path.join(ROOT, "raw", "signals")
    structure = build_data_structure(signals_dir)

    meta = {
        "data_metadata": {
            "dataset_name": "Inria_Train",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 200,
                "window_size_seconds": 1.3,
                "num_subjects": len(structure),
                "num_sessions_per_subject": 5,
            },
            "targets": {
                "count": 2,
                "type": "error_related_potential",
                "0": {"label": "Correct feedback"},
                "1": {"label": "Erroneous feedback",
                      "description": "ErrP: subject perceived the speller's feedback as an error"},
            },
            "channels": {
                "count": len(channels),
                "system": "10-20/10-10 International System",
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
