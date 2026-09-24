"""
EmotionVideo (Mendeley 10.17632/58rydc6vwc.2, "An EEG Dataset for Brainwave
Recording During Emotion Elicitation via Video Clips") -- metadata.json
generator. See docs/agents/adding-a-dataset.md Step 3.

Source: data.mendeley.com/datasets/58rydc6vwc/2 (renamed from the raw
"Mendeley" placeholder folder name 2026-09-22). 30 subjects, g.tec Unicorn
Hybrid Black, 8ch (Fz/C3/Cz/C4/Pz/PO7/Oz/PO8, standard 10-20 -- confirmed via
the device's official user manual and independent papers, NOT stated in the
CSVs themselves, which only label columns "EEG 1".."EEG 8"), 250 Hz nominal
(verified 2026-09-22: ~30.6k-30.9k rows for a nominal 120s/2min recording).
12 CSV trials/subject (01.csv-12.csv), each ~2 minutes, columns EEG 1-8 +
accelerometer/gyroscope/battery/counter/validation (dropped, only EEG 1-8
used).

PRETRAIN-ONLY, dummy label (targets.count=1) -- decided 2026-09-22. The
paper's own description says subjects watched 4 emotion-eliciting videos
(fear/sorrow/happiness/neutrality), but the actual release has 12 videos
(MOV1-4 with audio, DEAF1-4 silent-designed, MUTE_MOV1-4 = muted MOV1-4 --
see raw/Videos' "read me.txt", not extracted here to save space) and
NOTHING in the release (not the CSVs, not participant_info.xlsx, not the
Mendeley page) states which of the 12 numbered CSVs corresponds to which
video, or whether presentation order was fixed or randomized per subject.
Real 4-class (or 12-way) emotion labels are NOT recoverable from what's
here -- revisit if the order turns up (e.g. from the authors).
"""
import json
import os
import re

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "https://data.mendeley.com/datasets/58rydc6vwc/2",
    "file_format": "CSV",
    "description": (
        "30 subjects, g.tec Unicorn Hybrid Black (8ch), continuous EEG while "
        "watching emotion-eliciting video clips (fear/sorrow/happiness/"
        "neutrality per the paper's description; the actual release has 12 "
        "video conditions per subject, see notes)."
    ),
    "task_type": "emotion_elicitation_video",
    "reference": "Mendeley Data, DOI 10.17632/58rydc6vwc.2, Sharda University "
                 "(Uttar Pradesh), published 2026-03-02, CC BY 4.0",
    "notes": (
        "PRETRAIN-ONLY: no video/emotion label is recoverable for the 12 "
        "numbered CSVs/subject (01.csv-12.csv) -- see this file's module "
        "docstring for why. targets.count=1 dummy label throughout. "
        "Channel names (Fz/C3/Cz/C4/Pz/PO7/Oz/PO8) are the Unicorn Hybrid "
        "Black's documented default montage, not stated in the CSVs "
        "themselves (columns are just 'EEG 1'..'EEG 8') -- resolved via MNE "
        "standard_1020 at load time, same as other datasets. Sample rate "
        "(250 Hz) is the device's documented nominal rate, cross-checked "
        "against real row counts (~30.6k-30.9k rows per ~120s recording)."
    ),
}

CHANNELS = {
    str(i + 1): {"label": label}
    for i, label in enumerate(["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"])
}

TARGETS = {
    "count": 1,
    "type": "unlabelled",
    "0": {"label": "no recoverable label -- pretrain-only, see dataset_info.notes"},
}

N_TRIALS = 12


def build_data_structure(signals_dir):
    structure = {}
    for dname in sorted(os.listdir(signals_dir)):
        m = re.match(r"S(\d+)$", dname)
        if not m:
            continue
        sub = m.group(1)
        files = [f"raw/EEG_Signals/S{sub}/{i:02d}.csv" for i in range(1, N_TRIALS + 1)
                 if os.path.exists(os.path.join(signals_dir, dname, f"{i:02d}.csv"))]
        if files:
            structure[str(int(sub))] = {"files": files}
    return structure


def main():
    signals_dir = os.path.join(ROOT, "raw", "EEG_Signals")
    structure = build_data_structure(signals_dir)

    meta = {
        "data_metadata": {
            "dataset_name": "EmotionVideo",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 250,
                "num_subjects": len(structure),
            },
            "targets": TARGETS,
            "channels": {"count": len(CHANNELS), "system": "10-20 International System", **CHANNELS},
        },
        "data_structure": structure,
    }

    out_path = os.path.join(ROOT, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    n_files = sum(len(v["files"]) for v in structure.values())
    print(f"wrote {out_path}: {len(structure)} subjects, {n_files} trials")


if __name__ == "__main__":
    main()
