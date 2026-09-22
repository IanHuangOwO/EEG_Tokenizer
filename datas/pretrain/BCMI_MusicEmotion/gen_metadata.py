"""
BCMI_MusicEmotion (OpenNeuro ds002721, "An EEG dataset recorded during
affective music listening") -- metadata.json generator. See
docs/agents/adding-a-dataset.md Step 3.

Source verified directly (2026-09-22) via the dataset's own BIDS files
fetched from the public S3 bucket (s3://openneuro.org/ds002721/, unsigned/
no-auth): dataset_description.json, README, one subject's *_eeg.json/
*_channels.tsv/*_events.json/*_events.tsv, participants.tsv.

31 subjects (sub-01..sub-31), 19ch (10-20 system, standard names, confirmed
identical across subjects via md5), 1000 Hz, EDF, CC0. 6 runs/subject: run1
and run6 are 300s eyes-open(?) resting state (per README: "sit still and
rest"), run2-5 each contain 10 x 20s music-listening trials (40 clips total
across the 4 runs) drawn from the Eerola & Vuoskoski film-score corpus, each
followed by 8 post-clip Likert (1-9) ratings (pleasant/energetic/tense/
angry/afraid/happy/sad/tender) delivered as sequential trigger events
(Question NN -> Response NN -> Answer NN codes, see one subject's real
events.json, not reproduced in full here). sub-06 is missing run6 (only
5/6 runs present in the source bucket, not a download error -- verified via
the bucket's own object listing).

PRETRAIN-ONLY, dummy label -- decided with user 2026-09-22. The 8-dim
Likert ratings are a real, usable label but need careful sequential
trigger-event parsing (Question/Response/Answer, see the source events.json
loaded above) to extract correctly; deferred rather than guessed. Each run
is treated as one continuous block and chunked into window_size_seconds
windows at load time (same pattern as SPIS/GraspAndLift_Train's loaders),
discarding all trigger/event timing for now.
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "https://openneuro.org/datasets/ds002721/versions/1.0.1",
    "file_format": "EDF (BIDS)",
    "description": (
        "31 subjects, 19ch EEG (10-20 system), continuous recording across "
        "6 runs: run1/run6 are 300s resting state, run2-5 each contain 10 "
        "x 20s music-listening trials (film-score clips) each followed by "
        "8 self-reported emotion ratings."
    ),
    "task_type": "music_emotion_listening",
    "reference": "Daly, I., Nicolaou, N., Williams, D., Hwang, F., Kirke, A., "
                 "Miranda, E., Nasuto, S.J. (2018). Neural and physiological "
                 "data from participants listening to affective music. "
                 "Scientific Data. DOI: 10.18112/openneuro.ds002721.v1.0.1",
    "notes": (
        "PRETRAIN-ONLY: targets.count=1 dummy label throughout -- the real "
        "8-dim Likert rating label (pleasant/energetic/tense/angry/afraid/"
        "happy/sad/tender, 1-9 each) is present in each run's events.tsv/ "
        "events.json but requires sequential Question/Response/Answer "
        "trigger parsing not yet implemented, see this file's module "
        "docstring. sub-06 has only 5/6 runs (run6 absent from the source "
        "bucket itself, verified via S3 object listing, not a download "
        "gap). Runs are NOT split by event -- each run is one continuous "
        "block, chunked into window_size_seconds windows in loader.py."
    ),
}

# BIDS channels.tsv order, identical across all 31 subjects (verified via md5, 2026-09-22).
CHANNELS = {
    str(i + 1): {"label": label} for i, label in enumerate([
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
        "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
    ])
}

TARGETS = {
    "count": 1,
    "type": "unlabelled",
    "0": {"label": "no label extracted yet -- pretrain-only, see dataset_info.notes"},
}

N_RUNS = 6


def build_data_structure(raw_dir):
    """One entry per subject with at least one run present -- shape C (see
    Step 4), runs discovered by globbing rather than assumed all 6 exist
    (sub-06 is short one run)."""
    structure = {}
    for dname in sorted(os.listdir(raw_dir)):
        m = re.match(r"sub-(\d+)$", dname)
        if not m:
            continue
        sub = m.group(1)
        files = []
        for run in range(1, N_RUNS + 1):
            rel = f"raw/sub-{sub}/eeg/sub-{sub}_task-run{run}_eeg.edf"
            if os.path.exists(os.path.join(ROOT, rel)):
                files.append(rel)
        if files:
            structure[str(int(sub))] = {"files": files}
    return structure


def main():
    raw_dir = os.path.join(ROOT, "raw")
    structure = build_data_structure(raw_dir)

    meta = {
        "data_metadata": {
            "dataset_name": "BCMI_MusicEmotion",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 1000,
                "window_size_seconds": 5.0,
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
    n_runs = sum(len(v["files"]) for v in structure.values())
    print(f"wrote {out_path}: {len(structure)} subjects, {n_runs} runs")


if __name__ == "__main__":
    main()
