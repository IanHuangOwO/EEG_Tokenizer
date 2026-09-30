"""
ADFTD (OpenNeuro ds004504, Miltiadous et al. 2023) -- metadata.json generator. See docs/agents/adding-a-dataset.md Step 3.

Source verified (2026-09-30) from the download's README, participants.tsv, channels.tsv and eeg.json: 88 subjects,
eyes-closed resting EEG, Nihon Kohden EEG 2100, 19 scalp electrodes (10-20) ("referential montage"),
A1/A2 only for impedance, 500 Hz, amplifier high-frequency filter 70 Hz, ~5-21 min per recording. Groups (Group
column): A = Alzheimer's disease (36), F = frontotemporal dementia (23), C = healthy controls (29). The label is a
per-subject diagnosis: every window of a subject carries it (loso is the meaningful split; within-subject few-shot is
not, a subject has one class).

The unprocessed recordings (raw/set/sub-XXX) are used, not derivatives/ (0.5-45 Hz band-passed, A1-A2 re-referenced,
ASR + ICA cleaned): our pipeline applies its own 0.5-100 Hz band-pass to every dataset. Window 10 s, as EEG-FM-Bench
(arXiv 2508.17742) cuts this dataset.
"""
import csv
import json
import os

ROOT = os.path.dirname(__file__)
SET = os.path.join(ROOT, 'raw', 'set')

DATASET_INFO = {
    "source_url": "https://openneuro.org/datasets/ds004504",
    "file_format": "EEGLAB (.set), BIDS",
    "description": ("88 subjects (36 Alzheimer's disease, 23 frontotemporal dementia, 29 healthy), eyes-closed resting "
                    "EEG, 19ch 10-20, 500 Hz, ~5-21 min each; classify the subject's diagnosis."),
    "task_type": "clinical_dementia",
    "reference": ("Miltiadous A, Tzimourta KD, Afrantou T, et al. (2023). A Dataset of Scalp EEG Recordings of "
                  "Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG. Data 8(6):95. "
                  "doi:10.3390/data8060095"),
    "notes": ("Unprocessed recordings (raw/set/sub-XXX), referential montage (all 19 channels carry signal, Cz "
              "included); derivatives/ (their filtered, ASR/ICA-cleaned version) is on disk but unused. Per-subject "
              "label: loso only. 10 s windows as EEG-FM-Bench."),
}

# channel order of every recording (channels.tsv / the .set header)
CHANNELS = {str(i + 1): {"label": l} for i, l in enumerate(
    ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"])}

GROUP = {'A': 0, 'F': 1, 'C': 2}
TARGETS = {
    "count": 3,
    "type": "clinical_dementia",
    "0": {"label": "Alzheimer's disease (group A)"},
    "1": {"label": "frontotemporal dementia (group F)"},
    "2": {"label": "healthy control (group C)"},
}


def main():
    structure = {}
    with open(os.path.join(SET, 'participants.tsv')) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            sid = row['participant_id']                                   # sub-001
            rel = f"raw/set/{sid}/eeg/{sid}_task-eyesclosed_eeg.set"
            if os.path.exists(os.path.join(ROOT, rel)):
                structure[str(int(sid[4:]))] = {"file": rel, "label": GROUP[row['Group']]}
    meta = {
        "data_metadata": {
            "dataset_name": "ADFTD",
            "dataset_info": DATASET_INFO,
            "acquisition": {"sample_frequency": 500, "window_size_seconds": 10.0, "num_subjects": len(structure)},
            "targets": TARGETS,
            "channels": {"count": len(CHANNELS), "system": "10-20 International System", **CHANNELS},
        },
        "data_structure": structure,
    }
    out = os.path.join(ROOT, 'metadata.json')
    json.dump(meta, open(out, 'w'), indent=4)
    print(f"wrote {out}: {len(structure)} subjects, classes " +
          str({k: sum(v['label'] == i for v in structure.values()) for k, i in GROUP.items()}))


if __name__ == '__main__':
    main()
