"""
MDD Patients and Healthy Controls EEG Data (Mumtaz et al.), NEMAR nm000114 v1.0.0
(BIDS) -- metadata.json generator. See docs/agents/adding-a-dataset.md Step 3.

Source verified (2026-09-24) against nemar.org/dataset/nm000114, the dataset README
and the extracted files. The README text says 34 participants (19 healthy, 15 MDD),
but the BIDS release actually has 64 subject folders -- sub-HS1..HS30 (healthy) and
sub-MDDS1..MDDS34 (MDD) -- matching the original figshare release; the files are
taken as authoritative. 181 continuous recordings (eyesClosed / eyesOpen / P300
auditory oddball; most subjects have all three, some fewer, sub-HS15 has two eyesOpen
runs), all 256 Hz, ~20.5 h total.

EDF channels are 'EEG <site>-LE' (19 scalp electrodes referenced to the left ear)
plus 'EEG A2-A1' and, in some files, 'EEG 23A-23R' / 'EEG 24A-24R' -- the last ones
are not scalp EEG and are left out. T3/T4/T5/T6 are relabelled T7/T8/P7/P8.

Added for pretraining: each recording is cut into non-overlapping 5 s windows,
labelled with the subject's group (0 = healthy control, 1 = MDD). Subject keys must be
integers and HS/MDDS numbers overlap, so healthy subject n -> 'n', MDD subject n ->
'100+n'.
"""
import glob
import json
import os
import re
from collections import OrderedDict

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "https://nemar.org/dataset/nm000114",
    "file_format": "BIDS (EDF)",
    "description": (
        "64 subjects (30 healthy controls, 34 major depressive disorder), 19-channel "
        "10-20 EEG referenced to the left ear, 256 Hz: eyes-closed rest, eyes-open "
        "rest and an auditory oddball P300 task."
    ),
    "task_type": "clinical_mdd",
    "reference": "Mumtaz W, Xia L, Ali SSA, Yasin MAM, Hussain M, Malik AS (2017). "
                 "EEG-based computer-aided technique to diagnose major depressive disorder "
                 "(MDD). Biomed Signal Process Control 31:108-115. "
                 "doi:10.1016/j.bspc.2016.07.006. BIDS release: doi:10.82901/nemar.nm000114",
    "notes": (
        "Files have 64 subjects (HS1-30, MDDS1-34) although the README text says 34. "
        "5 s non-overlapping windows, label = subject group (0 healthy, 1 MDD); "
        "recording condition (EC/EO/P300) not encoded. Subject key: HS n -> n, "
        "MDDS n -> 100+n."
    ),
}

# (10-10 label, EDF channel name)
_CH = [("Fp1", "Fp1"), ("F3", "F3"), ("C3", "C3"), ("P3", "P3"), ("O1", "O1"),
       ("F7", "F7"), ("T7", "T3"), ("P7", "T5"), ("Fz", "Fz"), ("Fp2", "Fp2"),
       ("F4", "F4"), ("C4", "C4"), ("P4", "P4"), ("O2", "O2"), ("F8", "F8"),
       ("T8", "T4"), ("P8", "T6"), ("Cz", "Cz"), ("Pz", "Pz")]
CHANNELS = OrderedDict((str(i + 1), {"label": lab, "original_label": f"EEG {raw}-LE"})
                       for i, (lab, raw) in enumerate(_CH))

TARGETS = {
    "count": 2,
    "type": "clinical_group",
    "0": {"label": "healthy control"},
    "1": {"label": "major depressive disorder"},
}


def build_data_structure(raw_dir):
    structure = {}
    for sub_dir in glob.glob(os.path.join(raw_dir, "sub-*")):
        m = re.match(r"sub-(HS|MDDS)(\d+)$", os.path.basename(sub_dir))
        if not m:
            continue
        group, n = m.group(1), int(m.group(2))
        key = str(n if group == "HS" else 100 + n)
        files = sorted(glob.glob(os.path.join(sub_dir, "**", "*_eeg.edf"), recursive=True))
        structure[key] = {"label": 0 if group == "HS" else 1,
                          "files": ["raw/" + os.path.relpath(f, raw_dir) for f in files]}
    return OrderedDict(sorted(structure.items(), key=lambda kv: int(kv[0])))


def main():
    structure = build_data_structure(os.path.join(ROOT, "raw"))
    meta = {
        "data_metadata": {
            "dataset_name": "MDD_Mumtaz",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 256,
                "window_size_seconds": 5.0,
                "num_subjects": len(structure),
            },
            "targets": TARGETS,
            "channels": {"count": len(CHANNELS), "system": "10-20, linked-ear/left-ear reference", **CHANNELS},
        },
        "data_structure": structure,
    }
    out_path = os.path.join(ROOT, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    n_hc = sum(v["label"] == 0 for v in structure.values())
    print(f"wrote {out_path}: {len(structure)} subjects ({n_hc} healthy, {len(structure) - n_hc} MDD), "
          f"{sum(len(v['files']) for v in structure.values())} recordings")


if __name__ == "__main__":
    main()
