import json
import os
import re

ROOT = os.path.dirname(__file__)

# Hybrid BCI dataset (SSVEP + Motor Imagery), 8 occipital channels, 256 Hz,
# 12-class SSVEP speller. Channel positions are this repo's own custom
# occipital-only placement (not part of the raw dataset), kept here as typed
# constants since there's no raw coordinate file to derive them from.
DATASET_INFO = {
    "description": "Hybrid BCI dataset (SSVEP + Motor Imagery)",
}

CHANNELS = {
    "1": {"label": "PO7", "coordinates": {"polar_angle_deg": -144.1, "polar_radius": 0.511}},
    "2": {"label": "PO3", "coordinates": {"polar_angle_deg": -156.4, "polar_radius": 0.422}},
    "3": {"label": "POz", "coordinates": {"polar_angle_deg": 180.0, "polar_radius": 0.383}},
    "4": {"label": "PO4", "coordinates": {"polar_angle_deg": 156.4, "polar_radius": 0.422}},
    "5": {"label": "PO8", "coordinates": {"polar_angle_deg": 144.1, "polar_radius": 0.511}},
    "6": {"label": "O1", "coordinates": {"polar_angle_deg": -162.0, "polar_radius": 0.515}},
    "7": {"label": "Oz", "coordinates": {"polar_angle_deg": 180.0, "polar_radius": 0.511}},
    "8": {"label": "O2", "coordinates": {"polar_angle_deg": 162.0, "polar_radius": 0.515}},
}

# 12 SSVEP targets on a 4x3 stimulus-frequency grid (0.5 Hz spacing within a
# row, 1.0 Hz spacing between rows).
_FREQS = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]


def build_targets():
    targets = {"count": len(_FREQS), "type": "ssvep"}
    for i, f in enumerate(_FREQS):
        targets[str(i)] = {"label": f"{f} Hz", "stimulus_frequency_hz": f}
    return targets


def build_data_structure(signals_dir, labels_dir):
    entries = []
    for fname in os.listdir(signals_dir):
        m = re.match(r"DataSub_(\d+)\.mat", fname)
        if not m:
            continue
        n = int(m.group(1))
        label_fname = f"LabSub_{n}.mat"
        if not os.path.exists(os.path.join(labels_dir, label_fname)):
            continue
        entries.append((n, fname, label_fname))
    return {
        str(n): {"signals": f"raw/signals/{fname}", "labels": f"raw/labels/{label_fname}"}
        for n, fname, label_fname in sorted(entries)
    }


def main():
    signals_dir = os.path.join(ROOT, "raw", "signals")
    labels_dir = os.path.join(ROOT, "raw", "labels")
    structure = build_data_structure(signals_dir, labels_dir)

    meta = {
        "data_metadata": {
            "dataset_name": "Dial",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 256,
                "window_size_seconds": 4.0,
                "num_subjects": len(structure),
                "num_trials_per_subject": 15,
            },
            "channels": {"count": len(CHANNELS), "system": "Custom Occipital Placement", **CHANNELS},
            "targets": build_targets(),
        },
        "data_structure": structure,
    }

    out_path = os.path.join(ROOT, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"wrote {out_path}: {len(structure)} subjects")


if __name__ == "__main__":
    main()
