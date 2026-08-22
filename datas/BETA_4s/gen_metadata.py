import json
import os
import re

ROOT = os.path.dirname(__file__)  # datas/BETA_4s — also writes sibling datas/BETA_3s
DATAS_ROOT = os.path.dirname(ROOT)

# BETA (Benchmark dataset for SSVEP-based BCI) -- Tsinghua University.
# Reference: Liu, B. et al. (2020), "BETA: A Large Benchmark Database Toward
# SSVEP-BCI Application". Frontiers in Neuroscience.
# Source: https://bci.med.tsinghua.edu.cn/download.html
# This repo ships two fixed-window subsets cut from the same 64-channel
# 250 Hz recordings: BETA_4s (4.0s window, 55 subjects, S16-S70) and
# BETA_3s (3.0s window, 15 subjects, S01-S15).
DATASET_INFO_TEMPLATE = {
    "source_url": "https://bci.med.tsinghua.edu.cn/download.html",
    "file_format": "MATLAB (.mat)",
    "description": "Beijing EEG Signal Data for Chinese Brain Computer Interface Competition (BETA) - {w:.0f} second trials subset",
    "window_size": "{w:.1f}s",
    "task_type": "SSVEP (Steady-State Visual Evoked Potential)",
    "note": "This is a {w:.0f}-second window variant of the full BETA dataset (original has 70 subjects total)",
    "contact": "wuhaolin@tsinghua.edu.cn",
    "reference": "Tsinghua University, School of Software. BCI Competition IV Dataset 2B",
}

# 40 SSVEP stimulus frequencies, 8.6-15.8 Hz in the BETA target layout
# (targets 0-35 in ascending order, then the four lowest frequencies
# 8.0-8.4 Hz appended last -- matches the panel numbering in the BETA paper).
_FREQS = [round(8.6 + 0.2 * i, 2) for i in range(37)] + [8.0, 8.2, 8.4]

# 64-channel 10-20 montage with BETA's own polar coordinates (paper Fig. 1 /
# accompanying channel-location file), covering the full scalp (not just
# occipital) since BETA records all 64 standard 10-20 sites.
CHANNELS = {
    "1": {"label": "FP1", "coordinates": {"polar_angle_deg": -17.926, "polar_radius": 0.51499}},
    "2": {"label": "FPZ", "coordinates": {"polar_angle_deg": 0.0, "polar_radius": 0.50669}},
    "3": {"label": "FP2", "coordinates": {"polar_angle_deg": 17.926, "polar_radius": 0.51499}},
    "4": {"label": "AF3", "coordinates": {"polar_angle_deg": -22.461, "polar_radius": 0.42113}},
    "5": {"label": "AF4", "coordinates": {"polar_angle_deg": 22.461, "polar_radius": 0.42113}},
    "6": {"label": "F7", "coordinates": {"polar_angle_deg": -53.913, "polar_radius": 0.52808}},
    "7": {"label": "F5", "coordinates": {"polar_angle_deg": -49.405, "polar_radius": 0.43159}},
    "8": {"label": "F3", "coordinates": {"polar_angle_deg": -39.947, "polar_radius": 0.34459}},
    "9": {"label": "F1", "coordinates": {"polar_angle_deg": -23.493, "polar_radius": 0.27903}},
    "10": {"label": "FZ", "coordinates": {"polar_angle_deg": 0.0, "polar_radius": 0.25338}},
    "11": {"label": "F2", "coordinates": {"polar_angle_deg": 23.493, "polar_radius": 0.27878}},
    "12": {"label": "F4", "coordinates": {"polar_angle_deg": 39.897, "polar_radius": 0.3445}},
    "13": {"label": "F6", "coordinates": {"polar_angle_deg": 49.405, "polar_radius": 0.43128}},
    "14": {"label": "F8", "coordinates": {"polar_angle_deg": 53.867, "polar_radius": 0.52807}},
    "15": {"label": "FT7", "coordinates": {"polar_angle_deg": -71.948, "polar_radius": 0.53192}},
    "16": {"label": "FC5", "coordinates": {"polar_angle_deg": -69.332, "polar_radius": 0.40823}},
    "17": {"label": "FC3", "coordinates": {"polar_angle_deg": -62.425, "polar_radius": 0.28822}},
    "18": {"label": "FC1", "coordinates": {"polar_angle_deg": -44.925, "polar_radius": 0.18118}},
    "19": {"label": "FCz", "coordinates": {"polar_angle_deg": 0.0, "polar_radius": 0.12662}},
    "20": {"label": "FC2", "coordinates": {"polar_angle_deg": 44.925, "polar_radius": 0.18118}},
    "21": {"label": "FC4", "coordinates": {"polar_angle_deg": 62.425, "polar_radius": 0.28822}},
    "22": {"label": "FC6", "coordinates": {"polar_angle_deg": 69.332, "polar_radius": 0.40823}},
    "23": {"label": "FT8", "coordinates": {"polar_angle_deg": 71.948, "polar_radius": 0.53192}},
    "24": {"label": "T7", "coordinates": {"polar_angle_deg": -90.0, "polar_radius": 0.53318}},
    "25": {"label": "C5", "coordinates": {"polar_angle_deg": -90.0, "polar_radius": 0.3999}},
    "26": {"label": "C3", "coordinates": {"polar_angle_deg": -90.0, "polar_radius": 0.26669}},
    "27": {"label": "C1", "coordinates": {"polar_angle_deg": -90.0, "polar_radius": 0.13319}},
    "28": {"label": "Cz", "coordinates": {"polar_angle_deg": 177.496, "polar_radius": 0.00918}},
    "29": {"label": "C2", "coordinates": {"polar_angle_deg": 90.0, "polar_radius": 0.13348}},
    "30": {"label": "C4", "coordinates": {"polar_angle_deg": 90.0, "polar_radius": 0.26667}},
    "31": {"label": "C6", "coordinates": {"polar_angle_deg": 90.0, "polar_radius": 0.3999}},
    "32": {"label": "T8", "coordinates": {"polar_angle_deg": 90.0, "polar_radius": 0.53318}},
    "33": {"label": "M1", "coordinates": {"polar_angle_deg": -100.42, "polar_radius": 0.74733}},
    "34": {"label": "TP7", "coordinates": {"polar_angle_deg": -108.05, "polar_radius": 0.53192}},
    "35": {"label": "CP5", "coordinates": {"polar_angle_deg": -110.67, "polar_radius": 0.40823}},
    "36": {"label": "CP3", "coordinates": {"polar_angle_deg": -117.57, "polar_radius": 0.28822}},
    "37": {"label": "CP1", "coordinates": {"polar_angle_deg": -135.07, "polar_radius": 0.18118}},
    "38": {"label": "CPZ", "coordinates": {"polar_angle_deg": 180.0, "polar_radius": 0.12662}},
    "39": {"label": "CP2", "coordinates": {"polar_angle_deg": 135.07, "polar_radius": 0.18118}},
    "40": {"label": "CP4", "coordinates": {"polar_angle_deg": 117.57, "polar_radius": 0.28822}},
    "41": {"label": "CP6", "coordinates": {"polar_angle_deg": 110.67, "polar_radius": 0.40823}},
    "42": {"label": "TP8", "coordinates": {"polar_angle_deg": 108.11, "polar_radius": 0.53191}},
    "43": {"label": "M2", "coordinates": {"polar_angle_deg": 100.42, "polar_radius": 0.74733}},
    "44": {"label": "P7", "coordinates": {"polar_angle_deg": -126.09, "polar_radius": 0.52808}},
    "45": {"label": "P5", "coordinates": {"polar_angle_deg": -130.6, "polar_radius": 0.43159}},
    "46": {"label": "P3", "coordinates": {"polar_angle_deg": -140.05, "polar_radius": 0.34459}},
    "47": {"label": "P1", "coordinates": {"polar_angle_deg": -156.51, "polar_radius": 0.27903}},
    "48": {"label": "PZ", "coordinates": {"polar_angle_deg": 180.0, "polar_radius": 0.25338}},
    "49": {"label": "P2", "coordinates": {"polar_angle_deg": 156.51, "polar_radius": 0.27878}},
    "50": {"label": "P4", "coordinates": {"polar_angle_deg": 140.1, "polar_radius": 0.3445}},
    "51": {"label": "P6", "coordinates": {"polar_angle_deg": 130.6, "polar_radius": 0.43128}},
    "52": {"label": "P8", "coordinates": {"polar_angle_deg": 126.13, "polar_radius": 0.52807}},
    "53": {"label": "PO7", "coordinates": {"polar_angle_deg": -144.11, "polar_radius": 0.52233}},
    "54": {"label": "PO5", "coordinates": {"polar_angle_deg": -149.46, "polar_radius": 0.46649}},
    "55": {"label": "PO3", "coordinates": {"polar_angle_deg": -157.54, "polar_radius": 0.42113}},
    "56": {"label": "POz", "coordinates": {"polar_angle_deg": 180.0, "polar_radius": 0.37994}},
    "57": {"label": "PO4", "coordinates": {"polar_angle_deg": 157.54, "polar_radius": 0.42113}},
    "58": {"label": "PO6", "coordinates": {"polar_angle_deg": 149.46, "polar_radius": 0.46649}},
    "59": {"label": "PO8", "coordinates": {"polar_angle_deg": 144.14, "polar_radius": 0.52231}},
    "60": {"label": "CB1", "coordinates": {"polar_angle_deg": -170.0, "polar_radius": 0.52}},
    "61": {"label": "O1", "coordinates": {"polar_angle_deg": -162.07, "polar_radius": 0.51499}},
    "62": {"label": "Oz", "coordinates": {"polar_angle_deg": 180.0, "polar_radius": 0.50669}},
    "63": {"label": "O2", "coordinates": {"polar_angle_deg": 162.07, "polar_radius": 0.51499}},
    "64": {"label": "CB2", "coordinates": {"polar_angle_deg": 170.0, "polar_radius": 0.52}},
}


def build_targets():
    targets = {"count": 40, "type": "ssvep"}
    for i, f in enumerate(_FREQS):
        targets[str(i)] = {"label": f"{f} Hz", "stimulus_frequency_hz": f}
    return targets


def build_data_structure(signals_dir):
    structure = {}
    for fname in sorted(os.listdir(signals_dir)):
        m = re.match(r"S(\d+)\.mat", fname)
        if not m:
            continue
        structure[str(int(m.group(1)))] = {"file": f"raw/signals_labels/{fname}"}
    return structure


def build_metadata(dataset_name, window_seconds, signals_dir):
    info = {k: (v.format(w=window_seconds) if isinstance(v, str) and "{w" in v else v)
            for k, v in DATASET_INFO_TEMPLATE.items()}
    structure = build_data_structure(signals_dir)
    return {
        "data_metadata": {
            "dataset_name": dataset_name,
            "dataset_info": info,
            "acquisition": {
                "sample_frequency": 250,
                "window_size_seconds": window_seconds,
                "num_subjects": len(structure),
                "num_runs_per_subject": 4,
            },
            "targets": build_targets(),
            "channels": {"count": len(CHANNELS), "system": "10-20 International System", **CHANNELS},
        },
        "data_structure": structure,
    }


def main():
    for name, window_seconds in (("BETA_4s", 4.0), ("BETA_3s", 3.0)):
        signals_dir = os.path.join(DATAS_ROOT, name, "raw", "signals_labels")
        meta = build_metadata(name, window_seconds, signals_dir)
        out_path = os.path.join(DATAS_ROOT, name, "metadata.json")
        with open(out_path, "w") as f:
            json.dump(meta, f, indent=4)
        print(f"wrote {out_path}: {len(meta['data_structure'])} subjects")


if __name__ == "__main__":
    main()
