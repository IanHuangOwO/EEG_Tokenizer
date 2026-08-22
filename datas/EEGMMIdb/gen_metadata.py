import json
import os
import re

ROOT = os.path.dirname(__file__)

# EEG Motor Movement/Imagery Database -- PhysioNet.
# Reference: Goldberger, A. L., Amaral, L. A., Glass, L., et al. PhysioBank,
# PhysioToolkit, and PhysioNet: Components of a new research resource for
# complex physiologic signals. Circulation 101(23):e215-e220 (2000).
# Source: https://www.physionet.org/content/eegmmidb/1.0.0/
DATASET_INFO = {
    "source_url": "https://www.physionet.org/content/eegmmidb/1.0.0/",
    "file_format": "EDF+",
    "description": "EEG Motor Movement/Imagery Database (EEGMMIdb) - PhysioNet collection",
    "electrode_note": "64 electrodes from 10-10 international system (excluding Nz, F9, F10, FT9, FT10, A1, A2, TP9, TP10, P9, P10)",
    "contact": "PhysioNet",
    "reference": "Goldberger, A. L., Amaral, L. A., Glass, L., et al. PhysioBank, PhysioToolkit, and PhysioNet: "
                 "Components of a new research resource for complex physiologic signals. Circulation 101(23):e215-e220 (2000).",
}

TARGETS = {
    "count": 3,
    "type": "motor_imagery",
    "0": {"label": "Rest", "description": "T0 - Resting state (no motor imagery)", "stimulus_frequency_hz": None},
    "1": {"label": "Left fist or both fists", "description": "T1 - Left fist or both fists motor imagery", "stimulus_frequency_hz": None},
    "2": {"label": "Right fist or both feet", "description": "T2 - Right fist or both feet motor imagery", "stimulus_frequency_hz": None},
}

# 64-channel 10-10 montage. Raw EDF channel names carry BCI2000's dotted
# padding (e.g. "Fc5."), preserved as original_label. Midline sites
# (Fcz/Cpz/Fpz/Afz/Pz/Poz/Oz/Iz) omit 'coordinates' entirely -- MNE's
# standard_1020 montage always resolves these by label, and the true
# digitized positions for these particular sites weren't otherwise sourced.
CHANNELS = {
    "1": {"label": "FC5", "original_label": "Fc5.", "coordinates": {"polar_angle_deg": -76.42590504508098, "polar_radius": 0.7943370453969775}},
    "2": {"label": "FC3", "original_label": "Fc3.", "coordinates": {"polar_angle_deg": -69.32053429378861, "polar_radius": 0.6432640849643324}},
    "3": {"label": "FC1", "original_label": "Fc1.", "coordinates": {"polar_angle_deg": -52.63312388070132, "polar_radius": 0.428577922298851}},
    "4": {"label": "FCZ", "original_label": "Fcz."},
    "5": {"label": "FC2", "original_label": "Fc2.", "coordinates": {"polar_angle_deg": 52.76309456726431, "polar_radius": 0.43690916323876755}},
    "6": {"label": "FC4", "original_label": "Fc4.", "coordinates": {"polar_angle_deg": 69.15189142384068, "polar_radius": 0.6665734428740767}},
    "7": {"label": "FC6", "original_label": "Fc6.", "coordinates": {"polar_angle_deg": 75.9283864803437, "polar_radius": 0.8199454370444414}},
    "8": {"label": "C5", "original_label": "C5..", "coordinates": {"polar_angle_deg": -99.72577392042757, "polar_radius": 0.8145074462581665}},
    "9": {"label": "C3", "original_label": "C3..", "coordinates": {"polar_angle_deg": -100.09120486162583, "polar_radius": 0.6638507121710421}},
    "10": {"label": "C1", "original_label": "C1..", "coordinates": {"polar_angle_deg": -105.43582452956097, "polar_radius": 0.37511054680053985}},
    "11": {"label": "CZ", "original_label": "Cz..", "coordinates": {"polar_angle_deg": 177.4958818648799, "polar_radius": 0.09175762083336728}},
    "12": {"label": "C2", "original_label": "C2..", "coordinates": {"polar_angle_deg": 104.33088263904357, "polar_radius": 0.3888190947998824}},
    "13": {"label": "C4", "original_label": "C4..", "coordinates": {"polar_angle_deg": 99.22459761294736, "polar_radius": 0.679972723019093}},
    "14": {"label": "C6", "original_label": "C6..", "coordinates": {"polar_angle_deg": 98.70385908672239, "polar_radius": 0.844282007773469}},
    "15": {"label": "CP5", "original_label": "Cp5.", "coordinates": {"polar_angle_deg": -120.32187527550083, "polar_radius": 0.9220567212124209}},
    "16": {"label": "CP3", "original_label": "Cp3.", "coordinates": {"polar_angle_deg": -126.4881645271155, "polar_radius": 0.7905199450918363}},
    "17": {"label": "CP1", "original_label": "Cp1.", "coordinates": {"polar_angle_deg": -143.09586519893588, "polar_radius": 0.5914139055872799}},
    "18": {"label": "CPZ", "original_label": "Cpz."},
    "19": {"label": "CP2", "original_label": "Cp2.", "coordinates": {"polar_angle_deg": 140.8059092127152, "polar_radius": 0.6073872608188288}},
    "20": {"label": "CP4", "original_label": "Cp4.", "coordinates": {"polar_angle_deg": 124.99718074071295, "polar_radius": 0.813151912195993}},
    "21": {"label": "CP6", "original_label": "Cp6.", "coordinates": {"polar_angle_deg": 118.95541217578682, "polar_radius": 0.9522527089449523}},
    "22": {"label": "FP1", "original_label": "Fp1.", "coordinates": {"polar_angle_deg": -19.330007944279625, "polar_radius": 0.8893030405491709}},
    "23": {"label": "FPZ", "original_label": "Fpz."},
    "24": {"label": "FP2", "original_label": "Fp2.", "coordinates": {"polar_angle_deg": 19.38542872808649, "polar_radius": 0.8999815633722726}},
    "25": {"label": "AF7", "original_label": "Af7.", "coordinates": {"polar_angle_deg": -38.650605787642725, "polar_radius": 0.8780398230678378}},
    "26": {"label": "AF3", "original_label": "Af3.", "coordinates": {"polar_angle_deg": -23.682223771278377, "polar_radius": 0.8390278372557134}},
    "27": {"label": "AFZ", "original_label": "Afz."},
    "28": {"label": "AF4", "original_label": "Af4.", "coordinates": {"polar_angle_deg": 24.677106634440914, "polar_radius": 0.8553761688345075}},
    "29": {"label": "AF8", "original_label": "Af8.", "coordinates": {"polar_angle_deg": 38.66876483804009, "polar_radius": 0.8921538702000905}},
    "30": {"label": "F7", "original_label": "F7..", "coordinates": {"polar_angle_deg": -58.84681304243345, "polar_radius": 0.8210323548374937}},
    "31": {"label": "F5", "original_label": "F5..", "coordinates": {"polar_angle_deg": -53.30915771914511, "polar_radius": 0.8039421257609282}},
    "32": {"label": "F3", "original_label": "F3..", "coordinates": {"polar_angle_deg": -43.410838497590674, "polar_radius": 0.7311114144834561}},
    "33": {"label": "F1", "original_label": "F1..", "coordinates": {"polar_angle_deg": -25.778974747105302, "polar_radius": 0.6322316952549911}},
    "34": {"label": "FZ", "original_label": "Fz..", "coordinates": {"polar_angle_deg": 0.3057077627961028, "polar_radius": 0.5851283289023016}},
    "35": {"label": "F2", "original_label": "F2..", "coordinates": {"polar_angle_deg": 27.129803194947772, "polar_radius": 0.6472300120706703}},
    "36": {"label": "F4", "original_label": "F4..", "coordinates": {"polar_angle_deg": 43.667669733479386, "polar_radius": 0.7507331705393068}},
    "37": {"label": "F6", "original_label": "F6..", "coordinates": {"polar_angle_deg": 53.73192644346953, "polar_radius": 0.8423382671902065}},
    "38": {"label": "F8", "original_label": "F8..", "coordinates": {"polar_angle_deg": 58.693815790713, "polar_radius": 0.8549024440542909}},
    "39": {"label": "FT7", "original_label": "Ft7.", "coordinates": {"polar_angle_deg": -80.08430223446933, "polar_radius": 0.8199989937243825}},
    "40": {"label": "FT8", "original_label": "Ft8.", "coordinates": {"polar_angle_deg": 79.32868817343721, "polar_radius": 0.8325494115606592}},
    "41": {"label": "T7", "original_label": "T7..", "coordinates": {"polar_angle_deg": -100.77642362288974, "polar_radius": 0.8567198785425724}},
    "42": {"label": "T8", "original_label": "T8..", "coordinates": {"polar_angle_deg": 100.01202894815384, "polar_radius": 0.8639559477253456}},
    "43": {"label": "T9", "original_label": "T9..", "coordinates": {"polar_angle_deg": -100.44141246827564, "polar_radius": 0.8734039247965399}},
    "44": {"label": "T10", "original_label": "T10.", "coordinates": {"polar_angle_deg": 100.82576354258411, "polar_radius": 0.8711020965248562}},
    "45": {"label": "TP7", "original_label": "Tp7.", "coordinates": {"polar_angle_deg": -118.48051813638996, "polar_radius": 0.9650989432659224}},
    "46": {"label": "TP8", "original_label": "Tp8.", "coordinates": {"polar_angle_deg": 118.0303773930428, "polar_radius": 0.9691734382209407}},
    "47": {"label": "P7", "original_label": "P7..", "coordinates": {"polar_angle_deg": -135.3999607509189, "polar_radius": 1.0316020043495455}},
    "48": {"label": "P5", "original_label": "P5..", "coordinates": {"polar_angle_deg": -138.59450681731226, "polar_radius": 1.0171446924494076}},
    "49": {"label": "P3", "original_label": "P3..", "coordinates": {"polar_angle_deg": -146.0679005881858, "polar_radius": 0.9495941913328029}},
    "50": {"label": "P1", "original_label": "P1..", "coordinates": {"polar_angle_deg": -160.43368108248998, "polar_radius": 0.854598215075365}},
    "51": {"label": "PZ", "original_label": "Pz.."},
    "52": {"label": "P2", "original_label": "P2..", "coordinates": {"polar_angle_deg": 158.367635802523, "polar_radius": 0.8658545209502574}},
    "53": {"label": "P4", "original_label": "P4..", "coordinates": {"polar_angle_deg": 144.67912725864227, "polar_radius": 0.9628336571251546}},
    "54": {"label": "P6", "original_label": "P6..", "coordinates": {"polar_angle_deg": 138.19101442334932, "polar_radius": 1.0183419155558706}},
    "55": {"label": "P8", "original_label": "P8..", "coordinates": {"polar_angle_deg": 135.00494050819904, "polar_radius": 1.033252716782298}},
    "56": {"label": "PO7", "original_label": "Po7.", "coordinates": {"polar_angle_deg": -150.65074594060442, "polar_radius": 1.1188905554418627}},
    "57": {"label": "PO3", "original_label": "Po3.", "coordinates": {"polar_angle_deg": -160.0984128428656, "polar_radius": 1.0725851839537035}},
    "58": {"label": "POZ", "original_label": "Poz."},
    "59": {"label": "PO4", "original_label": "Po4.", "coordinates": {"polar_angle_deg": 159.96211790757454, "polar_radius": 1.0734722664964382}},
    "60": {"label": "PO8", "original_label": "Po8.", "coordinates": {"polar_angle_deg": 150.30787135744788, "polar_radius": 1.1238073903285206}},
    "61": {"label": "O1", "original_label": "O1..", "coordinates": {"polar_angle_deg": -165.34150269088192, "polar_radius": 1.1623220595239514}},
    "62": {"label": "OZ", "original_label": "Oz.."},
    "63": {"label": "O2", "original_label": "O2..", "coordinates": {"polar_angle_deg": 165.09990657799872, "polar_radius": 1.1605838664551562}},
    "64": {"label": "IZ", "original_label": "Iz.."},
}


def build_data_structure(root):
    structure = {}
    for entry in sorted(os.listdir(root)):
        m = re.match(r"S(\d+)$", entry)
        if not m or not os.path.isdir(os.path.join(root, entry)):
            continue
        subject_num = int(m.group(1))
        runs = sorted(
            f for f in os.listdir(os.path.join(root, entry))
            if re.match(rf"S{m.group(1)}R\d+\.edf$", f)
        )
        structure[str(subject_num)] = {"folder": f"raw/{entry}", "runs": runs}
    return structure


def main():
    root = os.path.join(ROOT, "raw")
    structure = build_data_structure(root)

    meta = {
        "data_metadata": {
            "dataset_name": "EEGMMIdb",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 160.0,
                "window_size_seconds": 4.0,
                "num_subjects": len(structure),
                "num_runs_per_subject": 14,
            },
            "targets": TARGETS,
            "channels": {"count": len(CHANNELS), "system": "10-10 International System", **CHANNELS},
        },
        "data_structure": structure,
    }

    out_path = os.path.join(ROOT, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"wrote {out_path}: {len(structure)} subjects")


if __name__ == "__main__":
    main()
