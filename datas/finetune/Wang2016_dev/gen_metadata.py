"""
Wang2016_dev -- the Wang2016 SSVEP benchmark cut into trials, as the phase-locked SSVEP DEV set (2026-10-06): tunes the
SSVEP head for Nakanishi2015, never reported. Raw files shared with datas/pretrain/Wang2016 via raw_root.

Caveat: Wang2016 is in the pretraining corpus (event-free windows, no labels), so the backbone has seen this EEG; only
the head's settings are chosen here. Kalunga2016 and MAMEM3 were tried first and are not phase-locked to the trial
onset (raw TRCA at chance). 40 classes (8-15.8 Hz, 0.2 Hz apart, phase-coded), 6 blocks; window [0.5, 4.5] s from the
cue = the first 4 s of flicker, the length of a Nakanishi2015 trial. Runs read the 8 occipital channels of
Nakanishi2015 (channels_to_use).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Wang2016_dev", "Wang2016",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Wang2016.html",
        "file_format": "via MOABB (raw shared with datas/pretrain/Wang2016)",
        "description": "Phase-locked SSVEP DEV set, not reported; in the pretraining corpus as unlabeled windows. 40 classes, window [0.5, 4.5] s.",
    },
    target_labels={},
    kwargs={},
    window=[0.5, 4.5],
    raw_root="../../pretrain/Wang2016",
)
