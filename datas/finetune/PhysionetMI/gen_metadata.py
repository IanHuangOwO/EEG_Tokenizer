"""
PhysionetMI, loaded through MOABB's PhysionetMI (1.5.0) -- metadata.json generator. Raw files are
downloaded by MOABB into raw/ (its own MNE-<code>-data layout) on first use.

PhysioNet EEGMMIDB motor IMAGERY runs only (4, 6, 8, 10, 12, 14): 109 subjects, 64 EEG at 160 Hz, 5 classes (rest, left fist, right fist, both fists, both feet). Window [0, 4] s from each event (back-to-back trials, no pre-event headroom -- same as the old loader).

Label indices keep the order of the pre-MOABB loader (target_labels key order).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "PhysionetMI", "PhysionetMI",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.PhysionetMI.html",
        "file_format": "via MOABB",
        "description": 'PhysioNet EEGMMIDB motor IMAGERY runs only (4, 6, 8, 10, 12, 14): 109 subjects, 64 EEG at 160 Hz, 5 classes (rest, left fist, right fist, both fists, both feet). Window [0, 4] s from each event (back-to-back trials, no pre-event headroom -- same as the old loader).',
    },
    target_labels={'rest': 'Rest', 'left_hand': 'Left fist', 'right_hand': 'Right fist', 'hands': 'Both fists', 'feet': 'Both feet'},
    kwargs={'imagined': True, 'executed': False},
    window=[0.0, 4.0],
)
