"""
BNCI2014001, loaded through MOABB's BNCI2014_001 (1.5.0) -- metadata.json generator. Raw files are
downloaded by MOABB into raw/ (its own MNE-<code>-data layout) on first use.

BCI Competition IV 2a: 9 subjects, 2 sessions (T and E, both labelled via MOABB) x 6 runs x 48 trials, 4-class MI, 22 EEG (+3 EOG dropped) at 250 Hz. Window: global pre/post around the MI onset (cue, MOABB interval [2, 6]).

Label indices keep the order of the pre-MOABB loader (target_labels key order).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "BNCI2014001", "BNCI2014_001",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html",
        "file_format": "via MOABB",
        "description": 'BCI Competition IV 2a: 9 subjects, 2 sessions (T and E, both labelled via MOABB) x 6 runs x 48 trials, 4-class MI, 22 EEG (+3 EOG dropped) at 250 Hz. Window: global pre/post around the MI onset (cue, MOABB interval [2, 6]).',
    },
    target_labels={'left_hand': 'Left hand', 'right_hand': 'Right hand', 'feet': 'Feet', 'tongue': 'Tongue'},
    kwargs={},
    window=None,
    onset_window=[0.0, 4.0],   # EEG-FM-Compass: cue -> +4 s (MOABB MotorImagery interval)
)
