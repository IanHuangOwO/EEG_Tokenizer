"""
BNCI2014004, loaded through MOABB's BNCI2014_004 (1.5.0) -- metadata.json generator. Raw files are
downloaded by MOABB into raw/ (its own MNE-<code>-data layout) on first use.

BCI Competition IV 2b: 9 subjects, 5 sessions (3 T + 2 E, all labelled via MOABB), 2-class MI, bipolar C3/Cz/C4 (+3 EOG dropped) at 250 Hz. Window: global pre/post around the MI onset (MOABB interval [3, 7.5]).

Label indices keep the order of the pre-MOABB loader (target_labels key order).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "BNCI2014004", "BNCI2014_004",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_004.html",
        "file_format": "via MOABB",
        "description": 'BCI Competition IV 2b: 9 subjects, 5 sessions (3 T + 2 E, all labelled via MOABB), 2-class MI, bipolar C3/Cz/C4 (+3 EOG dropped) at 250 Hz. Window: global pre/post around the MI onset (MOABB interval [3, 7.5]).',
    },
    target_labels={'left_hand': 'Left hand', 'right_hand': 'Right hand'},
    kwargs={},
    window=None,
    onset_window=[0.0, 4.5],   # EEG-FM-Compass: cue -> +4.5 s (MOABB MotorImagery interval)
)
