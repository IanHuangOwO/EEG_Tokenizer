"""
Kalunga2016 (SSVEP Exo), loaded through MOABB's Kalunga2016 -- metadata.json generator. Raw files are downloaded by
MOABB into raw/ (its own MNE-<code>-data layout) on first use.

SSVEP DEV set (2026-10-05): tunes the SSVEP heads, never reported (the reported SSVEP cell is Nakanishi2015). Not in
the pretraining corpus. 12 subjects, 1 session, 8 occipital EEG (g.tec MobiLab, 256 Hz), 4 classes: LEDs at 13 / 17 /
21 Hz and rest, 16 trials per class. Window [2, 4] s from the cue, MOABB's interval for this dataset (the first 2 s
are the gaze shift / transition).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Kalunga2016", "Kalunga2016",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Kalunga2016.html",
        "reference": "Kalunga et al., Online SSVEP-based BCI using Riemannian geometry, Neurocomputing 2016 (doi 10.1016/j.neucom.2016.01.007)",
        "file_format": "via MOABB",
        "description": "SSVEP DEV set, not reported. 12 subjects, 8 occipital EEG at 256 Hz, 4 classes (13 / 17 / 21 Hz, rest), 16 trials per class; window [2, 4] s from the cue (MOABB interval).",
    },
    target_labels={'13': '13 Hz', '17': '17 Hz', '21': '21 Hz', 'rest': 'rest'},
    kwargs={},
    window=[2.0, 4.0],
)
