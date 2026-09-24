"""
OpenBMI motor imagery (Lee et al. 2019), loaded through MOABB's Lee2019_MI -- metadata.json
generator. Download: `python datas/pretrain/Lee2019_MI/fetch.py` (resumable, into raw/ in
MOABB's layout). Verified 2026-09-24 against MOABB 1.5.0 and subject 1's files: 54
subjects x 2 sessions (different days), each with an offline train run and an online test
run of 100 trials (50 left / 50 right hand), both labelled; 62 EEG + 4 EMG channels at
1000 Hz (EMG dropped). MOABB interval [0, 4] s after the cue.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Lee2019_MI", "Lee2019_MI",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Lee2019_MI.html",
        "file_format": "MATLAB (via MOABB)",
        "description": "OpenBMI: 54 subjects, 2 sessions, left/right-hand motor imagery, "
                       "62ch EEG at 1000 Hz, 200 trials per session (train + test runs).",
        "task_type": "motor_imagery",
        "reference": "Lee MH et al. (2019). EEG dataset and OpenBMI toolbox for three BCI "
                     "paradigms: an investigation into BCI illiteracy. GigaScience 8(5). "
                     "doi:10.1093/gigascience/giz002. GigaDB 100542.",
    },
    target_labels={"left_hand": "Left hand", "right_hand": "Right hand"},
    kwargs={"test_run": True},
)
