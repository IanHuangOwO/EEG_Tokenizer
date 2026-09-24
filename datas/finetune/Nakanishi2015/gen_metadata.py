"""
Nakanishi2015, loaded through MOABB's Nakanishi2015 (1.5.0) -- metadata.json generator. Raw files are
downloaded by MOABB into raw/ (its own MNE-<code>-data layout) on first use.

SSVEP 12-class: 9 subjects, 15 trials per class, 8 EEG at 256 Hz. Window [0, 4] s from each trial start, so the stimulus onset sits at 0 and is drawable on time-axis plots (the MOABB interval starts at 0.15 s visual latency; trials are stored with only a short zero buffer between them, so no pre-stimulus headroom). MOABB has 9 subjects of the original 12-class SSVEP data. The old raw copy (DataSub_N.mat, 10 subjects, 1024 samples/trial) was a filtered re-export, cropped ~0.29 s after trial start -- same trials in the same order, verified 2026-09-24.

Label indices keep the order of the pre-MOABB loader (target_labels key order).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Nakanishi2015", "Nakanishi2015",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Nakanishi2015.html",
        "file_format": "via MOABB",
        "description": 'SSVEP 12-class: 9 subjects, 15 trials per class, 8 EEG at 256 Hz. Window [0, 4] s from each trial start, so the stimulus onset sits at 0 and is drawable on time-axis plots (the MOABB interval starts at 0.15 s visual latency; trials are stored with only a short zero buffer between them, so no pre-stimulus headroom). MOABB has 9 subjects of the original 12-class SSVEP data. The old raw copy (DataSub_N.mat, 10 subjects, 1024 samples/trial) was a filtered re-export, cropped ~0.29 s after trial start -- same trials in the same order, verified 2026-09-24.',
    },
    target_labels={'9.25': '9.25 Hz', '11.25': '11.25 Hz', '13.25': '13.25 Hz', '9.75': '9.75 Hz', '11.75': '11.75 Hz', '13.75': '13.75 Hz', '10.25': '10.25 Hz', '12.25': '12.25 Hz', '14.25': '14.25 Hz', '10.75': '10.75 Hz', '12.75': '12.75 Hz', '14.75': '14.75 Hz'},
    kwargs={},
    window=[0.0, 4.0],
)
