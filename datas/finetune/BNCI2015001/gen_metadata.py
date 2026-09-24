"""
BNCI2015001, loaded through MOABB's BNCI2015_001 (1.5.0) -- metadata.json generator. Raw files are
downloaded by MOABB into raw/ (its own MNE-<code>-data layout) on first use.

MI right hand vs both feet: 12 subjects, 2-3 sessions, 13 EEG at 512 Hz. Window: global pre/post around the event, which marks the CUE (MOABB interval [0, 5]). The old hand-written loader assumed it marked trial start and cut [3, 8] s, i.e. mostly post-imagery rebound -- verified 2026-09-24 from the C3/C4 8-30 Hz power time course (ERD 1-5 s, beta rebound 5-7 s after the event).

Label indices keep the order of the pre-MOABB loader (target_labels key order).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "BNCI2015001", "BNCI2015_001",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2015_001.html",
        "file_format": "via MOABB",
        "description": 'MI right hand vs both feet: 12 subjects, 2-3 sessions, 13 EEG at 512 Hz. Window: global pre/post around the event, which marks the CUE (MOABB interval [0, 5]). The old hand-written loader assumed it marked trial start and cut [3, 8] s, i.e. mostly post-imagery rebound -- verified 2026-09-24 from the C3/C4 8-30 Hz power time course (ERD 1-5 s, beta rebound 5-7 s after the event).',
    },
    target_labels={'right_hand': 'right hand', 'feet': 'both feet'},
    kwargs={},
    window=None,
)
