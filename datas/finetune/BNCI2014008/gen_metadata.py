"""
BNCI2014008, loaded through MOABB's BNCI2014_008 (1.5.0) -- metadata.json generator. Raw files are
downloaded by MOABB into raw/ (its own MNE-<code>-data layout) on first use.

P300 speller (ALS patients): 8 subjects, 1 session, 8 EEG at 256 Hz. Window [-0.2, 0.8] s around each flash (overrides global pre/post; flashes are ~0.25 s apart).

Label indices keep the order of the pre-MOABB loader (target_labels key order).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "BNCI2014008", "BNCI2014_008",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_008.html",
        "file_format": "via MOABB",
        "description": 'P300 speller (ALS patients): 8 subjects, 1 session, 8 EEG at 256 Hz. Window [-0.2, 0.8] s around each flash (overrides global pre/post; flashes are ~0.25 s apart).',
    },
    target_labels={'NonTarget': 'non-target', 'Target': 'target'},
    kwargs={},
    window=[-0.2, 0.8],
)
