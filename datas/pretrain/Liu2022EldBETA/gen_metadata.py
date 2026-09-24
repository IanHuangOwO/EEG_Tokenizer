"""
Liu2022EldBETA, loaded through MOABB's Liu2022EldBETA (1.5.0) -- metadata.json generator (downloads subject 1).
Then `python datas/pretrain/Liu2022EldBETA/fetch.py` downloads the rest (resumable).

eldBETA: SSVEP from 100 ELDERLY subjects, 9 targets x 7 blocks, 64 EEG ch at 1000 Hz. Window [0, 5] s from stimulus onset (5 s flicker).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Liu2022EldBETA", "Liu2022EldBETA",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Liu2022EldBETA.html",
        "file_format": "via MOABB",
        "description": 'eldBETA: SSVEP from 100 ELDERLY subjects, 9 targets x 7 blocks, 64 EEG ch at 1000 Hz. Window [0, 5] s from stimulus onset (5 s flicker).',
    },
    target_labels={},
    kwargs={},
    window=[0.0, 5.0],
    continuous_seconds=None,
)
