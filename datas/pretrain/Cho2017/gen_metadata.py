"""
Cho2017, loaded through MOABB's Cho2017 (1.5.0) -- metadata.json generator (downloads subject 1).
Then `python datas/pretrain/Cho2017/fetch.py` downloads the rest (resumable).

Motor imagery L/R hand (GigaDB 100295): 52 subjects, 64 EEG ch (+4 EMG dropped) at 512 Hz, 200-240 trials. Window: global pre/post around the cue (MOABB interval [0, 3]).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Cho2017", "Cho2017",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Cho2017.html",
        "file_format": "via MOABB",
        "description": 'Motor imagery L/R hand (GigaDB 100295): 52 subjects, 64 EEG ch (+4 EMG dropped) at 512 Hz, 200-240 trials. Window: global pre/post around the cue (MOABB interval [0, 3]).',
    },
    target_labels={},
    kwargs={},
    window=None,
    continuous_seconds=None,
)
