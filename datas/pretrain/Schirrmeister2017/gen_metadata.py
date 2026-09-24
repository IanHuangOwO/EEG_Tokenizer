"""
Schirrmeister2017, loaded through MOABB's Schirrmeister2017 (1.5.0) -- metadata.json generator (downloads subject 1).
Then `python datas/pretrain/Schirrmeister2017/fetch.py` downloads the rest (resumable).

High-gamma dataset: motor EXECUTION (L/R hand, feet, rest), 14 subjects, 128 EEG ch at 500 Hz, ~960 trials per subject (train + test runs). Window: global pre/post around the cue (MOABB interval [0, 4]).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Schirrmeister2017", "Schirrmeister2017",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Schirrmeister2017.html",
        "file_format": "via MOABB",
        "description": 'High-gamma dataset: motor EXECUTION (L/R hand, feet, rest), 14 subjects, 128 EEG ch at 500 Hz, ~960 trials per subject (train + test runs). Window: global pre/post around the cue (MOABB interval [0, 4]).',
    },
    target_labels={},
    kwargs={},
    window=None,
    continuous_seconds=None,
)
