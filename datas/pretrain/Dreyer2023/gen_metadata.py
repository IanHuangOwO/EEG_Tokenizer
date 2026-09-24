"""
Dreyer2023, loaded through MOABB's Dreyer2023 (1.5.0) -- metadata.json generator (downloads subject 1).
Then `python datas/pretrain/Dreyer2023/fetch.py` downloads the rest (resumable).

Motor imagery L/R hand, 87 subjects (A/B/C subsets), 27 EEG ch at 512 Hz, 240 trials. Window: global pre/post around the MI onset (MOABB interval [0, 5]).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Dreyer2023", "Dreyer2023",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Dreyer2023.html",
        "file_format": "via MOABB",
        "description": 'Motor imagery L/R hand, 87 subjects (A/B/C subsets), 27 EEG ch at 512 Hz, 240 trials. Window: global pre/post around the MI onset (MOABB interval [0, 5]).',
    },
    target_labels={},
    kwargs={},
    window=None,
    continuous_seconds=None,
)
