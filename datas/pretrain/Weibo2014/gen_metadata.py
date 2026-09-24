"""
Weibo2014, loaded through MOABB's Weibo2014 (1.5.0) -- metadata.json generator (downloads subject 1).
Then `python datas/pretrain/Weibo2014/fetch.py` downloads the rest (resumable).

Motor imagery, 7 classes (single/combined limbs + rest): 10 subjects, 60 EEG ch at 200 Hz, 560 trials. Window: global pre/post around MI onset (MOABB interval [3, 7]).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Weibo2014", "Weibo2014",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Weibo2014.html",
        "file_format": "via MOABB",
        "description": 'Motor imagery, 7 classes (single/combined limbs + rest): 10 subjects, 60 EEG ch at 200 Hz, 560 trials. Window: global pre/post around MI onset (MOABB interval [3, 7]).',
    },
    target_labels={},
    kwargs={},
    window=None,
    continuous_seconds=None,
)
