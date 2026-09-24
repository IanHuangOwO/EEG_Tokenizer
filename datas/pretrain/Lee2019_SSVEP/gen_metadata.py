"""
Lee2019_SSVEP, loaded through MOABB's Lee2019_SSVEP (1.5.0) -- metadata.json generator (downloads subject 1).
Then `python datas/pretrain/Lee2019_SSVEP/fetch.py` downloads the rest (resumable).

OpenBMI SSVEP (same 54 subjects as Lee2019_MI): 2 sessions x (train + test runs), 4 targets, 62 EEG ch at 1000 Hz. Window: global pre/post around stimulus onset (MOABB interval [0, 4]).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Lee2019_SSVEP", "Lee2019_SSVEP",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Lee2019_SSVEP.html",
        "file_format": "via MOABB",
        "description": 'OpenBMI SSVEP (same 54 subjects as Lee2019_MI): 2 sessions x (train + test runs), 4 targets, 62 EEG ch at 1000 Hz. Window: global pre/post around stimulus onset (MOABB interval [0, 4]).',
    },
    target_labels={},
    kwargs={'test_run': True},
    window=None,
    continuous_seconds=None,
)
