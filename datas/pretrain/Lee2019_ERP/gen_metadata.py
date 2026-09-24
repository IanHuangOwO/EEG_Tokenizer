"""
Lee2019_ERP, loaded through MOABB's Lee2019_ERP (1.5.0) -- metadata.json generator (downloads subject 1).
Then `python datas/pretrain/Lee2019_ERP/fetch.py` downloads the rest (resumable).

OpenBMI ERP speller (same 54 subjects as Lee2019_MI): 2 sessions x (train + test runs), 62 EEG ch at 1000 Hz. Flashes are ~0.1 s apart, so trials would overlap: each run is cut into continuous non-overlapping 5 s windows with a dummy label instead (pretraining only).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Lee2019_ERP", "Lee2019_ERP",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Lee2019_ERP.html",
        "file_format": "via MOABB",
        "description": 'OpenBMI ERP speller (same 54 subjects as Lee2019_MI): 2 sessions x (train + test runs), 62 EEG ch at 1000 Hz. Flashes are ~0.1 s apart, so trials would overlap: each run is cut into continuous non-overlapping 5 s windows with a dummy label instead (pretraining only).',
    },
    target_labels={},
    kwargs={},
    window=None,
    continuous_seconds=5.0,
)
