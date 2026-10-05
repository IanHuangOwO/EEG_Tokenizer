"""
BI2015a, loaded through MOABB's BI2015a (1.5.0) -- metadata.json generator (reads subject 1 from raw/raw).
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), events ignored.
P300 (Brain Invaders 2015a).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "BI2015a", "BI2015a",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.BI2015a.html",
        "file_format": "via MOABB",
        "description": "P300 (Brain Invaders 2015a) (pretraining only, continuous windows).",
    },
    target_labels={},
    kwargs={},
)
