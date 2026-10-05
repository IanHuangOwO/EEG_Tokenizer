"""
Ma2020, loaded through MOABB's Ma2020 (1.5.0) -- metadata.json generator (reads subject 1 from raw/raw).
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), events ignored.
motor imagery, multi-session.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Ma2020", "Ma2020",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Ma2020.html",
        "file_format": "via MOABB",
        "description": "motor imagery, multi-session (pretraining only, continuous windows).",
    },
    target_labels={},
    kwargs={},
)
