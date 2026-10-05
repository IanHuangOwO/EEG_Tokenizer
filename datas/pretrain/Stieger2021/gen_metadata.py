"""
Stieger2021, loaded through MOABB's Stieger2021 (1.5.0) -- metadata.json generator (reads subject 1 from raw/raw).
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), events ignored.
motor imagery, 7-11 sessions (cursor control).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Stieger2021", "Stieger2021",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Stieger2021.html",
        "file_format": "via MOABB",
        "description": "motor imagery, 7-11 sessions (cursor control) (pretraining only, continuous windows).",
    },
    target_labels={},
    kwargs={},
)
