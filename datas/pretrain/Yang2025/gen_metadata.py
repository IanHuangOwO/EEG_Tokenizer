"""
Yang2025, loaded through MOABB's Yang2025 (1.5.0) -- metadata.json generator (reads subject 1 from raw/raw).
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), events ignored.
multi-day motor imagery (WBCIC-SHU).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Yang2025", "Yang2025",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Yang2025.html",
        "file_format": "via MOABB",
        "description": "multi-day motor imagery (WBCIC-SHU) (pretraining only, continuous windows).",
    },
    target_labels={},
    kwargs={},
)
