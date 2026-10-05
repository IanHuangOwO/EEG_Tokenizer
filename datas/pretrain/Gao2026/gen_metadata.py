"""
Gao2026, loaded through MOABB's Gao2026 (1.5.0) -- metadata.json generator (reads subject 1 from raw/raw).
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), events ignored.
visual imagery.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Gao2026", "Gao2026",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Gao2026.html",
        "file_format": "via MOABB",
        "description": "visual imagery (pretraining only, continuous windows).",
    },
    target_labels={},
    kwargs={},
)
