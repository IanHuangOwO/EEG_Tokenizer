"""
Liu2024, loaded through MOABB's Liu2024 (1.5.0) -- metadata.json generator (reads subject 1 from raw/raw).
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), events ignored.
motor imagery (stroke patients).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Liu2024", "Liu2024",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Liu2024.html",
        "file_format": "via MOABB",
        "description": "motor imagery (stroke patients) (pretraining only, continuous windows).",
    },
    target_labels={},
    kwargs={},
)
