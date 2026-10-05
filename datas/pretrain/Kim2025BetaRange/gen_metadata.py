"""
Kim2025BetaRange, loaded through MOABB's Kim2025BetaRange (1.5.0) -- metadata.json generator (reads subject 1 from raw/raw).
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), events ignored.
40-class beta-range SSVEP speller.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Kim2025BetaRange", "Kim2025BetaRange",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Kim2025BetaRange.html",
        "file_format": "via MOABB",
        "description": "40-class beta-range SSVEP speller (pretraining only, continuous windows).",
    },
    target_labels={},
    kwargs={},
)
