"""
Nieuwland2018_N400 (BIDS, NEMAR nm000228) -- metadata.json generator: scans raw/ (see IO/loader.py write_bids_metadata).
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), events ignored.
N400 sentence-reading replication (multi-lab).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_bids_metadata

write_bids_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Nieuwland2018_N400",
    dataset_info={
        "source_url": "https://nemar.org/dataexplorer/detail?dataset_id=nm000228",
        "source_version": "nm000228 v1.1.0",
        "file_format": "BIDS (bdf)",
        "description": "Nieuwland et al. 2018: Multi-site N400 Replication Study: N400 sentence-reading replication (multi-lab)",
    },
)
