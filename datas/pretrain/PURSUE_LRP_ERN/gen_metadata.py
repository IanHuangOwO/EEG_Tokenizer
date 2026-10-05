"""
PURSUE_LRP_ERN (BIDS, NEMAR on007139) -- metadata.json generator: scans raw/ (see IO/loader.py write_bids_metadata).
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), events ignored.
PURSUE LRP_ERN (ERP CORE task replicated across labs). Same participants across the PURSUE tasks: cohort PURSUE (subject ids = BIDS labels).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_bids_metadata

write_bids_metadata(
    os.path.dirname(os.path.abspath(__file__)), "PURSUE_LRP_ERN",
    dataset_info={
        "source_url": "https://nemar.org/dataexplorer/detail?dataset_id=on007139",
        "file_format": "BIDS (set)",
        "description": "PURSUE LRP/ERN Flanker: PURSUE LRP_ERN (ERP CORE task replicated across labs)",
    },
    cohort="PURSUE",
)
