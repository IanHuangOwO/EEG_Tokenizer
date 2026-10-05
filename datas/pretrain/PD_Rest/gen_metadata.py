"""
PD_Rest (BIDS, NEMAR on008768) -- metadata.json generator: scans raw/ (see IO/loader.py write_bids_metadata).
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), events ignored.
Parkinson's disease resting state.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_bids_metadata

write_bids_metadata(
    os.path.dirname(os.path.abspath(__file__)), "PD_Rest",
    dataset_info={
        "source_url": "https://nemar.org/dataexplorer/detail?dataset_id=on008768",
        "source_version": "on008768 v1.0.0",
        "file_format": "BIDS (vhdr)",
        "description": "Resting-State EEG in Parkinson's Disease and Healthy Controls: Parkinson's disease resting state",
    },
)
