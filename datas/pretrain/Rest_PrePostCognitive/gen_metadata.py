"""
Rest_PrePostCognitive (BIDS, NEMAR on005385) -- metadata.json generator: scans raw/ (see IO/loader.py write_bids_metadata).
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), events ignored.
resting state before / after a cognitive task.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_bids_metadata

write_bids_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Rest_PrePostCognitive",
    dataset_info={
        "source_url": "https://nemar.org/dataexplorer/detail?dataset_id=on005385",
        "source_version": "on005385 v1.0.0",
        "file_format": "BIDS (edf)",
        "description": "Resting-state EEG data before and after cognitive activity across the adult lifespan and a 5-year follow-up: resting state before / after a cognitive task",
    },
)
