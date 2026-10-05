"""
LSE_IndiaTanzania (BIDS, NEMAR on007358) -- metadata.json generator: scans raw/ (see IO/loader.py write_bids_metadata).
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), events ignored.
large-scale resting / task EEG (India + Tanzania subset).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_bids_metadata

write_bids_metadata(
    os.path.dirname(os.path.abspath(__file__)), "LSE_IndiaTanzania",
    dataset_info={
        "source_url": "https://nemar.org/dataexplorer/detail?dataset_id=on007358",
        "file_format": "BIDS (edf)",
        "description": "A subset of large-scale EEG dataset (India + Tanzania): large-scale resting / task EEG (India + Tanzania subset)",
    },
)
