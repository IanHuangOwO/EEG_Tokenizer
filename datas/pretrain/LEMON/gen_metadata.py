"""
LEMON (BIDS, NEMAR nm000179) -- metadata.json generator: scans raw/ (see IO/loader.py write_bids_metadata).
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), events ignored.
MPI LEMON resting state.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_bids_metadata

write_bids_metadata(
    os.path.dirname(os.path.abspath(__file__)), "LEMON",
    dataset_info={
        "source_url": "https://nemar.org/dataexplorer/detail?dataset_id=nm000179",
        "file_format": "BIDS (vhdr)",
        "description": "LEMON: MPI Leipzig Mind-Brain-Body EEG (Resting State): MPI LEMON resting state",
    },
)
