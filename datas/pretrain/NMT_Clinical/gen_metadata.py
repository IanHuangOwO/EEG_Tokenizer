"""
NMT_Clinical (BIDS, NEMAR nm000181) -- metadata.json generator: scans raw/ (see IO/loader.py write_bids_metadata).
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), events ignored.
NMT clinical EEG (Pakistan, normal / abnormal routine recordings).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_bids_metadata

write_bids_metadata(
    os.path.dirname(os.path.abspath(__file__)), "NMT_Clinical",
    dataset_info={
        "source_url": "https://nemar.org/dataexplorer/detail?dataset_id=nm000181",
        "source_version": "nm000181 v1.0.0",
        "file_format": "BIDS (edf)",
        "description": "NMT: Neurodiagnostic Montage Template Scalp EEG: NMT clinical EEG (Pakistan, normal / abnormal routine recordings)",
    },
)
