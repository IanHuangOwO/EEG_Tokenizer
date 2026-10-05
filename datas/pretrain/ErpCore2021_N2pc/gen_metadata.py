"""
ErpCore2021_N2pc, loaded through MOABB's ErpCore2021_N2pc (1.5.0) -- metadata.json generator (reads subject 1 from raw/../ErpCore2021/raw).
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), events ignored.
ERP CORE N2pc task. Raw files shared with the other ErpCore2021 folders via raw_root (same people: cohort ErpCore2021).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "ErpCore2021_N2pc", "ErpCore2021_N2pc",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.ErpCore2021_N2pc.html",
        "file_format": "via MOABB",
        "description": "ERP CORE N2pc task (pretraining only, continuous windows).",
    },
    target_labels={},
    kwargs={},
    raw_root="../ErpCore2021",
    cohort="ErpCore2021",
)
