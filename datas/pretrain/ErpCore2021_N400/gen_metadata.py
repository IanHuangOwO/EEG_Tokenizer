"""
ErpCore2021_N400, loaded through MOABB's ErpCore2021_N400 (1.5.0) -- metadata.json generator (reads subject 1 from raw/../ErpCore2021/raw).
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), events ignored.
ERP CORE N400 task. Raw files shared with the other ErpCore2021 folders via raw_root (same people: cohort ErpCore2021).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "ErpCore2021_N400", "ErpCore2021_N400",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.ErpCore2021_N400.html",
        "file_format": "via MOABB",
        "description": "ERP CORE N400 task (pretraining only, continuous windows).",
    },
    target_labels={},
    kwargs={},
    raw_root="../ErpCore2021",
    cohort="ErpCore2021",
)
