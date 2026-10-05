"""
GuttmannFlury2025_MI, loaded through MOABB's GuttmannFlury2025_MI (1.5.0) -- metadata.json generator (reads subject 1 from raw/../GuttmannFlury2025/raw).
Pretraining only: compiled as continuous windows (compile_params.continuous_seconds), events ignored.
Eye-BCI MI. Raw files shared with the other GuttmannFlury2025 folders via raw_root (same people: cohort GuttmannFlury2025).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata


def drop_non_eeg(meta):
    """MOABB types HEO (horizontal EOG) and Trig as EEG here: drop them (CB1 / CB2, cerebellar EEG, stay)."""
    ch = meta["data_metadata"]["channels"]
    keep = [ch[str(i + 1)] for i in range(ch["count"]) if ch[str(i + 1)]["label"] not in ("HEO", "VEO", "Trig")]
    meta["data_metadata"]["channels"] = {"count": len(keep), **{str(i + 1): c for i, c in enumerate(keep)}}

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "GuttmannFlury2025_MI", "GuttmannFlury2025_MI",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.GuttmannFlury2025_MI.html",
        "file_format": "via MOABB",
        "description": "Eye-BCI MI (pretraining only, continuous windows).",
    },
    target_labels={},
    kwargs={},
    raw_root="../GuttmannFlury2025",
    cohort="GuttmannFlury2025",
    postprocess=drop_non_eeg,
)
