"""SRM Resting-state EEG loader -- see gen_metadata.py for the format/dataset
notes and docs/agents/adding-a-dataset.md for the general contract."""
from typing import Dict, List, Optional, Tuple

import numpy as np

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.file_paths = [self._resolve(f) for f in entry['files']]  # shape C: one file per available session

    def _load_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        import mne

        existing = self._existing(self.file_paths)
        if not existing:
            return None, None

        sessions = []
        for path in existing:
            try:
                # A handful of this dataset's EDF headers have a malformed recording-start
                # timestamp (e.g. sub-041: "second must be in 0..59") that MNE's EDF reader
                # rejects outright -- a real per-file data-quality issue, not a code bug.
                # Skip just this session rather than losing the whole subject/dataset.
                raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            except Exception as e:
                print(f"  [Warning] {path}: failed to read ({e!r}), skipping this session")
                continue
            self._resample_if_needed(raw)
            sessions.append(raw.get_data(picks=self.channel_indices))  # [C, T]

        if not sessions:
            return None, None

        # Session durations can differ by a handful of samples (EDF discretization) --
        # truncate to the shortest so they stack into one (N, C, T) array, same
        # pattern as datas/finetune/Nakanishi2015/loader.py's split-file length mismatch handling.
        min_t = min(s.shape[-1] for s in sessions)
        eeg_data = np.stack([s[:, :min_t] for s in sessions], axis=0)  # [N=n_sessions, C, T]

        # No events/conditions at all (see gen_metadata.py's dataset_info.notes) --
        # every "trial" (session) gets the single placeholder class 0.
        labels = np.zeros(len(sessions), dtype=np.int64)

        return eeg_data, labels
