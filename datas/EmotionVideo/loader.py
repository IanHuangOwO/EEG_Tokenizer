"""EmotionVideo (Unicorn Hybrid Black) loader -- see gen_metadata.py for the
format/label notes and docs/agents/adding-a-dataset.md for the general
contract."""
from typing import Dict, List, Optional, Tuple

import numpy as np

from IO.loader import BaseSubjectLoader

N_EEG_CHANNELS = 8  # first 8 CSV columns ("EEG 1".."EEG 8"); accelerometer/gyroscope/battery/counter/validation dropped


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.file_paths = [self._resolve(f) for f in entry['files']]  # shape C: one file per trial (CSV 01-12)

    def _load_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        existing = self._existing(self.file_paths)
        if not existing:
            return None, None

        trials = []
        for path in existing:
            try:
                # A handful of this dataset's CSVs have genuine binary corruption mid-file
                # (a garbled row of non-UTF8 bytes, verified 2026-09-22 -- not a format
                # issue) -- skip just this trial rather than losing the whole subject,
                # same precedent as datas/SRM_RestingState/loader.py's bad-header skip.
                data = np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(N_EEG_CHANNELS))  # (T, 8)
            except (UnicodeDecodeError, ValueError) as e:
                print(f"  [Warning] {path}: failed to parse ({e!r}), skipping this trial")
                continue
            trials.append(data[:, self.channel_indices].T)  # (C, T)

        if not trials:
            return None, None

        # Trial lengths vary by a handful of samples (BLE streaming jitter, see
        # gen_metadata.py's notes) -- truncate to the shortest, same pattern as
        # datas/SRM_RestingState/loader.py's session-length handling.
        min_t = min(t.shape[-1] for t in trials)
        eeg_data = np.stack([t[:, :min_t] for t in trials], axis=0)  # (N=12, C, T)

        # No recoverable emotion/video label (see gen_metadata.py's dataset_info.notes) --
        # every trial gets the single placeholder class 0.
        labels = np.zeros(len(trials), dtype=np.int64)

        return eeg_data, labels
