"""EEGMAT (PhysioNet, mental arithmetic workload) loader -- see
gen_metadata.py for the format/label notes and docs/agents/adding-a-dataset.md
for the general contract."""
from typing import Dict, List, Optional, Tuple

import numpy as np

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.file_path = self._resolve(entry['file'])
        self.label = entry['label']  # per-subject workload-quality class, see gen_metadata.py

    def _load_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.standard_window is None:
            print(f"  [Warning] Subject {self.subject_id}: No standard_window defined, cannot segment data.")
            return None, None

        import mne

        if not self._existing([self.file_path]):
            return None, None

        raw = mne.io.read_raw_edf(self.file_path, preload=True, verbose=False)
        self._resample_if_needed(raw)
        data = raw.get_data(picks=self.channel_indices)  # [C, T]

        trial_len = int(self.standard_window * self.sample_freq)
        n_windows = data.shape[-1] // trial_len
        if n_windows == 0:
            return None, None

        windows = [data[:, i * trial_len:(i + 1) * trial_len] for i in range(n_windows)]
        eeg_data = np.stack(windows)  # (N, C, trial_len)
        labels = np.full(n_windows, self.label, dtype=np.int64)

        return eeg_data, labels
