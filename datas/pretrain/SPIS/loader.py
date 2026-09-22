"""SPIS Resting-State EEG loader -- see gen_metadata.py for the format/label
notes and docs/agents/adding-a-dataset.md for the general contract."""
from typing import Dict, List, Optional, Tuple

import numpy as np

from IO.loader import BaseSubjectLoader

N_EEG_CHANNELS = 64  # channels 1-64 of dataRest; 65-67 EOG, 68 trigger dropped, see gen_metadata.py


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.file_paths = {  # {'EC': path, 'EO': path}
            cond: self._resolve(entry[cond]) for cond in ("EC", "EO")
        }

    def _load_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.standard_window is None:
            print(f"  [Warning] Subject {self.subject_id}: No standard_window defined, cannot segment data.")
            return None, None

        import scipy.io as sio

        existing = self._existing(list(self.file_paths.values()))
        trial_len = int(self.standard_window * self.sample_freq)
        all_windows, all_labels = [], []

        # 0=EC, 1=EO -- see gen_metadata.py's TARGETS.
        for label, cond in enumerate(("EC", "EO")):
            path = self.file_paths[cond]
            if path not in existing:
                continue
            data = sio.loadmat(path)["dataRest"][:N_EEG_CHANNELS, :]  # (64, T)
            data = data[self.channel_indices, :]  # subset to requested channel order

            n_windows = data.shape[-1] // trial_len
            for i in range(n_windows):
                start = i * trial_len
                all_windows.append(data[:, start:start + trial_len])
                all_labels.append(label)

        if not all_windows:
            return None, None

        eeg_data = np.stack(all_windows)  # (N, C, trial_len)
        labels = np.array(all_labels, dtype=np.int64)
        return eeg_data, labels
