"""BCMI_MusicEmotion (OpenNeuro ds002721) loader -- see gen_metadata.py for
the format/label notes and docs/agents/adding-a-dataset.md for the general
contract."""
from typing import Dict, List, Optional, Tuple

import numpy as np

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.file_paths = [self._resolve(f) for f in entry['files']]  # shape C: one file per run (sub-06 missing run6)

    def _load_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.standard_window is None:
            print(f"  [Warning] Subject {self.subject_id}: No standard_window defined, cannot segment data.")
            return None, None

        import mne

        existing = self._existing(self.file_paths)
        if not existing:
            return None, None

        trial_len = int(self.standard_window * self.sample_freq)
        all_windows = []
        for path in existing:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            self._resample_if_needed(raw)
            data = raw.get_data(picks=self.channel_indices)  # [C, T]

            n_windows = data.shape[-1] // trial_len
            for i in range(n_windows):
                start = i * trial_len
                all_windows.append(data[:, start:start + trial_len])

        if not all_windows:
            return None, None

        eeg_data = np.stack(all_windows)  # (N, C, trial_len)
        # No label extracted yet (see gen_metadata.py's dataset_info.notes) --
        # every window gets the single placeholder class 0.
        labels = np.zeros(len(all_windows), dtype=np.int64)

        return eeg_data, labels
