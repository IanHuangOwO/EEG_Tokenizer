"""ERP_Longitudinal (RSVP oddball) loader -- see gen_metadata.py for the
format/trigger notes and docs/agents/adding-a-dataset.md for the general
contract."""
from typing import Dict, List, Optional, Tuple

import numpy as np

from IO.loader import BaseSubjectLoader

N_EEG_CHANNELS = 57  # columns 1-57 of each block; column 58 is the trigger, see gen_metadata.py

# Pretraining only: each continuous block is cut into non-overlapping
# standard_window (5 s) windows with dummy label 0. It used to cut a [-0.2, 0.8] s epoch
# around every stimulus, but RSVP stimuli come several per second, so those epochs
# overlapped (each sample stored several times) and were then spliced back together
# into 5 s pretrain windows -- changed 2026-09-24.
TARGET_TRIGGER = 1      # -> label 1
NONTARGET_TRIGGER = 2   # -> label 0
SESSIONS = ("Day_1", "Day_7", "Day_80", "Day_200")


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.file_path = self._resolve(entry['file'])

    def _load_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._existing([self.file_path]):
            return None, None
        import h5py
        win = int(self.standard_window * self.sample_freq)
        windows = []
        with h5py.File(self.file_path, 'r') as f:
            for session in SESSIONS:
                if session not in f:
                    print(f"  [Warning] Subject {self.subject_id}: missing '{session}', skipping")
                    continue
                for ref in f[session][:].flatten():
                    block = f[ref][:]  # (T, 58) -- v7.3 cell-array dereference
                    eeg = block[:, :N_EEG_CHANNELS][:, self.channel_indices].T.astype(np.float32)  # (C, T)
                    n = eeg.shape[1] // win
                    if n:
                        windows.append(eeg[:, :n * win].reshape(eeg.shape[0], n, win).transpose(1, 0, 2))
        if not windows:
            return None, None
        data = np.concatenate(windows)
        return data, np.zeros(len(data), dtype=np.int64)
