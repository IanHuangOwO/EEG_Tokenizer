"""BNCI2014009 (P300 Speller) loader -- see gen_metadata.py for the format/
label notes and docs/agents/adding-a-dataset.md for the general contract."""
from typing import Dict, List, Optional, Tuple

import numpy as np

from IO.loader import BaseSubjectLoader

# Hardcoded per-dataset epoch window (P300 scale), NOT the global
# compile.json pre_event_seconds/post_event_seconds -- see gen_metadata.py.
PRE_EVENT_S = 0.2
POST_EVENT_S = 0.8


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.file_path = self._resolve(entry['file'])

    def _load_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._existing([self.file_path]):
            return None, None

        import scipy.io as sio

        d = sio.loadmat(self.file_path, struct_as_record=False, squeeze_me=True)
        pre_pts = int(PRE_EVENT_S * self.sample_freq)
        post_pts = int(POST_EVENT_S * self.sample_freq)

        all_epochs, all_labels = [], []
        for run in d['data']:
            X = run.X[:, self.channel_indices]  # (T, C) -- native channel order already matches metadata
            y = run.y  # (T,) sparse: 0=none, 1=NonTarget, 2=Target
            rising = np.where((y[1:] != 0) & (y[:-1] == 0))[0] + 1
            for idx in rising:
                start, end = idx - pre_pts, idx + post_pts
                if start < 0 or end > X.shape[0]:
                    continue  # boundary event, drop (rare, edge of run)
                all_epochs.append(X[start:end, :].T)  # (C, pre+post)
                all_labels.append(int(y[idx]) - 1)  # 1=NonTarget->0, 2=Target->1

        if not all_epochs:
            return None, None

        eeg_data = np.stack(all_epochs)
        labels = np.array(all_labels, dtype=np.int64)
        return eeg_data, labels
