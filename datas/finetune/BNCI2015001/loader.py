"""BNCI2015001 (motor imagery, right hand vs both feet) loader -- see
gen_metadata.py for the format/label notes and docs/agents/adding-a-dataset.md
for the general contract."""
from typing import Dict, List, Optional, Tuple

import numpy as np

from IO.loader import BaseSubjectLoader

# Hardcoded per-dataset epoch offset/duration (cue-to-end-of-imagery window,
# description.pdf Figure 1), NOT the global compile.json pre_event_seconds/
# post_event_seconds -- see gen_metadata.py.
OFFSET_S = 3.0
DURATION_S = 5.0


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.file_paths = [self._resolve(f) for f in entry['files']]  # one file per session (2 or 3)

    def _load_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        existing = self._existing(self.file_paths)
        if not existing:
            return None, None

        import scipy.io as sio

        offset_pts = int(OFFSET_S * self.sample_freq)
        window_pts = int(DURATION_S * self.sample_freq)

        all_epochs, all_labels = [], []
        for path in existing:
            d = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
            run = d['data']
            X = run.X[:, self.channel_indices]  # (T, C)
            y = run.y  # (n_trials,) 1=right hand, 2=both feet
            trial = np.atleast_1d(run.trial)  # (n_trials,) onset sample indices

            for onset, label in zip(trial, y):
                start, end = onset + offset_pts, onset + offset_pts + window_pts
                if start < 0 or end > X.shape[0]:
                    continue  # boundary trial, drop (rare, edge of session)
                all_epochs.append(X[start:end, :].T)  # (C, window_pts)
                all_labels.append(int(label) - 1)  # 1=right hand->0, 2=both feet->1

        if not all_epochs:
            return None, None

        eeg_data = np.stack(all_epochs)
        labels = np.array(all_labels, dtype=np.int64)
        return eeg_data, labels
