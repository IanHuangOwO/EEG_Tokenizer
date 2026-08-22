import numpy as np
import scipy.io
from typing import Dict, List

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        self.file_path = self._resolve(self._require_subject(subject_id)['file'])

    def _load_data(self):
        if not self._existing([self.file_path]):
            return None, None
        mat = scipy.io.loadmat(self.file_path)
        cnt = mat['cnt'].astype(np.float32)  # (Time, Channels)

        if self.standard_window is None:
            print(f"  [Warning] Subject {self.subject_id}: No standard_window defined, cannot segment data.")
            return None, None

        trial_len = int(self.standard_window * self.sample_freq)

        if 'mrk' not in mat:
            pos = np.arange(0, cnt.shape[0] - trial_len, trial_len)
            y = np.zeros(len(pos))
        else:
            mrk = mat['mrk'][0, 0]
            pos = mrk['pos'][0]
            y = mrk['y'][0]

        trials, raw_labels = [], []
        for p, label in zip(pos, y):
            start = int(p)
            end = start + trial_len
            if end <= cnt.shape[0]:
                trials.append(cnt[start:end, self.channel_indices].T)
                raw_labels.append(int(label))

        if not trials:
            return None, None

        # Remap arbitrary label encodings (e.g. bipolar {-1, +1}, 1-indexed {1, 2, ...})
        # to dense 0-indexed class ids.
        raw_labels = np.array(raw_labels)
        uniq = np.unique(raw_labels)
        remap = {v: i for i, v in enumerate(uniq)}
        valid_y = np.array([remap[v] for v in raw_labels])
        return np.stack(trials), valid_y
