from typing import Dict, List

import numpy as np

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    """
    STEW: each subject has a rest (lo) and a multitask (hi) recording, 150 s x 14
    channels at 128 Hz as whitespace-separated text. Each recording is cut into
    non-overlapping standard_window windows, labelled with its condition
    (0 = rest, 1 = multitask) from metadata.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.files = [(self._resolve(f['file']), int(f['label'])) for f in entry['files']]

    def _load_data(self):
        win = int(self.standard_window * self.sample_freq)
        data, labels = [], []
        present = set(self._existing([p for p, _ in self.files]))
        for path, label in self.files:
            if path not in present:
                continue
            sig = np.loadtxt(path, dtype=np.float32)[:, self.channel_indices].T   # (C, T)
            n_win = sig.shape[1] // win
            if n_win == 0:
                continue
            data.append(sig[:, :n_win * win].reshape(sig.shape[0], n_win, win).transpose(1, 0, 2))
            labels += [label] * n_win
        if not data:
            return None, None
        return np.concatenate(data), np.array(labels, dtype=np.int64)
