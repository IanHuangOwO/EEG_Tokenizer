"""SEED loader -- see gen_metadata.py for the format/label notes and docs/agents/adding-a-dataset.md."""
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.files = [self._resolve(p) for p in entry['sessions']]
        self.clip_labels = self.data_metadata['clip_labels']

    def _load_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        trials, labels, sessions = [], [], []
        for s, path in enumerate(self.files):
            if not self._existing([path]):
                continue
            mat = sio.loadmat(path)
            clips = sorted((int(m.group(1)), k) for k in mat if (m := re.search(r'_eeg(\d+)$', k)))
            for c, key in clips:
                sig, sf = self._filter_run(mat[key][self.channel_indices].astype(np.float64))  # whole clip first
                w = int(self.standard_window * sf)
                n = sig.shape[-1] // w
                trials += [sig[:, i * w:(i + 1) * w] for i in range(n)]
                labels += [self.clip_labels[c - 1]] * n
                sessions += [s] * n
        if not trials:
            return None, None
        self._last_sessions = sessions
        return np.stack(trials), np.asarray(labels, dtype=np.int64)
