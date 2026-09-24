from typing import Dict, List

import numpy as np

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    """
    AAD KULeuven: one MATLAB v7.3 file per subject, 'trials' holds 20 trial structs,
    each RawData.EegData (64 x T, 128 Hz, uV). Each trial is cut into non-overlapping
    standard_window windows, label = attended ear (0 left, 1 right).
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        self.file = self._resolve(self._require_subject(subject_id)['file'])

    def _load_data(self):
        import h5py
        win = int(self.standard_window * self.sample_freq)
        data, labels = [], []
        with h5py.File(self.file, 'r') as f:
            for ref in f['trials'][()].ravel():
                g = f[ref]
                sig = g['RawData/EegData'][()][self.channel_indices].astype(np.float32)   # (C, T)
                label = 0 if chr(int(g['attended_ear'][()].item())) == 'L' else 1
                n_win = sig.shape[1] // win
                if n_win == 0:
                    continue
                data.append(sig[:, :n_win * win].reshape(sig.shape[0], n_win, win).transpose(1, 0, 2))
                labels += [label] * n_win
        if not data:
            return None, None
        return np.concatenate(data), np.array(labels, dtype=np.int64)
