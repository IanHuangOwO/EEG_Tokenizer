from typing import Dict, List

import numpy as np

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    """
    Helsinki neonatal EEG, pretraining only: one EDF per neonate, cut into
    non-overlapping standard_window windows with dummy label 0. Channels picked by
    EDF name (metadata 'original_label', e.g. 'EEG T3-Ref'), case-insensitively.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        self.file = self._resolve(self._require_subject(subject_id)['file'])
        ch = self.data_metadata['channels']
        self.pick_names = [ch[str(i + 1)]['original_label'] for i in desired_channel_indices]

    def _load_data(self):
        import mne
        win = int(self.standard_window * self.sample_freq)
        raw = mne.io.read_raw_edf(self.file, preload=False, verbose=False)
        self._resample_if_needed(raw)
        # 16 of the 79 files spell the reference '-REF' instead of '-Ref'.
        by_upper = {n.upper(): n for n in raw.ch_names}
        picks = [by_upper[n.upper()] for n in self.pick_names]
        sig = raw.get_data(picks=picks).astype(np.float32)   # (C, T)
        n_win = sig.shape[1] // win
        if n_win == 0:
            return None, None
        data = sig[:, :n_win * win].reshape(sig.shape[0], n_win, win).transpose(1, 0, 2)
        return data, np.zeros(n_win, dtype=np.int64)
