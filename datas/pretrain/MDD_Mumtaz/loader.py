from typing import Dict, List

import numpy as np

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    """
    Mumtaz MDD vs healthy controls (BIDS): every recording (eyes closed / eyes open /
    P300) is cut into non-overlapping standard_window windows, labelled with the
    subject's group from metadata (0 healthy, 1 MDD). Channels picked by EDF name
    (metadata 'original_label', e.g. 'EEG T3-LE').
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.files = [self._resolve(f) for f in entry['files']]
        self.label = int(entry['label'])
        ch = self.data_metadata['channels']
        self.pick_names = [ch[str(i + 1)]['original_label'] for i in desired_channel_indices]

    def _load_data(self):
        import mne
        win = int(self.standard_window * self.sample_freq)
        data = []
        for path in self._existing(self.files):
            try:
                raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
                sig = raw.get_data(picks=self.pick_names).astype(np.float32)   # (C, T)
            except Exception as e:
                print(f"  [Warning] {path}: {e}")
                continue
            n_win = sig.shape[1] // win
            if n_win:
                data.append(sig[:, :n_win * win].reshape(sig.shape[0], n_win, win).transpose(1, 0, 2))
        if not data:
            return None, None
        data = np.concatenate(data)
        return data, np.full(len(data), self.label, dtype=np.int64)
