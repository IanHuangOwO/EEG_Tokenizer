from typing import Dict, List

import numpy as np

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    """
    UCSD Parkinson's resting state (BIDS .bdf, 512 Hz): each recording (hc, or PD
    off/on medication) is average-referenced over the 32 scalp channels, then cut into
    non-overlapping standard_window windows labelled with its session from metadata.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.runs = [(self._resolve(r['file']), int(r['label'])) for r in entry['runs']]
        ch = self.data_metadata['channels']
        self.scalp = [ch[str(i + 1)]['label'] for i in range(int(ch['count']))]

    def _load_data(self):
        import mne
        win = int(self.standard_window * self.sample_freq)
        data, labels = [], []
        present = set(self._existing([p for p, _ in self.runs]))
        for path, label in self.runs:
            if path not in present:
                continue
            try:
                raw = mne.io.read_raw_bdf(path, preload=False, verbose=False)
                sig = raw.get_data(picks=self.scalp)                             # (32, T)
            except Exception as e:
                print(f"  [Warning] {path}: {e}")
                continue
            sig = (sig - sig.mean(axis=0, keepdims=True))[self.channel_indices].astype(np.float32)
            n_win = sig.shape[1] // win
            if n_win == 0:
                continue
            data.append(sig[:, :n_win * win].reshape(sig.shape[0], n_win, win).transpose(1, 0, 2))
            labels += [label] * n_win
        if not data:
            return None, None
        return np.concatenate(data), np.array(labels, dtype=np.int64)
