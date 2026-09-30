"""SEED-V loader -- see gen_metadata.py for the format/label notes and docs/agents/adding-a-dataset.md."""
from typing import Dict, List, Optional, Tuple

import numpy as np

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.files = [self._resolve(p) for p in entry['sessions']]
        self.session_clips = self.data_metadata['session_clips']
        ch = self.data_metadata['channels']
        self.picks = [ch[str(i + 1)]['label'] for i in self.channel_indices]  # by name: the .cnt adds M1/M2/VEO/HEO

    def _load_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        import mne
        trials, labels, sessions = [], [], []
        for s, path in enumerate(self.files):
            if not self._existing([path]):
                continue
            # every file is 32-bit; 7_1's header has a negative sample count, so MNE cannot infer it
            raw = mne.io.read_raw_cnt(path, data_format='int32', preload=True, verbose='error')
            self._resample_if_needed(raw)
            sig, sf = self._filter_run(raw.get_data(picks=self.picks))  # whole recording, before cutting
            w = int(self.standard_window * sf)
            for start, end, label in self.session_clips[s]:
                clip = sig[:, int(start * sf):int(end * sf)]
                n = clip.shape[-1] // w
                trials += [clip[:, i * w:(i + 1) * w] for i in range(n)]
                labels += [label] * n
                sessions += [s] * n
        if not trials:
            return None, None
        self._last_sessions = sessions
        return np.stack(trials), np.asarray(labels, dtype=np.int64)
