"""SRM Resting-state EEG loader -- see gen_metadata.py for the format/dataset
notes and docs/agents/adding-a-dataset.md for the general contract."""
from typing import Dict, List, Optional, Tuple

import numpy as np

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.file_paths = [self._resolve(f) for f in entry['files']]  # shape C: one file per available session

    def _load_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        import mne

        existing = self._existing(self.file_paths)
        if not existing:
            return None, None

        data = []
        for path in existing:
            try:
                # A handful of this dataset's EDF headers have a malformed recording-start
                # timestamp (e.g. sub-041: "second must be in 0..59") that MNE's EDF reader
                # rejects outright -- a real per-file data-quality issue, not a code bug.
                # Skip just this session rather than losing the whole subject/dataset.
                raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            except Exception as e:
                print(f"  [Warning] {path}: failed to read ({e!r}), skipping this session")
                continue
            sig = raw.get_data(picks=self.channel_indices).astype(np.float32)   # [C, T], native rate
            sig, sf = self._filter_run(sig)                    # whole session, before cutting
            win = int(self.standard_window * sf)
            n_win = sig.shape[1] // win
            if n_win:
                data.append(sig[:, :n_win * win].reshape(sig.shape[0], n_win, win).transpose(1, 0, 2))

        if not data:
            return None, None
        eeg_data = np.concatenate(data)                        # [n_windows, C, win]
        # No events/conditions at all (see gen_metadata.py's dataset_info.notes): dummy class 0.
        return eeg_data, np.zeros(len(eeg_data), dtype=np.int64)
