"""SRM Resting-state EEG loader -- see gen_metadata.py for the format/dataset
notes and docs/agents/adding-a-dataset.md for the general contract."""
from typing import Dict, List, Optional, Tuple

import numpy as np

from IO.loader import BaseSubjectLoader


def _read_edf(path):
    """mne.io.read_raw_edf, tolerating a malformed recording-start time in the EDF header (sub-041: starttime
    '10.55.60', 60 s is out of range and MNE refuses the file). The start time is never used here, so on that error
    the file is read from a temporary copy whose date/time fields (header bytes 168-183, 'dd.mm.yy' 'hh.mm.ss') are
    clamped to valid values. The raw file is not modified."""
    import os
    import tempfile
    import mne
    try:
        return mne.io.read_raw_edf(path, preload=True, verbose=False)
    except ValueError as e:
        if 'must be in' not in str(e):
            raise
    with open(path, 'rb') as f:
        buf = bytearray(f.read())
    limits = ((1, 31), (1, 12), (0, 99), (0, 23), (0, 59), (0, 59))  # dd mm yy hh mm ss
    parts = buf[168:176].decode('ascii').split('.') + buf[176:184].decode('ascii').split('.')
    fixed = [f'{min(max(int(v), lo), hi):02d}' for v, (lo, hi) in zip(parts, limits)]
    buf[168:184] = ('.'.join(fixed[:3]) + '.'.join(fixed[3:])).encode('ascii')
    fd, tmp = tempfile.mkstemp(suffix='.edf')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(buf)
        return mne.io.read_raw_edf(tmp, preload=True, verbose=False)
    finally:
        os.remove(tmp)


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
                raw = _read_edf(path)
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
