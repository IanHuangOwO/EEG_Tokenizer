"""ERP_Longitudinal (RSVP oddball) loader -- see gen_metadata.py for the
format/trigger notes and docs/agents/adding-a-dataset.md for the general
contract."""
from typing import Dict, List, Optional, Tuple

import numpy as np

from IO.loader import BaseSubjectLoader

N_EEG_CHANNELS = 57  # columns 1-57 of each block; column 58 is the trigger, see gen_metadata.py

# Hardcoded per-dataset epoch window (RSVP/P300 scale, ~1s), NOT the global
# compile.json pre_event_seconds/post_event_seconds (tuned for ~5s
# motor-imagery trials -- see gen_metadata.py's dataset_info.notes).
PRE_EVENT_S = 0.2
POST_EVENT_S = 0.8

TARGET_TRIGGER = 1      # -> label 1
NONTARGET_TRIGGER = 2   # -> label 0
SESSIONS = ("Day_1", "Day_7", "Day_80", "Day_200")


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.file_path = self._resolve(entry['file'])

    def _load_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._existing([self.file_path]):
            return None, None

        import h5py

        pre_pts = int(PRE_EVENT_S * self.sample_freq)
        post_pts = int(POST_EVENT_S * self.sample_freq)

        all_epochs, all_labels = [], []
        with h5py.File(self.file_path, 'r') as f:
            for session in SESSIONS:
                if session not in f:
                    print(f"  [Warning] Subject {self.subject_id}: missing '{session}', skipping")
                    continue
                for ref in f[session][:].flatten():
                    block = f[ref][:]  # (T, 58) -- v7.3 cell-array dereference
                    eeg = block[:, :N_EEG_CHANNELS]           # (T, 57)
                    trig = block[:, N_EEG_CHANNELS]           # (T,)

                    for trigger_val, label in ((TARGET_TRIGGER, 1), (NONTARGET_TRIGGER, 0)):
                        for idx in np.where(trig == trigger_val)[0]:
                            start, end = idx - pre_pts, idx + post_pts
                            if start < 0 or end > eeg.shape[0]:
                                continue  # boundary event, drop (rare, edge of block)
                            all_epochs.append(eeg[start:end, self.channel_indices].T)  # (C, pre+post)
                            all_labels.append(label)

        if not all_epochs:
            return None, None

        eeg_data = np.stack(all_epochs)  # (N, C, pre_pts+post_pts)
        labels = np.array(all_labels, dtype=np.int64)
        return eeg_data, labels
