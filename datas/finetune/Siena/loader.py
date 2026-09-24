import os
from typing import Dict, List

import numpy as np

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    """
    Siena Scalp EEG Database, for self-supervised pretraining only: each
    recording is chopped into non-overlapping standard_window windows with
    dummy label 0 (like GraspAndLift_Train). Seizure annotations are ignored.

    Channels are picked BY NAME per file (metadata 'original_label', matched
    case-insensitively after stripping the EDF 'EEG ' prefix) because the EDF
    channel order and case differ between recordings. A channel a recording
    doesn't have (PN10 has 20 of the 29) is zero-filled.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.run_paths = [self._resolve(os.path.join(entry['folder'], r)) for r in entry['runs']]
        ch = self.data_metadata['channels']
        self.wanted = [ch[str(i + 1)]['original_label'].upper() for i in desired_channel_indices]

    def _load_data(self):
        import mne
        win = int(self.standard_window * self.sample_freq)
        trials = []
        for path in self._existing(self.run_paths):
            try:
                raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
                self._resample_if_needed(raw)
                by_name = {n.upper().replace('EEG ', '', 1).strip(): n for n in raw.ch_names}
                present = [(k, by_name[w]) for k, w in enumerate(self.wanted) if w in by_name]
                if not present:
                    print(f"  [Warning] {os.path.basename(path)}: none of the wanted channels found")
                    continue
                sig = raw.get_data(picks=[n for _, n in present]).astype(np.float32)   # (C_present, T)
                full = np.zeros((len(self.wanted), sig.shape[1]), dtype=np.float32)
                full[[k for k, _ in present]] = sig
                n_win = full.shape[1] // win
                if n_win:
                    trials.append(full[:, :n_win * win].reshape(len(self.wanted), n_win, win).transpose(1, 0, 2))
            except Exception as e:
                print(f"  [Warning] {os.path.basename(path)}: {e}")
        if not trials:
            return None, None
        data = np.concatenate(trials)
        return data, np.zeros(len(data), dtype=np.int64)
