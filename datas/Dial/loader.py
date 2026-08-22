import numpy as np
import scipy.io
from typing import Dict, List

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.signal_path = self._resolve(entry['signals'])
        self.label_path = self._resolve(entry['labels'])

    def _load_data(self):
        if len(self._existing([self.signal_path, self.label_path])) < 2:
            return None, None

        samples = scipy.io.loadmat(self.signal_path)['Data']
        raw_labels = scipy.io.loadmat(self.label_path)['Label'].flatten()

        actual_trials = samples.shape[2]
        num_labels = len(raw_labels)
        if actual_trials != num_labels:
            print(f"  [Warning] Subject {self.subject_id}: {actual_trials} signals but {num_labels} labels. Using minimum.")
        trials_to_use = min(actual_trials, num_labels)

        eeg_data = samples[self.channel_indices, :, :trials_to_use]
        eeg_data = np.transpose(eeg_data, (2, 0, 1))
        # Labels are 1-indexed in Dial, convert to 0-indexed
        labels = (raw_labels[:trials_to_use] - 1).astype(np.int64)
        return eeg_data, labels
