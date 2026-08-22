from typing import Dict, List

import numpy as np

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    """
    Loader for the Kaggle "Grasp-and-Lift EEG Detection" held-out test split
    (series 9-10 per subject). Unlike GraspAndLift_Train, test.zip ships
    *_data.csv only -- no *_events.csv, no ground-truth labels were ever
    released for this split. Chops each continuous series into fixed-length
    non-overlapping windows with dummy label 0, same as GraspAndLift_Train
    (which discards its real event columns for the same effective result).
    Fine for self-supervised pretraining only.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.data_paths = [self._resolve(f['data']) for f in entry['files']]

    def _load_data(self):
        if self.standard_window is None:
            print(f"  [Warning] Subject {self.subject_id}: No standard_window defined, cannot segment data.")
            return None, None

        import pandas as pd
        trial_len = int(self.standard_window * self.sample_freq)
        all_trials = []

        for data_path in self._existing(self.data_paths):
            df = pd.read_csv(data_path)
            data = df.iloc[:, 1:].values.astype(np.float32)  # (Time, Channels), drop 'id' column

            n_windows = data.shape[0] // trial_len
            for i in range(n_windows):
                start = i * trial_len
                all_trials.append(data[start:start + trial_len, self.channel_indices].T)

        if not all_trials:
            return None, None
        trials = np.stack(all_trials)
        return trials, np.zeros(len(trials), dtype=np.int64)
