from typing import Dict, List

import numpy as np

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    """
    Loader for the Kaggle "Grasp-and-Lift EEG Detection" dataset. Each
    series is a continuous recording annotated with 6 binary event columns
    (HandStart/FirstDigitTouch/BothStartLoadPhase/LiftOff/Replace/
    BothReleased) that overlap in time -- a multi-label sequence-labeling
    task, not one discrete class per trial like every other loader here.
    Rather than force that into a single dense label, this just chops each
    series into fixed-length non-overlapping windows (like BCICIVLoader's
    no-mrk fallback) with dummy label 0 -- fine for self-supervised
    pretraining, real event columns are read but discarded. Do not use for
    supervised finetune/eval of the event-detection task as-is.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.file_pairs = [(self._resolve(f['data']), self._resolve(f['events'])) for f in entry['files']]

    def _load_data(self):
        if self.standard_window is None:
            print(f"  [Warning] Subject {self.subject_id}: No standard_window defined, cannot segment data.")
            return None, None

        import pandas as pd
        trial_len = int(self.standard_window * self.sample_freq)
        all_trials = []

        existing_data_paths = set(self._existing([d for d, _ in self.file_pairs]))
        for data_path, events_path in self.file_pairs:
            if data_path not in existing_data_paths:
                continue
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
