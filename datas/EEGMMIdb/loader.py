import os
import numpy as np
from typing import Dict, List

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    """
    Loader for PhysioNet Motor Imagery (BCI2000) dataset.
    Segments EDF files by T0 (rest), T1, and T2 annotations.
    """
    _EVENT_TO_LABEL = {'T0': 0, 'T1': 1, 'T2': 2}

    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.run_paths = [self._resolve(os.path.join(entry['folder'], r)) for r in entry['runs']]

    def _load_data(self):
        if self.standard_window is None:
            print(f"  [Warning] Subject {self.subject_id}: No standard_window defined, cannot segment data.")
            return None, None

        import mne
        trial_len_pts = int(self.standard_window * self.sample_freq)
        all_trials, all_labels = [], []

        for run_path in self._existing(self.run_paths):
            try:
                raw = mne.io.read_raw_edf(run_path, preload=True, verbose=False)
                self._resample_if_needed(raw)

                trials, labels = self._segment_by_annotations(raw, trial_len_pts, self._EVENT_TO_LABEL, self.channel_indices)
                if not trials:
                    print(f"  [Warning] {os.path.basename(run_path)}: no T0/T1/T2 events found.")
                    continue
                all_trials.extend(trials)
                all_labels.extend(labels)

            except Exception as e:
                print(f"  [Warning] Error loading {os.path.basename(run_path)}: {e}")

        if not all_trials:
            return None, None
        return np.stack(all_trials), np.array(all_labels)
