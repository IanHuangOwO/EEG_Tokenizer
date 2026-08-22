import os
from typing import Dict, List

import numpy as np

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    """
    Loader for BCI Competition IV Dataset 2b (GDF, 2-class motor imagery,
    bipolar C3/Cz/C4). Each subject has 3 *T (training) session files with
    real labels -- referenced via 'runs' in metadata.json. The 2 *E
    (evaluation) sessions' true labels ship as separate .mat files not
    included in this raw download, so they're excluded. Segments each
    recording into fixed-length windows starting at the per-class cue event
    (769/770), concatenated across all 3 runs for the subject.
    """
    _EVENT_TO_LABEL = {'769': 0, '770': 1}

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
                raw = mne.io.read_raw_gdf(run_path, preload=True, verbose=False)
                self._resample_if_needed(raw)

                trials, labels = self._segment_by_annotations(raw, trial_len_pts, self._EVENT_TO_LABEL, self.channel_indices)
                if not trials:
                    print(f"  [Warning] {os.path.basename(run_path)}: no cue events (769/770) found.")
                    continue
                all_trials.extend(trials)
                all_labels.extend(labels)

            except Exception as e:
                print(f"  [Warning] Error loading {os.path.basename(run_path)}: {e}")

        if not all_trials:
            return None, None
        return np.stack(all_trials), np.array(all_labels)
