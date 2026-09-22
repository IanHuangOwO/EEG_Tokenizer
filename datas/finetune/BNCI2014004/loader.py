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
    recording into [pre_event_seconds + post_event_seconds] windows (config/
    compile.json's global setting, see IO/loader.py's BaseSubjectLoader) around
    the per-class cue event (769/770), concatenated across all 3 runs for the
    subject. Not currently in any active dataset_params (see config/compile.json's
    _comment) -- shares BNCI2014001's paradigm/timing convention, treated the same
    way here, but its own real headroom hasn't been directly measured the way
    docs/model-analysis-checklist.md does for the datasets actually in use.
    """
    _EVENT_TO_LABEL = {'769': 0, '770': 1}

    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.run_paths = [self._resolve(os.path.join(entry['folder'], r)) for r in entry['runs']]

    def _load_data(self):
        if not self.post_event_seconds:
            print(f"  [Warning] Subject {self.subject_id}: post_event_seconds not set, cannot segment data.")
            return None, None

        import mne
        pre_pts = int(self.pre_event_seconds * self.sample_freq)
        post_pts = int(self.post_event_seconds * self.sample_freq)
        all_trials, all_labels, all_valid_ranges = [], [], []

        for run_path in self._existing(self.run_paths):
            try:
                raw = mne.io.read_raw_gdf(run_path, preload=True, verbose=False)
                self._resample_if_needed(raw)

                trials, labels = self._segment_by_annotations(raw, self._EVENT_TO_LABEL, self.channel_indices,
                                                                pre_pts, post_pts)
                if not trials:
                    print(f"  [Warning] {os.path.basename(run_path)}: no cue events (769/770) found.")
                    continue
                all_trials.extend(trials)
                all_labels.extend(labels)
                # _segment_by_annotations overwrites self._last_valid_ranges each call --
                # this loop calls it once per run, so accumulate immediately, same as
                # all_trials/all_labels, or every run but the last silently loses its
                # own validity info to the next run's overwrite.
                all_valid_ranges.extend(self._last_valid_ranges)

            except Exception as e:
                print(f"  [Warning] Error loading {os.path.basename(run_path)}: {e}")

        if not all_trials:
            return None, None
        self._last_valid_ranges = all_valid_ranges
        return np.stack(all_trials), np.array(all_labels)
