import os
import numpy as np
from typing import Dict, List

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    """
    Loader for PhysioNet Motor Imagery (BCI2000) dataset.
    Segments EDF files by T0 (rest), T1, and T2 annotations.

    Deliberately ignores self.pre_event_seconds/post_event_seconds (config/
    compile.json's global setting) even though it's injected into every loader's
    dataset_params -- pre_pts is hardcoded 0 and post_pts stays standard_window-
    derived below, regardless of what the global compile-time setting is, unlike
    BCICIV2a/BCICIV2b which share _segment_by_annotations. Checked the real
    annotation stream directly (S001R03.edf): T0/T1/T2 events are back-to-back,
    each starting almost exactly trial_len_pts after the previous one (e.g.
    T0@0.0s, T2@4.2s, T0@8.3s, T1@12.5s...). T0 (rest) IS one of the labeled
    classes here, unlike BCICIV2a's unlabeled pre-cue fixation period -- so
    there is no idle gap to borrow from; shifting the window back would pull in
    the tail of the PRECEDING, differently-labeled trial and mix classes into
    one window. cut_event_window's padding only covers running off a
    recording's edge, not this "adjacent real trial" case, so raising the
    global pre_event_seconds would NOT be safe to apply here even though it
    wouldn't error.
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
        all_trials, all_labels, all_valid_ranges = [], [], []

        for run_path in self._existing(self.run_paths):
            try:
                raw = mne.io.read_raw_edf(run_path, preload=True, verbose=False)
                self._resample_if_needed(raw)

                trials, labels = self._segment_by_annotations(raw, self._EVENT_TO_LABEL, self.channel_indices,
                                                                0, trial_len_pts)
                if not trials:
                    print(f"  [Warning] {os.path.basename(run_path)}: no T0/T1/T2 events found.")
                    continue
                all_trials.extend(trials)
                all_labels.extend(labels)
                # _segment_by_annotations overwrites self._last_valid_ranges each call (one
                # per run here) -- accumulate immediately, same as all_trials/all_labels, or
                # get_subject_data ends up with a valid_ranges list shorter than data.shape[0].
                all_valid_ranges.extend(self._last_valid_ranges)

            except Exception as e:
                print(f"  [Warning] Error loading {os.path.basename(run_path)}: {e}")

        if not all_trials:
            return None, None
        self._last_valid_ranges = all_valid_ranges
        return np.stack(all_trials), np.array(all_labels)
