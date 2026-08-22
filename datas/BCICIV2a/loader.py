from typing import Dict, List

import numpy as np

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    """
    Loader for BCI Competition IV Dataset 2a (GDF, 4-class motor imagery).
    Only *T (training) session files are referenced in metadata.json -- the
    *E (evaluation) session's true labels ship as separate .mat files not
    included in this raw download, so E trials have no usable label.
    Segments each recording into fixed-length windows starting at the
    per-class cue event (769/770/771/772).
    """
    _EVENT_TO_LABEL = {'769': 0, '770': 1, '771': 2, '772': 3}

    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        self.file_path = self._resolve(self._require_subject(subject_id)['file'])

    def _load_data(self):
        if not self._existing([self.file_path]):
            return None, None
        if self.standard_window is None:
            print(f"  [Warning] Subject {self.subject_id}: No standard_window defined, cannot segment data.")
            return None, None

        import mne
        raw = mne.io.read_raw_gdf(self.file_path, preload=True, verbose=False)
        self._resample_if_needed(raw)

        trial_len_pts = int(self.standard_window * self.sample_freq)
        trials, labels = self._segment_by_annotations(raw, trial_len_pts, self._EVENT_TO_LABEL, self.channel_indices)
        if not trials:
            print(f"  [Warning] Subject {self.subject_id}: no cue events (769-772) found.")
            return None, None
        return np.stack(trials), np.array(labels)
