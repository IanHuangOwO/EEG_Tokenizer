"""
Copy-paste starting point for a new dataset's loader. Save as
datas/MyDataset/loader.py — the class MUST be named `Loader` (that fixed name,
plus this file's directory location, is the whole discovery contract: no
registry to edit anywhere else). See docs/agents/adding-a-dataset.md for the
full walkthrough (metadata.json schema, array shape contract, wiring steps).
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        # Single-file style (BETA):        self.file_path = self._resolve(entry['file'])
        # Split signal/label style (Dial):  self._resolve(entry['signals']) / self._resolve(entry['labels'])
        # Multi-run style (EEGMMIdb):       [self._resolve(os.path.join(entry['folder'], r)) for r in entry['runs']]
        self.file_path = self._resolve(entry['file'])

    def _load_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._existing([self.file_path]):
            return None, None

        # TODO: read the raw file, subset channels via self.channel_indices,
        # and produce:
        #   eeg_data: np.ndarray (N, C, T) — N trials, C = len(self.channel_indices)
        #   labels:   np.ndarray (N,) int, 0-indexed, dense in [0, self.num_targets)
        # For GDF/EDF event-marker datasets, self._segment_by_annotations(raw,
        # trial_len_pts, code_to_label, self.channel_indices) does the windowing.
        raise NotImplementedError("Fill in Loader._load_data for your dataset.")
