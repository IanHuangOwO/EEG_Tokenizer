import numpy as np
import scipy.io
from typing import Dict, List

from IO.loader import BaseSubjectLoader
from IO.preprocessing import cut_event_window


class Loader(BaseSubjectLoader):
    # Real marker-triggered trials (the 'mrk' branch below) measured with >=4.0s
    # headroom before every trial in the dataset (evenly-spaced, back-to-back at
    # trial_len + headroom) -- see docs/model-analysis-checklist.md. self.pre_event_
    # seconds/post_event_seconds (config/compile.json's global setting) only apply
    # there; the no-marker fallback branch has no real event to be "pre"/"post" of,
    # its trial_len stays standard_window-derived, always fully valid by
    # construction (np.arange already stops short of running past the recording).

    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        self.file_path = self._resolve(self._require_subject(subject_id)['file'])

    def _load_data(self):
        if not self._existing([self.file_path]):
            return None, None
        mat = scipy.io.loadmat(self.file_path)
        cnt = mat['cnt'].astype(np.float32)[:, self.channel_indices].T  # (C, T)

        if self.standard_window is None:
            print(f"  [Warning] Subject {self.subject_id}: No standard_window defined, cannot segment data.")
            return None, None

        trial_len = int(self.standard_window * self.sample_freq)

        if 'mrk' not in mat:
            pos = np.arange(0, cnt.shape[-1] - trial_len, trial_len)
            y = np.zeros(len(pos))
            trials = [cnt[:, p:p + trial_len] for p in pos]
            valid_ranges = [(0, trial_len)] * len(pos)
            raw_labels = list(y)
        else:
            mrk = mat['mrk'][0, 0]
            pos = mrk['pos'][0]
            y = mrk['y'][0]
            pre_pts = int(self.pre_event_seconds * self.sample_freq)
            post_pts = int(self.post_event_seconds * self.sample_freq) or trial_len
            trials, raw_labels, valid_ranges = [], [], []
            for p, label in zip(pos, y):
                window, vs, ve = cut_event_window(cnt, int(p), pre_pts, post_pts)
                if ve <= vs:
                    continue
                trials.append(window)
                raw_labels.append(int(label))
                valid_ranges.append((vs, ve))

        if not trials:
            return None, None

        # Remap arbitrary label encodings (e.g. bipolar {-1, +1}, 1-indexed {1, 2, ...})
        # to dense 0-indexed class ids.
        raw_labels = np.array(raw_labels)
        uniq = np.unique(raw_labels)
        remap = {v: i for i, v in enumerate(uniq)}
        valid_y = np.array([remap[v] for v in raw_labels])
        self._last_valid_ranges = valid_ranges
        return np.stack(trials), valid_y
