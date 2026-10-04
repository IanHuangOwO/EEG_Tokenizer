import csv
from typing import Dict, List

import numpy as np

from IO.loader import BaseSubjectLoader, _windows_with_tail


class Loader(BaseSubjectLoader):
    """
    BCI Competition 2020 track 3 (BIDS): each run is the original epoched data laid
    back to back in one EDF. One window per trial: events.tsv 'sample' onset,
    standard_window (795 samples) long, label = events.tsv 'value' - 1. All runs
    (train/validation/test) are used. Channels are picked by name.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.runs = [(self._resolve(r['edf']), self._resolve(r['events'])) for r in entry['runs']]
        ch = self.data_metadata['channels']
        self.pick_names = [ch[str(i + 1)]['label'] for i in desired_channel_indices]

    def _load_data(self):
        import mne
        data, labels, ranges = [], [], []
        present = set(self._existing([e for e, _ in self.runs]))
        for edf, events in self.runs:
            if edf not in present:
                continue
            try:
                raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
                sig = raw.get_data(picks=self.pick_names).astype(np.float32)   # (C, T)
                with open(events, encoding='utf-8-sig') as f:
                    rows = list(csv.DictReader(f, delimiter='\t'))
                if self.continuous_seconds:
                    # event-free compile: the run is epochs laid back to back, so each epoch is its own recording --
                    # filtered alone and kept as one zero-padded window, never joined to its neighbours (user, 2026-10-03)
                    onsets = sorted(int(r['sample']) for r in rows) + [sig.shape[1]]
                    for a, b in zip(onsets[:-1], onsets[1:]):
                        piece, sf_p = self._filter_run(sig[:, a:b])
                        for w, r in _windows_with_tail(piece, int(round(self.continuous_seconds * sf_p))):
                            data.append(w); labels.append(0); ranges.append(r)
                    continue
                sig, sf = self._filter_run(sig)                                  # whole run, before cutting
                win, scale = int(round(self.standard_window * sf)), sf / self.sample_freq
                for r in rows:
                    s = int(round(int(r['sample']) * scale))
                    if s + win <= sig.shape[1]:
                        data.append(sig[:, s:s + win])
                        labels.append(int(r['value']) - 1)
            except Exception as e:
                print(f"  [Warning] {edf}: {e}")
        if not data:
            return None, None
        if ranges:
            self._last_valid_ranges = ranges        # real content of each padded epoch window (compiled rate)
        return np.stack(data), np.array(labels, dtype=np.int64)
