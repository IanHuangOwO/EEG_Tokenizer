import os
from typing import Dict, List

import numpy as np

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    """
    CHB-MIT, binary seizure detection, class-balanced per subject (see
    gen_metadata.py's docstring for the full rationale).

    ictal (1): every non-overlapping standard_window window fully inside a seizure.
    interictal (0): the same number of windows, sampled with a fixed per-subject
    seed from recordings that have no seizure and are not directly before/after a
    seizure recording (in sorted recording order). Only the chosen windows are
    read from disk (raw.get_data with start/stop), never whole hour-long files.

    Channels are picked by bipolar derivation name (metadata 'original_label'),
    case-insensitively; MNE suffixes duplicate names with '-0', '-1', so
    'T8-P8-0' also counts as 'T8-P8'. A derivation a recording lacks is
    zero-filled.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.runs = [(self._resolve(os.path.join(entry['folder'], r['file'])), r['seizures'])
                     for r in entry['runs']]
        ch = self.data_metadata['channels']
        self.wanted = [ch[str(i + 1)]['original_label'].upper() for i in desired_channel_indices]

    def _picks(self, raw):
        names = {n.upper(): n for n in raw.ch_names}
        out = []
        for k, w in enumerate(self.wanted):
            n = names.get(w) or names.get(w + '-0')
            if n is not None:
                out.append((k, n))
        return out

    def _read(self, path, starts, win):
        """-> (len(starts), C, win) float32 windows from one file."""
        import mne
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        if abs(raw.info['sfreq'] - self.sample_freq) > 1e-6:
            raise ValueError(f"sfreq {raw.info['sfreq']} != {self.sample_freq}")
        picks = self._picks(raw)
        out = np.zeros((len(starts), len(self.wanted), win), dtype=np.float32)
        if not picks:
            return out
        for j, s in enumerate(starts):
            out[j, [k for k, _ in picks]] = raw.get_data(picks=[n for _, n in picks], start=s, stop=s + win)
        return out

    def _load_data(self):
        import mne
        fs, win = self.sample_freq, int(self.standard_window * self.sample_freq)
        present = set(self._existing([p for p, _ in self.runs]))
        runs = [(p, sz) for p, sz in self.runs if p in present]
        if not runs:
            return None, None

        # ictal: non-overlapping windows fully inside each seizure
        ictal = {}
        for path, seizures in runs:
            for s, e in seizures:
                starts = list(range(int(s * fs), int(e * fs) - win + 1, win))
                if starts:
                    ictal.setdefault(path, []).extend(starts)
        n_ictal = sum(len(v) for v in ictal.values())
        if n_ictal == 0:
            return None, None

        # interictal candidates: seizure-free runs not adjacent to a seizure run
        # (adjacency judged on the full recording list, downloaded or not)
        seizure_idx = {i for i, (_, sz) in enumerate(self.runs) if sz}
        near = seizure_idx | {i - 1 for i in seizure_idx} | {i + 1 for i in seizure_idx}
        cands = []
        for i, (path, _) in enumerate(self.runs):
            if i in near or path not in present:
                continue
            try:
                n_times = mne.io.read_raw_edf(path, preload=False, verbose=False).n_times
            except Exception as e:
                print(f"  [Warning] {os.path.basename(path)}: {e}")
                continue
            cands += [(path, s) for s in range(0, n_times - win + 1, win)]
        rng = np.random.default_rng(1000 + int(self.subject_id))
        take = rng.choice(len(cands), size=min(n_ictal, len(cands)), replace=False) if cands else []
        inter = {}
        for t in sorted(take):
            path, s = cands[t]
            inter.setdefault(path, []).append(s)

        data, labels = [], []
        for groups, label in ((ictal, 1), (inter, 0)):
            for path, starts in groups.items():
                try:
                    w = self._read(path, starts, win)
                except Exception as e:
                    print(f"  [Warning] {os.path.basename(path)}: {e}")
                    continue
                data.append(w)
                labels += [label] * len(w)
        if not data:
            return None, None
        return np.concatenate(data), np.array(labels, dtype=np.int64)
