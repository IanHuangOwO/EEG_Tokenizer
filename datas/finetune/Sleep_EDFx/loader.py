"""Sleep_EDFx (Sleep Cassette cohort, sleep staging) loader -- see
gen_metadata.py for the format/label notes and docs/agents/adding-a-dataset.md
for the general contract."""
from typing import Dict, List, Optional, Tuple

import numpy as np

from IO.loader import BaseSubjectLoader

EPOCH_S = 30.0
CROP_MARGIN_MIN = 30.0  # standard convention for this dataset, see gen_metadata.py

# Real PSG channel names (order matches gen_metadata.py's CHANNELS) -- picked by
# name, not position, since these are fixed bipolar derivations, not a montage
# self.channel_indices maps onto positionally like other datasets' loaders.
REAL_CHANNEL_NAMES = ["EEG Fpz-Cz", "EEG Pz-Oz"]

# R&K stage description -> Table 14's 5-class scheme. Stages 3+4 merged into N3
# (AASM convention); 'Sleep stage ?' (unscored) and 'Movement time' dropped.
STAGE_LABEL = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.night_paths = [
            (self._resolve(f['psg']), self._resolve(f['hypnogram'])) for f in entry['files']
        ]
        self.pick_names = [REAL_CHANNEL_NAMES[i] for i in self.channel_indices]

    def _load_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        import mne

        epoch_pts = int(EPOCH_S * self.sample_freq)
        all_epochs, all_labels = [], []

        for psg_path, hyp_path in self.night_paths:
            if not self._existing([psg_path, hyp_path]):
                continue

            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
            ann = mne.read_annotations(hyp_path)

            # Sleep-period crop: exclude the long pre-lights-off/post-wake-up W
            # padding, keeping CROP_MARGIN_MIN before the first and after the
            # last non-W annotation (standard convention, see gen_metadata.py).
            non_w = [(o, o + d) for o, d, desc in zip(ann.onset, ann.duration, ann.description)
                     if desc != "Sleep stage W"]
            if not non_w:
                continue  # whole night unscored/all-wake, skip
            sleep_start = max(0.0, min(s for s, _ in non_w) - CROP_MARGIN_MIN * 60)
            sleep_end = min(raw.times[-1], max(e for _, e in non_w) + CROP_MARGIN_MIN * 60)

            data = raw.get_data(picks=self.pick_names)  # (C, T)

            t = sleep_start
            while t + EPOCH_S <= sleep_end:
                # Label from the annotation covering this epoch's start sample
                # (R&K/AASM annotations are themselves 30s-epoch-aligned, so this
                # matches the whole epoch in the near-totality of cases).
                desc = None
                for onset, dur, d in zip(ann.onset, ann.duration, ann.description):
                    if onset <= t < onset + dur:
                        desc = d
                        break
                label = STAGE_LABEL.get(desc)
                if label is not None:
                    start_pt = int(t * self.sample_freq)
                    end_pt = start_pt + epoch_pts
                    if end_pt <= data.shape[-1]:
                        all_epochs.append(data[:, start_pt:end_pt])
                        all_labels.append(label)
                t += EPOCH_S

        if not all_epochs:
            return None, None

        eeg_data = np.stack(all_epochs)
        labels = np.array(all_labels, dtype=np.int64)
        return eeg_data, labels
