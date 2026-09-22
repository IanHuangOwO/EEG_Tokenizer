"""DEAP loader -- see gen_metadata.py for the format/label-scheme notes and
docs/agents/adding-a-dataset.md for the general contract."""
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from IO.loader import BaseSubjectLoader

N_EEG_CHANNELS = 32  # first 32 of the 40 stored channels; the rest are peripheral, see gen_metadata.py


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.file_path = self._resolve(entry['file'])

    def _load_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._existing([self.file_path]):
            return None, None

        with open(self.file_path, 'rb') as f:
            # DEAP's .dat files were pickled under Python 2 -- 'latin1' is required to
            # unpickle the numpy arrays under Python 3 without raising UnicodeDecodeError.
            d = pickle.load(f, encoding='latin1')

        data = d['data'][:, :N_EEG_CHANNELS, :]                     # [40, 32, 8064] -> drop peripheral
        eeg_data = data[:, self.channel_indices, :]                 # subset to the requested channel order

        ratings = d['labels']                                       # [40, 4]: valence, arousal, dominance, liking
        valence_hi = ratings[:, 0] >= 5.0
        arousal_hi = ratings[:, 1] >= 5.0
        # 0=LVLA 1=LVHA 2=HVLA 3=HVHA (see gen_metadata.py's TARGETS)
        labels = (valence_hi.astype(np.int64) * 2) + arousal_hi.astype(np.int64)

        return eeg_data, labels
