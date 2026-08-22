import numpy as np
import scipy.io
from typing import Dict, List

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        self.file_path = self._resolve(self._require_subject(subject_id)['file'])

    def _load_data(self):
        if not self._existing([self.file_path]):
            return None, None

        mat_data = scipy.io.loadmat(self.file_path)
        raw_data = mat_data['data']['EEG'][0, 0]  # (Channels, Time, Blocks, Targets)

        target_blocks = self.data_metadata.get('Number_of_Blocks', 0)
        actual_blocks = raw_data.shape[2]
        if target_blocks > 0 and actual_blocks != target_blocks:
            print(f"  [Warning] Subject {self.subject_id} has {actual_blocks} blocks, expected {target_blocks}.")
        blocks_to_use = min(actual_blocks, target_blocks) if target_blocks > 0 else actual_blocks

        eeg_data = raw_data[self.channel_indices, :, :blocks_to_use, :]
        eeg_data = np.transpose(eeg_data, (3, 2, 0, 1))
        eeg_data = eeg_data.reshape(-1, len(self.channel_indices), raw_data.shape[1])
        labels = np.array([i for i in range(self.num_targets) for _ in range(blocks_to_use)])
        return eeg_data, labels
