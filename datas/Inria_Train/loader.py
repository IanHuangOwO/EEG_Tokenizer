import os
import numpy as np
from typing import Dict, List

from IO.loader import BaseSubjectLoader


class Loader(BaseSubjectLoader):
    """
    Loader for the Inria BCI Challenge (P300 speller error-related potential,
    ErrP) dataset. Each real subject has 5 recording sessions, listed under
    'signals' as a list so trials from all sessions concatenate under one
    subject key -- keeps the train/val subject split (train_pretrain.py) from
    ever putting two sessions of the same person on opposite sides.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        signals = entry['signals']
        self.signal_paths = [self._resolve(p) for p in signals] if isinstance(signals, list) else [self._resolve(signals)]
        self.label_path = self._resolve(entry['labels'])

    def _load_data(self):
        import pandas as pd
        labels_df = pd.read_csv(self.label_path)
        alt_path = os.path.join(os.path.dirname(self.label_path), 'SampleSubmission.csv')
        alt_df = pd.read_csv(alt_path) if os.path.exists(alt_path) else None

        all_trials, all_labels = [], []
        for signal_path in self._existing(self.signal_paths):
            df = pd.read_csv(signal_path)
            data = df.iloc[:, 1:-1].values    # (Time, Channels)
            markers = df.iloc[:, -1].values   # (Time,)

            trig_indices = np.where(markers == 1)[0]

            sub_sess = os.path.basename(signal_path).replace('Data_', '').replace('.csv', '')
            sub_labels = labels_df[labels_df['IdFeedBack'].str.contains(sub_sess)]['Prediction'].values
            if len(sub_labels) == 0 and alt_df is not None:
                sub_labels = alt_df[alt_df['IdFeedBack'].str.contains(sub_sess)]['Prediction'].values

            if self.standard_window:
                trial_len = int(self.standard_window * self.sample_freq)
            elif len(trig_indices) > 1:
                trial_len = int(np.median(np.diff(trig_indices)))
            else:
                continue

            for i, idx in enumerate(trig_indices):
                if i < len(sub_labels):
                    end = idx + trial_len
                    if end <= data.shape[0]:
                        all_trials.append(data[idx:end, self.channel_indices].T)
                        all_labels.append(int(sub_labels[i]))

        if not all_trials:
            return None, None
        return np.stack(all_trials), np.array(all_labels)
