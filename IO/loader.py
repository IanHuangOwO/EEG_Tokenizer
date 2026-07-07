import os
import numpy as np
import scipy.io
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any


class BaseSubjectLoader(ABC):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        self.config = config
        self.subject_id = subject_id
        self.channel_indices = desired_channel_indices

        self.data_metadata = config['data_metadata']
        self.dataset_params = config['dataset_params']
        self.data_root = self.dataset_params['dataset_path']

        targets = self.data_metadata['targets']
        self.num_targets = targets.get('count', 2)
        self.target_type = targets.get('type', 'motor_imagery')

        acquisition = self.data_metadata['acquisition']
        self.sample_freq = acquisition['sample_frequency']
        self.standard_window = acquisition.get('window_size_seconds', None)
        self.target_points = int(self.standard_window * self.sample_freq) if self.standard_window else None

    def _get_standard_coords(self, ch_name: str) -> Optional[np.ndarray]:
        try:
            import mne
            if not hasattr(self, '_std_montage'):
                self._std_montage = mne.channels.make_standard_montage('standard_1020')
            if not hasattr(self, '_std_positions'):
                self._std_positions = self._std_montage.get_positions()['ch_pos']
            keys = {k.upper(): k for k in self._std_positions.keys()}
            if ch_name.upper() in keys:
                return self._std_positions[keys[ch_name.upper()]]
        except Exception:
            pass
        return None

    def _load_coords_from_metadata(self) -> np.ndarray:
        """
        Default coord loader: tries MNE standard_1020 first, then falls back to
        polar coords from metadata. Returns zeros for unknown channels.
        """
        channel_config = self.data_metadata.get('channels', {})
        sorted_keys = sorted(
            [k for k in channel_config.keys() if isinstance(k, str) and k.isdigit()],
            key=lambda k: int(k)
        )
        coords_list = []
        for idx in self.channel_indices:
            ch_info = channel_config.get(sorted_keys[idx]) if idx < len(sorted_keys) else None
            label = ch_info.get('label', 'Unknown') if isinstance(ch_info, dict) else 'Unknown'

            mne_coords = self._get_standard_coords(label)
            if mne_coords is not None:
                coords_list.append(mne_coords)
                continue

            if isinstance(ch_info, dict) and 'coordinates' in ch_info:
                v = ch_info['coordinates']
                theta = np.deg2rad(v.get('polar_angle_deg', 0))
                r = v.get('polar_radius', 0)
                coords_list.append([r * np.sin(theta), r * np.cos(theta), 0.0])
            else:
                coords_list.append([0.0, 0.0, 0.0])

        return np.array(coords_list, dtype=np.float32)

    @abstractmethod
    def _load_coords(self) -> np.ndarray:
        pass

    @abstractmethod
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_subject_data(self) -> Optional[Dict[str, Any]]:
        data, labels = self._load_data()
        if data is None:
            return None
        return {
            'data': data.astype(np.float32),
            'labels': labels.astype(np.int64),
            'coords': self._load_coords().astype(np.float32),
            'subject_id': self.subject_id
        }


class BETALoader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        subject_str = str(subject_id)
        structure = config['data_structure']
        if subject_str not in structure:
            raise ValueError(f"Subject {subject_id} not found in data structure.")
        self.file_path = os.path.join(self.data_root, structure[subject_str]['file'].lstrip('./'))

    def _load_coords(self) -> np.ndarray:
        return self._load_coords_from_metadata()

    def _load_data(self):
        if not os.path.exists(self.file_path):
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


class DialLoader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        subject_str = str(subject_id)
        structure = config['data_structure']
        if subject_str not in structure:
            raise ValueError(f"Subject {subject_id} not found in data structure.")
        self.signal_path = os.path.join(self.data_root, structure[subject_str]['signals'].lstrip('./'))
        self.label_path = os.path.join(self.data_root, structure[subject_str]['labels'].lstrip('./'))

    def _load_coords(self) -> np.ndarray:
        return self._load_coords_from_metadata()

    def _load_data(self):
        if not os.path.exists(self.signal_path) or not os.path.exists(self.label_path):
            return None, None

        samples = scipy.io.loadmat(self.signal_path)['Data']
        raw_labels = scipy.io.loadmat(self.label_path)['Label'].flatten()

        actual_trials = samples.shape[2]
        num_labels = len(raw_labels)
        if actual_trials != num_labels:
            print(f"  [Warning] Subject {self.subject_id}: {actual_trials} signals but {num_labels} labels. Using minimum.")
        trials_to_use = min(actual_trials, num_labels)

        eeg_data = samples[self.channel_indices, :, :trials_to_use]
        eeg_data = np.transpose(eeg_data, (2, 0, 1))
        # Labels are 1-indexed in Dial, convert to 0-indexed
        labels = (raw_labels[:trials_to_use] - 1).astype(np.int64)
        return eeg_data, labels


class BCICIVLoader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        subject_str = str(subject_id)
        structure = config['data_structure']
        self.file_path = os.path.join(self.data_root, structure[subject_str]['file'].lstrip('./'))

    def _load_coords(self) -> np.ndarray:
        return self._load_coords_from_metadata()

    def _load_data(self):
        if not os.path.exists(self.file_path):
            return None, None
        mat = scipy.io.loadmat(self.file_path)
        cnt = mat['cnt'].astype(np.float32)  # (Time, Channels)

        if self.standard_window is None:
            print(f"  [Warning] Subject {self.subject_id}: No standard_window defined, cannot segment data.")
            return None, None

        trial_len = int(self.standard_window * self.sample_freq)

        if 'mrk' not in mat:
            pos = np.arange(0, cnt.shape[0] - trial_len, trial_len)
            y = np.zeros(len(pos))
        else:
            mrk = mat['mrk'][0, 0]
            pos = mrk['pos'][0]
            y = mrk['y'][0]

        trials, raw_labels = [], []
        for p, label in zip(pos, y):
            start = int(p)
            end = start + trial_len
            if end <= cnt.shape[0]:
                trials.append(cnt[start:end, self.channel_indices].T)
                raw_labels.append(int(label))

        if not trials:
            return None, None

        # Remap arbitrary label encodings (e.g. bipolar {-1, +1}, 1-indexed {1, 2, ...})
        # to dense 0-indexed class ids.
        raw_labels = np.array(raw_labels)
        uniq = np.unique(raw_labels)
        remap = {v: i for i, v in enumerate(uniq)}
        valid_y = np.array([remap[v] for v in raw_labels])
        return np.stack(trials), valid_y


class InriaLoader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        subject_str = str(subject_id)
        structure = config['data_structure']
        self.signal_path = os.path.join(self.data_root, structure[subject_str]['signals'].lstrip('./'))
        self.label_path = os.path.join(self.data_root, structure[subject_str]['labels'].lstrip('./'))

    def _load_coords(self) -> np.ndarray:
        return self._load_coords_from_metadata()

    def _load_data(self):
        if not os.path.exists(self.signal_path):
            return None, None
        import pandas as pd
        df = pd.read_csv(self.signal_path)
        data = df.iloc[:, 1:-1].values    # (Time, Channels)
        markers = df.iloc[:, -1].values   # (Time,)

        trig_indices = np.where(markers == 1)[0]

        sub_sess = os.path.basename(self.signal_path).replace('Data_', '').replace('.csv', '')
        labels_df = pd.read_csv(self.label_path)
        sub_labels = labels_df[labels_df['IdFeedBack'].str.contains(sub_sess)]['Prediction'].values

        if len(sub_labels) == 0:
            alt_path = os.path.join(os.path.dirname(self.label_path), 'SampleSubmission.csv')
            if os.path.exists(alt_path):
                alt_df = pd.read_csv(alt_path)
                sub_labels = alt_df[alt_df['IdFeedBack'].str.contains(sub_sess)]['Prediction'].values

        if self.standard_window:
            trial_len = int(self.standard_window * self.sample_freq)
        elif len(trig_indices) > 1:
            trial_len = int(np.median(np.diff(trig_indices)))
        else:
            return None, None

        trials, final_y = [], []
        for i, idx in enumerate(trig_indices):
            if i < len(sub_labels):
                end = idx + trial_len
                if end <= data.shape[0]:
                    trials.append(data[idx:end, self.channel_indices].T)
                    final_y.append(int(sub_labels[i]))

        if not trials:
            return None, None
        return np.stack(trials), np.array(final_y)


class EEGMMIdbLoader(BaseSubjectLoader):
    """
    Loader for PhysioNet Motor Imagery (BCI2000) dataset.
    Segments EDF files by T0 (rest), T1, and T2 annotations.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        subject_str = str(subject_id)
        structure = config['data_structure']
        if subject_str not in structure:
            raise ValueError(f"Subject {subject_id} not found in EEGMMIdb data structure.")
        sub_info = structure[subject_str]
        self.run_paths = [os.path.join(self.data_root, sub_info['folder'], r) for r in sub_info['runs']]

    def _load_coords(self) -> np.ndarray:
        return self._load_coords_from_metadata()

    def _load_data(self):
        if self.standard_window is None:
            print(f"  [Warning] Subject {self.subject_id}: No standard_window defined, cannot segment data.")
            return None, None

        import mne
        trial_len_pts = int(self.standard_window * self.sample_freq)
        all_trials, all_labels = [], []

        for run_path in self.run_paths:
            if not os.path.exists(run_path):
                continue
            try:
                raw = mne.io.read_raw_edf(run_path, preload=True, verbose=False)
                if raw.info['sfreq'] != self.sample_freq:
                    raw.resample(self.sample_freq)

                events, event_id = mne.events_from_annotations(raw, verbose=False)
                if events.size == 0 or not event_id:
                    continue

                t0_id = event_id.get('T0')
                t1_id = event_id.get('T1')
                t2_id = event_id.get('T2')

                if t0_id is None and t1_id is None and t2_id is None:
                    print(f"  [Warning] {os.path.basename(run_path)}: No T0/T1/T2 events. Found: {list(event_id.keys())}")
                    continue

                data_np = raw.get_data(picks=self.channel_indices)
                label_map = {t0_id: 0, t1_id: 1, t2_id: 2}

                for event in events:
                    start_pts = event[0]
                    label = label_map.get(event[2], -1)
                    if label != -1:
                        end_pts = start_pts + trial_len_pts
                        if end_pts <= data_np.shape[1]:
                            all_trials.append(data_np[:, start_pts:end_pts])
                            all_labels.append(label)

            except Exception as e:
                print(f"  [Warning] Error loading {os.path.basename(run_path)}: {e}")

        if not all_trials:
            return None, None
        return np.stack(all_trials), np.array(all_labels)
