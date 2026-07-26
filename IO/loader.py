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

    def _require_subject(self, subject_id) -> Dict:
        """Looks up this subject's data_structure entry, raising a clear error if missing."""
        subject_str = str(subject_id)
        structure = self.config['data_structure']
        if subject_str not in structure:
            raise ValueError(f"Subject {subject_id} not found in data structure.")
        return structure[subject_str]

    def _resolve(self, rel_path: str) -> str:
        """Joins a data_structure-relative path onto this dataset's root."""
        return os.path.join(self.data_root, rel_path.lstrip('./'))

    def _resample_if_needed(self, raw) -> None:
        if raw.info['sfreq'] != self.sample_freq:
            raw.resample(self.sample_freq)

    @staticmethod
    def _existing(paths: List[str]) -> List[str]:
        """Filters to paths that exist, printing a warning for each one that doesn't."""
        for p in paths:
            if not os.path.exists(p):
                print(f"  [Warning] Missing file: {p}")
        return [p for p in paths if os.path.exists(p)]

    @staticmethod
    def _segment_by_annotations(
        raw, trial_len_pts: int, code_to_label: Dict[str, int], channel_indices: List[int]
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Cuts fixed-length [C, trial_len_pts] windows starting at each MNE
        annotation whose description is a key of code_to_label. Shared by the
        GDF/EDF event-marker loaders (EEGMMIdb, BCICIV2a, BCICIV2b).
        """
        import mne
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        label_for_code = {event_id[k]: v for k, v in code_to_label.items() if k in event_id}
        if not label_for_code:
            return [], []

        data_np = raw.get_data(picks=channel_indices)
        trials, labels = [], []
        for start_pts, _, code in events:
            label = label_for_code.get(code, -1)
            if label == -1:
                continue
            end_pts = start_pts + trial_len_pts
            if end_pts <= data_np.shape[1]:
                trials.append(data_np[:, start_pts:end_pts])
                labels.append(label)
        return trials, labels

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

    def _load_coords(self) -> np.ndarray:
        """Default coord source. Override only when the loader has real digitized coords."""
        return self._load_coords_from_metadata()

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


class DialLoader(BaseSubjectLoader):
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.signal_path = self._resolve(entry['signals'])
        self.label_path = self._resolve(entry['labels'])

    def _load_data(self):
        if len(self._existing([self.signal_path, self.label_path])) < 2:
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
        self.file_path = self._resolve(self._require_subject(subject_id)['file'])

    def _load_data(self):
        if not self._existing([self.file_path]):
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


class EEGMMIdbLoader(BaseSubjectLoader):
    """
    Loader for PhysioNet Motor Imagery (BCI2000) dataset.
    Segments EDF files by T0 (rest), T1, and T2 annotations.
    """
    _EVENT_TO_LABEL = {'T0': 0, 'T1': 1, 'T2': 2}

    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.run_paths = [self._resolve(os.path.join(entry['folder'], r)) for r in entry['runs']]

    def _load_data(self):
        if self.standard_window is None:
            print(f"  [Warning] Subject {self.subject_id}: No standard_window defined, cannot segment data.")
            return None, None

        import mne
        trial_len_pts = int(self.standard_window * self.sample_freq)
        all_trials, all_labels = [], []

        for run_path in self._existing(self.run_paths):
            try:
                raw = mne.io.read_raw_edf(run_path, preload=True, verbose=False)
                self._resample_if_needed(raw)

                trials, labels = self._segment_by_annotations(raw, trial_len_pts, self._EVENT_TO_LABEL, self.channel_indices)
                if not trials:
                    print(f"  [Warning] {os.path.basename(run_path)}: no T0/T1/T2 events found.")
                    continue
                all_trials.extend(trials)
                all_labels.extend(labels)

            except Exception as e:
                print(f"  [Warning] Error loading {os.path.basename(run_path)}: {e}")

        if not all_trials:
            return None, None
        return np.stack(all_trials), np.array(all_labels)


class BCICIV2aLoader(BaseSubjectLoader):
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


class BCICIV2bLoader(BaseSubjectLoader):
    """
    Loader for BCI Competition IV Dataset 2b (GDF, 2-class motor imagery,
    bipolar C3/Cz/C4). Each subject has 3 *T (training) session files with
    real labels -- referenced via 'runs' in metadata.json. The 2 *E
    (evaluation) sessions' true labels ship as separate .mat files not
    included in this raw download, so they're excluded. Segments each
    recording into fixed-length windows starting at the per-class cue event
    (769/770), concatenated across all 3 runs for the subject.
    """
    _EVENT_TO_LABEL = {'769': 0, '770': 1}

    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.run_paths = [self._resolve(os.path.join(entry['folder'], r)) for r in entry['runs']]

    def _load_data(self):
        if self.standard_window is None:
            print(f"  [Warning] Subject {self.subject_id}: No standard_window defined, cannot segment data.")
            return None, None

        import mne
        trial_len_pts = int(self.standard_window * self.sample_freq)
        all_trials, all_labels = [], []

        for run_path in self._existing(self.run_paths):
            try:
                raw = mne.io.read_raw_gdf(run_path, preload=True, verbose=False)
                self._resample_if_needed(raw)

                trials, labels = self._segment_by_annotations(raw, trial_len_pts, self._EVENT_TO_LABEL, self.channel_indices)
                if not trials:
                    print(f"  [Warning] {os.path.basename(run_path)}: no cue events (769/770) found.")
                    continue
                all_trials.extend(trials)
                all_labels.extend(labels)

            except Exception as e:
                print(f"  [Warning] Error loading {os.path.basename(run_path)}: {e}")

        if not all_trials:
            return None, None
        return np.stack(all_trials), np.array(all_labels)


class GraspAndLiftLoader(BaseSubjectLoader):
    """
    Loader for the Kaggle "Grasp-and-Lift EEG Detection" dataset. Each
    series is a continuous recording annotated with 6 binary event columns
    (HandStart/FirstDigitTouch/BothStartLoadPhase/LiftOff/Replace/
    BothReleased) that overlap in time -- a multi-label sequence-labeling
    task, not one discrete class per trial like every other loader here.
    Rather than force that into a single dense label, this just chops each
    series into fixed-length non-overlapping windows (like BCICIVLoader's
    no-mrk fallback) with dummy label 0 -- fine for self-supervised
    pretraining, real event columns are read but discarded. Do not use for
    supervised finetune/eval of the event-detection task as-is.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)
        self.file_pairs = [(self._resolve(f['data']), self._resolve(f['events'])) for f in entry['files']]

    def _load_data(self):
        if self.standard_window is None:
            print(f"  [Warning] Subject {self.subject_id}: No standard_window defined, cannot segment data.")
            return None, None

        import pandas as pd
        trial_len = int(self.standard_window * self.sample_freq)
        all_trials = []

        existing_data_paths = set(self._existing([d for d, _ in self.file_pairs]))
        for data_path, events_path in self.file_pairs:
            if data_path not in existing_data_paths:
                continue
            df = pd.read_csv(data_path)
            data = df.iloc[:, 1:].values.astype(np.float32)  # (Time, Channels), drop 'id' column

            n_windows = data.shape[0] // trial_len
            for i in range(n_windows):
                start = i * trial_len
                all_trials.append(data[start:start + trial_len, self.channel_indices].T)

        if not all_trials:
            return None, None
        trials = np.stack(all_trials)
        return trials, np.zeros(len(trials), dtype=np.int64)


class TemplateLoader(BaseSubjectLoader):
    """
    Copy-paste starting point for a new dataset. Not registered in
    IO/dataset.py's _resolve_loader() — rename the class, fill in
    _load_data(), then register it. See DATASET_FORMAT.md for the full
    walkthrough (metadata.json schema, array shape contract, wiring steps).
    """
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
        raise NotImplementedError("Fill in TemplateLoader._load_data for your dataset.")
