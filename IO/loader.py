import os
import numpy as np
import scipy.io
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any

class BaseSubjectLoader(ABC):
    """
    Abstract base class for loading a single subject's EEG data.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        self.config = config
        self.subject_id = subject_id
        self.channel_indices = desired_channel_indices
        
        # Standardize metadata access
        self.data_metadata = config['data_metadata']
        self.dataset_params = config['dataset_params']
        
        self.data_root = self.dataset_params['dataset_path']
        
        self.num_targets = self.data_metadata['Number_of_Targets']
        self.sample_freq = self.data_metadata['Sample_Frequency']
        
        # Standard target points based on metadata default if needed for padding
        self.standard_window = self.data_metadata.get('Window_Size', 1.0)
        self.target_points = int(self.standard_window * self.sample_freq)

    def _get_standard_coords(self, ch_name: str) -> Optional[np.ndarray]:
        """Tries to find 3D coordinates for a channel using MNE standard montages."""
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

    @abstractmethod
    def _load_coords(self) -> np.ndarray:
        """Loads 3D Cartesian coordinates (x, y, z) for selected channels."""
        pass

    @abstractmethod
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Loads and formats data as (Trials, Channels, Time) and labels."""
        pass

    def get_subject_data(self) -> Dict[str, Any]:
        """
        Returns the processed data for the subject.
        """
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
    """
    Loader for BETA dataset.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        self.file_path = self._get_file_path()

    def _get_file_path(self):
        subject_str = str(self.subject_id)
        structure = self.config['data_structure']
        if subject_str not in structure:
             raise ValueError(f"Subject {self.subject_id} not found in data structure.")
        relative_path = structure[subject_str]['file'].lstrip('./')
        return os.path.join(self.data_root, relative_path)

    def _load_coords(self) -> np.ndarray:
        channel_config = self.data_metadata.get('channels', {})
        coords_list = []
        sorted_keys = sorted(channel_config.keys(), key=lambda k: int(k))
        
        for idx in self.channel_indices:
            if idx < len(sorted_keys):
                meta_key = sorted_keys[idx]
                label = channel_config[meta_key]['label']
            else:
                label = "Unknown"
                meta_key = None

            mne_coords = self._get_standard_coords(label)
            if mne_coords is not None:
                coords_list.append(mne_coords)
                continue

            if meta_key and meta_key in channel_config and 'coordinates' in channel_config[meta_key]:
                v = channel_config[meta_key]['coordinates']
                theta = np.deg2rad(v['polar_angle_deg'])
                r = v['polar_radius']
                x = r * np.sin(theta)
                y = r * np.cos(theta)
                coords_list.append([x, y, 0.0])
            else:
                coords_list.append([0.0, 0.0, 0.0])
                
        return np.array(coords_list, dtype=np.float32)

    def _load_data(self):
        if not os.path.exists(self.file_path):
            return None, None

        mat_data = scipy.io.loadmat(self.file_path)
        raw_data = mat_data['data']['EEG'][0, 0] 
        # raw_data shape: (Channels, Time, Blocks, Targets)
        
        # Determine actual blocks vs metadata blocks
        target_blocks = self.data_metadata.get('Number_of_Blocks', 0)
        actual_blocks = raw_data.shape[2]
        
        if target_blocks > 0 and actual_blocks != target_blocks:
            print(f"  [Warning] Subject {self.subject_id} has {actual_blocks} blocks, but metadata expected {target_blocks}. Adjusting...")
            blocks_to_use = min(actual_blocks, target_blocks)
        else:
            blocks_to_use = actual_blocks

        eeg_data = raw_data[self.channel_indices, :, :blocks_to_use, :]
        
        # Transpose to (Targets, Blocks, Channels, Time)
        eeg_data = np.transpose(eeg_data, (3, 2, 0, 1))
        # Reshape to (Total_Trials, Channels, Time)
        eeg_data = eeg_data.reshape(-1, len(self.channel_indices), raw_data.shape[1])
        
        # Generate labels
        labels = np.array([i for i in range(self.num_targets) for _ in range(blocks_to_use)])

        return eeg_data, labels

class DialLoader(BaseSubjectLoader):
    """
    Loader for Dial dataset.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        self.signal_path, self.label_path = self._get_file_paths()
    
    def _get_file_paths(self):
        subject_str = str(self.subject_id)
        structure = self.config['data_structure']
        if subject_str not in structure:
             raise ValueError(f"Subject {self.subject_id} not found in data structure.")
        
        sig_rel = structure[subject_str]['signals'].lstrip('./')
        lab_rel = structure[subject_str]['labels'].lstrip('./')
        return os.path.join(self.data_root, sig_rel), os.path.join(self.data_root, lab_rel)

    def _load_coords(self) -> np.ndarray:
        channel_config = self.data_metadata.get('channels', {})
        coords_list = []
        sorted_keys = sorted(channel_config.keys(), key=lambda k: int(k))
        
        for idx in self.channel_indices:
            if idx < len(sorted_keys):
                meta_key = sorted_keys[idx]
                label = channel_config[meta_key]['label']
            else:
                label = "Unknown"
                meta_key = None

            mne_coords = self._get_standard_coords(label)
            if mne_coords is not None:
                coords_list.append(mne_coords)
                continue

            if meta_key and meta_key in channel_config and 'coordinates' in channel_config[meta_key]:
                v = channel_config[meta_key]['coordinates']
                theta = np.deg2rad(v['polar_angle_deg'])
                r = v['polar_radius']
                x = r * np.sin(theta)
                y = r * np.cos(theta)
                coords_list.append([x, y, 0.0])
            else:
                coords_list.append([0.0, 0.0, 0.0])
                
        return np.array(coords_list, dtype=np.float32)

    def _load_data(self):
        if not os.path.exists(self.signal_path) or not os.path.exists(self.label_path):
             return None, None

        samples = scipy.io.loadmat(self.signal_path)['Data'] 
        raw_labels = scipy.io.loadmat(self.label_path)['Label'].flatten() 

        actual_trials = samples.shape[2]
        num_labels = len(raw_labels)
        
        if actual_trials != num_labels:
            print(f"  [Warning] Subject {self.subject_id} has {actual_trials} signals but {num_labels} labels. Using minimum.")
            trials_to_use = min(actual_trials, num_labels)
        else:
            trials_to_use = actual_trials

        # Load trials (Channels, Time, Trials) -> (Trials, Channels, Time)
        eeg_data = samples[self.channel_indices, :, :trials_to_use]
        eeg_data = np.transpose(eeg_data, (2, 0, 1))
        
        # Labels are 1-indexed in Dial mat files, convert to 0-indexed
        final_labels = (raw_labels[:trials_to_use] - 1).astype(np.int64)

        return eeg_data, final_labels

class BCICIVLoader(BaseSubjectLoader):
    """
    Loader for BCICIV dataset (Motor Imagery).
    Handles continuous data segmentation using markers.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        self.file_path = self._get_file_path()

    def _get_file_path(self):
        subject_str = str(self.subject_id)
        structure = self.config['data_structure']
        relative_path = structure[subject_str]['file'].lstrip('./')
        return os.path.join(self.data_root, relative_path)

    def _load_coords(self) -> np.ndarray:
        channel_config = self.data_metadata.get('channels', {})
        coords_list = []
        sorted_keys = sorted(channel_config.keys(), key=lambda k: int(k))
        for idx in self.channel_indices:
            meta_key = sorted_keys[idx]
            label = channel_config[meta_key]['label']
            mne_coords = self._get_standard_coords(label)
            if mne_coords is not None:
                coords_list.append(mne_coords)
            else:
                v = channel_config[meta_key].get('coordinates', {})
                theta = np.deg2rad(v.get('polar_angle_deg', 0))
                r = v.get('polar_radius', 0)
                coords_list.append([r * np.sin(theta), r * np.cos(theta), 0.0])
        return np.array(coords_list, dtype=np.float32)

    def _load_data(self):
        if not os.path.exists(self.file_path): return None, None
        mat = scipy.io.loadmat(self.file_path)
        cnt = mat['cnt'].astype(np.float32) # (Time, Channels)
        
        if 'mrk' not in mat:
            # print(f"  [Info] Subject {self.subject_id} in {self.file_path} has no markers (Evaluation Set).")
            trial_len = int(self.standard_window * self.sample_freq)
            pos = np.arange(0, cnt.shape[0] - trial_len, trial_len)
            y = np.zeros(len(pos))
        else:
            mrk = mat['mrk'][0,0]
            pos = mrk['pos'][0] # (Trials,)
            y = mrk['y'][0]     # (Trials,) labels are 1 and 2
        
        # Segment into trials
        trial_len = int(self.standard_window * self.sample_freq)
        trials = []
        valid_y = []
        
        for p, label in zip(pos, y):
            start = int(p)
            end = start + trial_len
            if end <= cnt.shape[0]:
                trial = cnt[start:end, self.channel_indices].T # (Channels, Time)
                trials.append(trial)
                valid_y.append(int(label) - 1 if label > 0 else 0) # 0-indexed
        
        if not trials: return None, None
        return np.stack(trials), np.array(valid_y)

class InriaLoader(BaseSubjectLoader):
    """
    Loader for Inria dataset.
    Loads from CSV files.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        self.signal_path, self.label_path = self._get_file_paths()
    
    def _get_file_paths(self):
        subject_str = str(self.subject_id)
        structure = self.config['data_structure']
        sig_rel = structure[subject_str]['signals'].lstrip('./')
        lab_rel = structure[subject_str]['labels'].lstrip('./')
        return os.path.join(self.data_root, sig_rel), os.path.join(self.data_root, lab_rel)

    def _load_coords(self) -> np.ndarray:
        channel_config = self.data_metadata.get('channels', {})
        coords_list = []
        sorted_keys = sorted(channel_config.keys(), key=lambda k: int(k))
        for idx in self.channel_indices:
            meta_key = sorted_keys[idx]
            v = channel_config[meta_key].get('coordinates', {})
            theta = np.deg2rad(v.get('polar_angle_deg', 0))
            r = v.get('polar_radius', 0)
            coords_list.append([r * np.sin(theta), r * np.cos(theta), 0.0])
        return np.array(coords_list, dtype=np.float32)

    def _load_data(self):
        if not os.path.exists(self.signal_path): return None, None
        import pandas as pd
        df = pd.read_csv(self.signal_path)
        data = df.iloc[:, 1:-1].values # (Time, Channels)
        markers = df.iloc[:, -1].values # (Time,)
        
        # Find trigger points
        trig_indices = np.where(np.diff(markers) > 0)[0] + 1
        
        # Load labels
        sub_sess = os.path.basename(self.signal_path).replace('Data_', '').replace('.csv', '')
        labels_df = pd.read_csv(self.label_path)
        sub_labels = labels_df[labels_df['IdFeedBack'].str.contains(sub_sess)]['Prediction'].values
        
        if len(sub_labels) == 0:
            # Fallback to SampleSubmission
            alt_path = os.path.join(os.path.dirname(self.label_path), "SampleSubmission.csv")
            if os.path.exists(alt_path):
                alt_df = pd.read_csv(alt_path)
                sub_labels = alt_df[alt_df['IdFeedBack'].str.contains(sub_sess)]['Prediction'].values

        trial_len = int(self.standard_window * self.sample_freq)
        trials = []
        final_y = []
        
        for i, idx in enumerate(trig_indices):
            if i < len(sub_labels):
                start = idx
                end = start + trial_len
                if end <= data.shape[0]:
                    trial = data[start:end, self.channel_indices].T # (Channels, Time)
                    trials.append(trial)
                    final_y.append(sub_labels[i])
                    
        if not trials: return None, None
        return np.stack(trials), np.array(final_y)

class BCI2000Loader(BaseSubjectLoader):
    """
    Loader for BCI2000 (PhysioNet MI) dataset.
    Loads from EDF files and segments based on T1/T2 annotations.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        self.run_paths = self._get_run_paths()

    def _get_run_paths(self):
        subject_str = str(self.subject_id)
        structure = self.config['data_structure']
        if subject_str not in structure:
            raise ValueError(f"Subject {self.subject_id} not found in BCI2000 data structure.")
        
        sub_info = structure[subject_str]
        sub_folder = sub_info['folder']
        return [os.path.join(self.data_root, sub_folder, r) for r in sub_info['runs']]

    def _load_coords(self) -> np.ndarray:
        channel_config = self.data_metadata.get('channels', {})
        coords_list = []
        sorted_keys = sorted(channel_config.keys(), key=lambda k: int(k))
        
        for idx in self.channel_indices:
            if idx < len(sorted_keys):
                meta_key = sorted_keys[idx]
                label = channel_config[meta_key]['label']
            else:
                label = "Unknown"
                meta_key = None

            mne_coords = self._get_standard_coords(label)
            if mne_coords is not None:
                coords_list.append(mne_coords)
                continue

            if meta_key and meta_key in channel_config and 'coordinates' in channel_config[meta_key]:
                v = channel_config[meta_key]['coordinates']
                theta = np.deg2rad(v['polar_angle_deg'])
                r = v['polar_radius']
                x = r * np.sin(theta)
                y = r * np.cos(theta)
                coords_list.append([x, y, 0.0])
            else:
                coords_list.append([0.0, 0.0, 0.0])
                
        return np.array(coords_list, dtype=np.float32)

    def _load_data(self):
        import mne
        all_trials = []
        all_labels = []
        
        trial_len_pts = int(self.standard_window * self.sample_freq)
        
        for run_path in self.run_paths:
            if not os.path.exists(run_path): continue
            
            try:
                raw = mne.io.read_raw_edf(run_path, preload=True, verbose=False)
                # Ensure correct SFreq by resampling if necessary, but BCI2000 is usually consistent
                if raw.info['sfreq'] != self.sample_freq:
                    raw.resample(self.sample_freq)
                
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                # Physionet MI event IDs: T1=2, T2=3 usually, but events_from_annotations might vary
                # We look for 'T1' and 'T2'
                rev_event_id = {v: k for k, v in event_id.items()}
                
                # We want T1 (class 0) and T2 (class 1)
                t1_id = event_id.get('T1')
                t2_id = event_id.get('T2')
                
                data_np = raw.get_data(picks=self.channel_indices) # (Channels, Time)
                
                for event in events:
                    start_pts = event[0]
                    ev_type = event[2]
                    
                    label = -1
                    if ev_type == t1_id: label = 0
                    elif ev_type == t2_id: label = 1
                    
                    if label != -1:
                        end_pts = start_pts + trial_len_pts
                        if end_pts <= data_np.shape[1]:
                            trial = data_np[:, start_pts:end_pts]
                            all_trials.append(trial)
                            all_labels.append(label)
            except Exception as e:
                print(f"  [Warning] Error loading {run_path}: {e}")
                
        if not all_trials:
            return None, None
            
        return np.stack(all_trials), np.array(all_labels)
