import os
import json
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, ConcatDataset
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Callable

# --- Base Loader ---

class BaseSubjectLoader(Dataset, ABC):
    """
    Abstract base class for loading a single subject's EEG data.
    """
    def __init__(
        self, 
        data_root: str, 
        subject_id: int, 
        config: Dict, 
        desired_channel_indices: List[int], 
        trials_to_use: int, 
        window_size: Optional[float] = None
    ):
        self.data_root = data_root
        self.subject_id = subject_id
        self.config = config
        self.channel_indices = desired_channel_indices
        self.trials_to_use = trials_to_use
        
        # Standardize target parameters
        self.num_targets = config['Number_of_Targets']
        self.sample_freq = config['Sample_Frequency']
        
        # Determine Window Size and Target Points
        self.window_size = window_size if window_size is not None else config['Window_Size']
        self.target_points = int(self.window_size * self.sample_freq)

    def _get_standard_coords(self, ch_name: str) -> Optional[np.ndarray]:
        """Tries to find 3D coordinates for a channel using MNE standard montages."""
        try:
            import mne
            # Static cache for montage to avoid reloading
            if not hasattr(self, '_std_montage'):
                self._std_montage = mne.channels.make_standard_montage('standard_1020')
            
            # MNE stores keys in upper case usually
            if not hasattr(self, '_std_positions'):
                 self._std_positions = self._std_montage.get_positions()['ch_pos']
            
            # Normalize name
            keys = {k.upper(): k for k in self._std_positions.keys()}
            if ch_name.upper() in keys:
                return self._std_positions[keys[ch_name.upper()]]
                
        except ImportError:
            pass
        except Exception as e:
            # print(f"MNE lookup error: {e}")
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

    def pad_or_crop(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Ensures time dimension matches self.target_points.
        Input: (Channels, Time)
        """
        actual_time = eeg_data.shape[-1]
        if actual_time < self.target_points:
            pad_width = self.target_points - actual_time
            # Pad the last dimension (Time) with zeros
            return np.pad(eeg_data, ((0, 0), (0, pad_width)), mode='constant')
        elif actual_time > self.target_points:
            return eeg_data[:, :self.target_points]
        return eeg_data

# --- Specific Loaders ---

class BETALoader(BaseSubjectLoader):
    """
    Loader for BETA dataset.
    File format: Single .mat file per subject.
    Data structure: 'EEG' -> (Channels, Time, Blocks, Targets)
    """
    def __init__(self, data_root, subject_id, data_structure, config, desired_channel_indices, trials_to_use, window_size=None):
        self.file_path = self._get_file_path(data_root, subject_id, data_structure)
        super().__init__(data_root, subject_id, config, desired_channel_indices, trials_to_use, window_size)
        self.coords = self._load_coords()
        self.data, self.labels = self._load_data()

    def _get_file_path(self, data_root, subject_id, data_structure):
        subject_str = str(subject_id)
        if subject_str not in data_structure:
             raise ValueError(f"Subject {subject_id} not found in data structure.")
        relative_path = data_structure[subject_str]['file'].lstrip('./')
        return os.path.join(data_root, relative_path)

    def _load_coords(self) -> np.ndarray:
        channel_config = self.config.get('channels', {})
        coords_list = []
        
        # We need the channel NAMES to lookup MNE, not just indices
        # Invert the mapping or look up by index from sorted keys
        sorted_keys = sorted(channel_config.keys(), key=lambda k: int(k))
        # This list corresponds to indices 0, 1, 2...
        
        for idx in self.channel_indices:
            # Get Label
            if idx < len(sorted_keys):
                meta_key = sorted_keys[idx]
                label = channel_config[meta_key]['label']
            else:
                label = "Unknown"

            # 1. Try MNE Standard (Preferred for 3D)
            mne_coords = self._get_standard_coords(label)
            if mne_coords is not None:
                coords_list.append(mne_coords) # [x, y, z]
                continue

            # 2. Fallback to Metadata 2D -> 3D Projection
            # Project 2D polar onto a sphere of radius approx 0.095m (MNE scale)
            # Or just use z=0 if we really have to
            if meta_key in channel_config and 'coordinates' in channel_config[meta_key]:
                v = channel_config[meta_key]['coordinates']
                theta = np.deg2rad(v['polar_angle_deg'])
                r = v['polar_radius'] # This is likely normalized or projected
                
                # Assume r is projected on 2D, scale it to similar magnitude as MNE if needed
                # MNE coords are usually in meters (~0.09)
                # BETA r is ~0.5. Let's just use x, y, 0 for fallback.
                x = r * np.sin(theta)
                y = r * np.cos(theta)
                coords_list.append([x, y, 0.0])
            else:
                print(f"Warning: No coords for {label}, using (0,0,0)")
                coords_list.append([0.0, 0.0, 0.0])
                
        return np.array(coords_list, dtype=np.float32)

    def _load_data(self):
        if not os.path.exists(self.file_path):
            print(f"Warning: File not found {self.file_path}")
            return None, None

        mat_data = scipy.io.loadmat(self.file_path)
        if 'data' not in mat_data or 'EEG' not in mat_data['data'].dtype.names:
            raise ValueError(f"Invalid BETA format in {self.file_path}")

        # Raw shape: (Channels, Time, Blocks, Targets)
        raw_data = mat_data['data']['EEG'][0, 0] 

        # 1. Select Channels
        # Shape: (Selected_Channels, Time, Blocks, Targets)
        eeg_data = raw_data[self.channel_indices, :, :, :]
        
        # 2. Check and Select Trials (Blocks)
        actual_blocks = eeg_data.shape[2]
        if self.trials_to_use > actual_blocks:
             raise ValueError(f"Subject {self.subject_id}: Requested {self.trials_to_use} trials, but only {actual_blocks} available.")
        
        # Shape: (Selected_Channels, Time, Selected_Trials, Targets)
        eeg_data = eeg_data[:, :, :self.trials_to_use, :]

        # 3. Reshape to (Total_Trials, Channels, Time)
        # Transpose to (Targets, Selected_Trials, Channels, Time)
        eeg_data = np.transpose(eeg_data, (3, 2, 0, 1))
        # Flatten Targets * Trials
        eeg_data = eeg_data.reshape(-1, len(self.channel_indices), raw_data.shape[1])

        # 4. Generate Labels
        # Labels 0 to num_targets-1, repeated for each block
        labels = np.array([i for i in range(self.num_targets) for _ in range(self.trials_to_use)])

        return eeg_data, labels

    def __getitem__(self, index):
        sample = self.data[index] # (Channels, Time)
        sample = self.pad_or_crop(sample)
        return torch.from_numpy(sample).float(), self.labels[index]

    def __len__(self):
        return len(self.labels) if self.labels is not None else 0


class DialLoader(BaseSubjectLoader):
    """
    Loader for Dial dataset.
    File format: Separate Signal and Label .mat files per subject.
    Data structure: 'Data' -> (Channels, Time, Trials)
    """
    def __init__(self, data_root, subject_id, data_structure, config, desired_channel_indices, trials_to_use, window_size=None):
        self.signal_path, self.label_path = self._get_file_paths(data_root, subject_id, data_structure)
        super().__init__(data_root, subject_id, config, desired_channel_indices, trials_to_use, window_size)
        self.coords = self._load_coords()
        self.data, self.labels = self._load_data()
    
    def _get_file_paths(self, data_root, subject_id, data_structure):
        subject_str = str(subject_id)
        if subject_str not in data_structure:
             raise ValueError(f"Subject {subject_id} not found in data structure.")
        
        sig_rel = data_structure[subject_str]['signals'].lstrip('./')
        lab_rel = data_structure[subject_str]['labels'].lstrip('./')
        
        return os.path.join(data_root, sig_rel), os.path.join(data_root, lab_rel)

    def _load_coords(self) -> np.ndarray:
        # Same logic as BETA, can be refactored to base if strictly identical metadata structure
        channel_config = self.config.get('channels', {})
        coords_list = []
        sorted_keys = sorted(channel_config.keys(), key=lambda k: int(k))
        
        for idx in self.channel_indices:
            if idx < len(sorted_keys):
                meta_key = sorted_keys[idx]
                label = channel_config[meta_key]['label']
            else:
                label = "Unknown"

            mne_coords = self._get_standard_coords(label)
            if mne_coords is not None:
                coords_list.append(mne_coords)
                continue

            if meta_key in channel_config and 'coordinates' in channel_config[meta_key]:
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
             print(f"Warning: Missing files for subject {self.subject_id}")
             return None, None

        # Load Signals: (Channels, Time, All_Trials)
        samples = scipy.io.loadmat(self.signal_path)['Data'] 

        # Load Labels: 1D array of 1-based labels
        raw_labels = scipy.io.loadmat(self.label_path)['Label'].flatten() 

        # Filter trials to ensure balanced classes based on trials_to_use
        indices_to_keep = []
        for label_val in range(1, self.num_targets + 1):
            matching_indices = np.where(raw_labels == label_val)[0]
            if len(matching_indices) < self.trials_to_use:
                 raise ValueError(f"Subject {self.subject_id}: Not enough trials for class {label_val}. Needed {self.trials_to_use}, found {len(matching_indices)}.")
            indices_to_keep.extend(matching_indices[:self.trials_to_use])

        # Select Data: (Selected_Channels, Time, Selected_Trials)
        eeg_data = samples[self.channel_indices, :, :][:, :, indices_to_keep]
        
        # Reshape to (Selected_Trials, Selected_Channels, Time)
        eeg_data = np.transpose(eeg_data, (2, 0, 1))

        # Adjust labels to 0-based index
        final_labels = raw_labels[indices_to_keep] - 1

        return eeg_data, final_labels

    def __getitem__(self, index):
        sample = self.data[index]
        sample = self.pad_or_crop(sample)
        return torch.from_numpy(sample).float(), int(self.labels[index])

    def __len__(self):
        return len(self.labels) if self.labels is not None else 0


# --- Main Dataset Wrapper ---

class EEGDataset(Dataset):
    """
    Universal EEG Dataset Wrapper.
    Combines data from multiple subjects into a single PyTorch Dataset.
    Exposes labels and subject indices for visualization/analysis.
    """
    def __init__(
        self, 
        data_root: str, 
        metadata_json: Dict, 
        subject_list: List[int], 
        desired_channels: List[str], 
        trials_to_use: int, 
        loader_class: Callable, 
        window_size: Optional[float] = None, 
        transform: Optional[Callable] = None
    ):
        self.transform = transform
        self.datasets = []
        
        config = metadata_json['data_metadata']
        structure = metadata_json['data_structure']

        # Store Metadata
        self.channel_names = desired_channels
        self.Nc = len(desired_channels)
        self.subject_list = subject_list

        # Map channel names to indices
        channel_indices = self._map_channels(desired_channels, config['channels'])

        # Store Global Indices
        # We need to know: Index i corresponds to Subject S and Label L
        self.all_labels = []
        self.all_subject_ids = []
        
        self.coords = None

        # Initialize Subject Loaders
        print(f"Loading {len(subject_list)} subjects...")
        for subject_id in subject_list:
            try:
                loader = loader_class(
                    data_root=data_root, 
                    subject_id=subject_id, 
                    data_structure=structure, 
                    config=config, 
                    desired_channel_indices=channel_indices, 
                    trials_to_use=trials_to_use, 
                    window_size=window_size
                )
                if len(loader) > 0:
                    self.datasets.append(loader)
                    
                    # Store coordinates from first successful loader
                    if self.coords is None:
                        self.coords = loader.coords
                    
                    # Store labels and subject IDs for quick lookup
                    # loader.labels is numpy array
                    self.all_labels.append(torch.from_numpy(loader.labels))
                    self.all_subject_ids.append(torch.full((len(loader),), subject_id, dtype=torch.long))
                    
            except Exception as e:
                print(f"Failed to load Subject {subject_id}: {e}")

        if not self.datasets:
            raise RuntimeError("No datasets were loaded successfully.")

        self.full_dataset = ConcatDataset(self.datasets)
        
        # Concatenate metadata tensors for global lookup
        self.label_data = torch.cat(self.all_labels)
        self.subject_data = torch.cat(self.all_subject_ids)

    def _map_channels(self, desired_channels: List[str], channel_config: Dict) -> List[int]:
        """Converts list of channel names to their corresponding 0-based indices."""
        name_to_index = {}
        for key, info in channel_config.items():
            if 'label' in info:
                # Metadata keys are 1-based strings ('1', '2'...)
                name_to_index[info['label']] = int(key) - 1
        
        indices = []
        for name in desired_channels:
            if name in name_to_index:
                indices.append(name_to_index[name])
            else:
                print(f"Warning: Channel '{name}' not found in metadata.")
        return indices

    def __getitem__(self, index):
        x, y = self.full_dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.full_dataset)
    

def build_dataset_from_config(config_dict: Dict, transform: Optional[Callable] = None) -> EEGDataset:
    """
    Factory function to build an EEGDataset from a configuration dictionary.
    """
    params = config_dict.get('dataset_params', {})
    
    # Extract Parameters
    data_root = params.get('dataset_path')
    subjects = params.get('subjects', [1])
    trials_to_use = params.get('trials_to_use', 1)
    channels_to_use = params.get('channels_to_use', "all")
    window_size = params.get('window_size_to_use')
    
    if not data_root:
        raise ValueError("dataset_path must be specified in config['dataset_params']")
    
    # 1. Load Metadata
    meta_path = os.path.join(data_root, 'metadata.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")
        
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
        
    # 2. Determine Channels
    if channels_to_use == "all" or channels_to_use == ["all"]:
         channel_dict = metadata.get('data_metadata', {}).get('channels', {})
         # Sort channels by their numeric index in metadata to ensure correct order
         sorted_keys = sorted(channel_dict.keys(), key=lambda k: int(k))
         channels_to_use = [channel_dict[k]['label'] for k in sorted_keys]
         print(f"Selected all {len(channels_to_use)} channels from metadata.")

    # 3. Determine Loader Class
    dataset_name = metadata.get('data_metadata', {}).get('dataset_name')
    if dataset_name == 'BETA':
        loader_cls = BETALoader
    elif dataset_name == 'Dial':
        loader_cls = DialLoader
    else:
        raise ValueError(f"Unknown dataset_name in metadata: {dataset_name}")

    # 4. Instantiate Dataset
    return EEGDataset(
        data_root=data_root,
        metadata_json=metadata,
        subject_list=subjects,
        desired_channels=channels_to_use,
        trials_to_use=trials_to_use,
        loader_class=loader_cls,
        window_size=window_size,
        transform=transform
    )

# --- Tokenizer Specific Wrappers ---

class TokenizerWrapperDataset(Dataset):
    """
    Wraps the standard EEGDataset to yield fixed-length patches (e.g., 1s)
    instead of full trials. Returns (patch, coordinates).
    """
    def __init__(self, base_dataset: EEGDataset, patch_len: int = 200):
        self.base_dataset = base_dataset
        self.patch_len = patch_len
        
        # Determine patches per trial
        sample_x, _ = self.base_dataset[0]
        self.total_len = sample_x.shape[-1]
        self.patches_per_trial = self.total_len // patch_len
        
        # Pre-convert coords to tensor for efficiency
        self.coords_tensor = torch.from_numpy(base_dataset.coords).float()
        
        print(f"Tokenizer Wrapper: {len(self.base_dataset)} trials -> {len(self)} patches ({self.patches_per_trial} per trial)")

    def __len__(self):
        return len(self.base_dataset) * self.patches_per_trial

    def __getitem__(self, index):
        trial_idx = index // self.patches_per_trial
        patch_offset = (index % self.patches_per_trial) * self.patch_len
        
        x, _ = self.base_dataset[trial_idx]
        patch = x[:, patch_offset : patch_offset + self.patch_len]
        
        return patch, self.coords_tensor


class TokenizerWrapperDatasetWithLabels(Dataset):
    """
    Similar to TokenizerWrapperDataset, but also returns the trial label.
    Used for visualization (t-SNE) where class info is needed.
    """
    def __init__(self, base_dataset: EEGDataset, patch_len: int = 200):
        self.base_dataset = base_dataset
        self.patch_len = patch_len
        
        sample_x, _ = self.base_dataset[0]
        self.total_len = sample_x.shape[-1]
        self.patches_per_trial = self.total_len // patch_len
        
        self.coords_tensor = torch.from_numpy(base_dataset.coords).float()

    def __len__(self):
        return len(self.base_dataset) * self.patches_per_trial

    def __getitem__(self, index):
        trial_idx = index // self.patches_per_trial
        patch_offset = (index % self.patches_per_trial) * self.patch_len
        
        x, y = self.base_dataset[trial_idx]
        patch = x[:, patch_offset : patch_offset + self.patch_len]
        
        return patch, self.coords_tensor, y