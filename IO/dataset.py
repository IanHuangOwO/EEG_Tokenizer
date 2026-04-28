import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Callable, Any
from abc import ABC, abstractmethod

from .loader import BETALoader, DialLoader, BCICIVLoader, InriaLoader, BCI2000Loader
from model.factory import build_preprocessing_from_config

# --- Masking Strategies ---

class BaseMaskingStrategy(ABC):
    @abstractmethod
    def generate_mask(self, num_channels: int, num_patches: int, mask_ratio: float) -> torch.Tensor:
        """
        Generates a boolean mask.
        Returns:
            torch.Tensor: Boolean mask of shape (num_channels * num_patches,)
        """
        pass

class RandomMaskingStrategy(BaseMaskingStrategy):
    def generate_mask(self, num_channels: int, num_patches: int, mask_ratio: float) -> torch.Tensor:
        num_tokens = num_channels * num_patches
        num_masked = int(num_tokens * mask_ratio)
        
        indices = torch.randperm(num_tokens)
        masked_indices = indices[:num_masked]
        
        mask = torch.zeros(num_tokens, dtype=torch.bool)
        mask[masked_indices] = True
        return mask

class BlockMaskingStrategy(BaseMaskingStrategy):
    """
    Masks entire channels or entire temporal patches.
    Mimics sensor loss or device disconnects.
    """
    def __init__(self, row_prob: float = 0.5, col_prob: float = 0.5):
        self.row_prob = row_prob
        self.col_prob = col_prob

    def generate_mask(self, num_channels: int, num_patches: int, mask_ratio: float) -> torch.Tensor:
        mask = torch.zeros((num_channels, num_patches), dtype=torch.bool)
        total_tokens = num_channels * num_patches
        target_total = int(total_tokens * mask_ratio)
        
        # Calculate target budgets for each block type independently
        # Row tokens: tokens assigned to full-channel masking
        target_row_tokens = target_total * self.row_prob
        target_col_tokens = target_total * self.col_prob
        
        # 1. Row Masking (Channel Loss)
        num_rows = int(np.round(target_row_tokens / num_patches))
        num_rows = min(num_rows, num_channels)
        if num_rows > 0:
            rows = torch.randperm(num_channels)[:num_rows]
            mask[rows, :] = True
            
        # 2. Column Masking (Disconnect)
        num_cols = int(np.round(target_col_tokens / num_channels))
        num_cols = min(num_cols, num_patches)
        if num_cols > 0:
            cols = torch.randperm(num_patches)[:num_cols]
            mask[:, cols] = True
            
        current_masked = mask.sum().item()

        # 3. Fill remaining budget with random tokens
        if current_masked < target_total:
            remaining = target_total - current_masked
            flat_mask = mask.flatten()
            unmasked_indices = torch.where(~flat_mask)[0]
            if len(unmasked_indices) > 0:
                to_mask = unmasked_indices[torch.randperm(len(unmasked_indices))[:remaining]]
                flat_mask[to_mask] = True
                mask = flat_mask.reshape(num_channels, num_patches)
        
        # 4. Optional: If we significantly exceeded target_total due to block overlap/size,
        # we can leave it (block masking is often slightly more than random).
        # But here we just return the union.

        return mask.flatten()

# --- Base Dataset ---

class EEGDataset(Dataset):
    """
    Universal EEG Dataset Wrapper.
    Handles loading from multiple subjects and precomputing FFTs.
    """
    def __init__(
        self, 
        config: Dict,
        loading_tasks: List[Dict[str, Any]], 
        desired_channels: List[str], 
        fft_params: Optional[Dict] = None
    ):
        self.config = config
        self.channel_names = desired_channels
        self.Nc = len(desired_channels)

        self.all_data = []
        self.all_labels = []
        self.all_subject_ids = []
        self.all_dataset_names = []
        self.all_coords = [] # List of (Nc, 3) tensors, one per subject/task
        
        # Determine target length
        model_type = config.get('training_params', {}).get('model_type', 'AttnVQ')
        preprocess_params = config.get('model_params', {}).get(model_type, {}).get('preprocess', {})
        self.target_T = preprocess_params.get('trial_length_to_use', None)

        print(f"Loading {len(loading_tasks)} subject-dataset tasks...")
        for task in loading_tasks:
            try:
                ds_name = task.get('dataset_name', 'Unknown')
                subject_id = task['subject_id']
                # print(f"  [Task] Loading Subject {subject_id} from {ds_name}...")
                loader_cls = task['loader_class']
                transform = task['transform']
                ds_config = task['dataset_config']
                
                # Map channel names: get indices in dataset and their positions in desired_channels
                ds_indices, target_pos = self._map_channels_v2(desired_channels, ds_config['data_metadata']['channels'])

                loader = loader_cls(
                    config=ds_config,
                    subject_id=subject_id, 
                    desired_channel_indices=ds_indices
                )
                subject_data = loader.get_subject_data()
                
                if subject_data:
                    raw_data = torch.from_numpy(subject_data['data'])
                    num_trials, _, num_time = raw_data.shape
                    
                    # Create zero-padded tensor for consistent channel count
                    padded_data = torch.zeros((num_trials, self.Nc, num_time), dtype=torch.float32)
                    padded_data[:, target_pos, :] = raw_data
                    
                    # Apply dataset-specific transform
                    if transform is not None:
                        processed_trials = []
                        for i in range(len(padded_data)):
                            x_proc = transform(padded_data[i])
                            processed_trials.append(x_proc)
                        padded_data = torch.stack(processed_trials)
                    
                    self.all_data.append(padded_data)
                    self.all_labels.append(torch.from_numpy(subject_data['labels']))
                    self.all_subject_ids.append(torch.full((len(subject_data['labels']),), subject_id, dtype=torch.long))
                    self.all_dataset_names.extend([ds_name] * len(subject_data['labels']))
                    
                    # Store task-specific coords (padded to match self.Nc)
                    task_coords = torch.zeros((self.Nc, 3), dtype=torch.float32)
                    task_coords[target_pos] = torch.from_numpy(subject_data['coords'])
                    self.all_coords.append(task_coords)
                    
            except Exception as e:
                print(f"Failed to load Task (Subject {task.get('subject_id')}): {e}")

        if not self.all_data:
            raise RuntimeError("No datasets were loaded successfully.")

        # Handle varying temporal lengths by padding to max_T
        max_T = max(d.shape[-1] for d in self.all_data)
        standardized_data = []
        for d in self.all_data:
            if d.shape[-1] < max_T:
                padding = torch.zeros((d.shape[0], d.shape[1], max_T - d.shape[-1]), dtype=d.dtype)
                d = torch.cat([d, padding], dim=-1)
            standardized_data.append(d)

        self.data = torch.cat(standardized_data)
        self.labels = torch.cat(self.all_labels)
        self.subject_data = torch.cat(self.all_subject_ids)
        self.dataset_names = self.all_dataset_names
        
        # Track which trial uses which coordinate set
        self.trial_to_coords_idx = []
        for i, d in enumerate(self.all_data):
            self.trial_to_coords_idx.extend([i] * d.shape[0])

        print(f"All trials loaded and standardized to length {max_T}. Total trials: {len(self.data)}. Data shape: {self.data.shape}")

        # Store FFT params for on-the-fly CPU computation
        self.fft_params = fft_params

    def _map_channels_v2(self, desired_channels: List[str], channel_config: Dict) -> Tuple[List[int], List[int]]:
        """Returns (indices_in_dataset, positions_in_desired)"""
        name_to_index = {}
        for key, info in channel_config.items():
            if 'label' in info:
                name_to_index[info['label'].upper()] = int(key) - 1
        
        ds_indices = []
        target_pos = []
        for i, name in enumerate(desired_channels):
            if name.upper() in name_to_index:
                ds_indices.append(name_to_index[name.upper()])
                target_pos.append(i)
        return ds_indices, target_pos

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

# --- Wrappers ---

class TokenizerDataset(Dataset):
    """
    Wraps EEGDataset to yield trials structured for AttnVQ.
    Yields: (x, coords, time_indices, label, x_fft)
    x: [C, N, L]
    coords: [C, 3]
    time_indices: [N]
    x_fft: [C, N, F]
    """
    def __init__(self, base_dataset: EEGDataset, patch_len: Optional[int] = None):
        self.base_dataset = base_dataset
        
        if patch_len is None:
            model_type = base_dataset.config.get('training_params', {}).get('model_type', 'AttnVQ')
            preprocess_params = base_dataset.config.get('model_params', {}).get(model_type, {}).get('preprocess', {})
            patch_len = preprocess_params.get('patch_length', 200)
            
        self.patch_len = patch_len
        
        # Calculate N based on standardized data length
        _, _, total_T = self.base_dataset.data.shape
        self.num_patches = total_T // self.patch_len
        
        print(f"Initializing Tokenizer Dataset: {len(self.base_dataset)} trials, {self.num_patches} patches per trial.")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, label = self.base_dataset[index]
        C, T = x.shape
        N = self.num_patches
        L = self.patch_len
        
        # Reshape to [C, N, L]
        x_reshaped = x[:, :N*L].reshape(C, N, L)
        
        # Coordinates: [C, 3]
        coords_idx = self.base_dataset.trial_to_coords_idx[index]
        coords = self.base_dataset.all_coords[coords_idx]
        
        # Time indices: [N]
        time_indices = torch.arange(N, dtype=torch.long)
        
        # Trial-level FFT: [C, n_fft//2+1] — one spectrum for the full trial
        # n_fft comes from freq_resolution in config: n_fft = fs / freq_resolution
        # If n_fft < T: truncates; if n_fft > T: zero-pads (both handled by rfft's n= arg)
        if self.base_dataset.fft_params is not None:
            n_fft = self.base_dataset.fft_params.get('n_fft')
            norm = self.base_dataset.fft_params.get('norm', 'ortho')
            x_trial = x_reshaped.reshape(C, -1)  # [C, T]
            x_fft = torch.fft.rfft(x_trial, n=n_fft, dim=-1, norm=norm)
        else:
            x_fft = torch.empty(0)
            
        return x_reshaped, coords, time_indices, label, x_fft

class MaskedPretrainDataset(Dataset):
    """
    Wraps EEGDataset for Masked Pretraining.
    Yields: (x_patches, coords, mask, time_indices, y, [fft_patches])
    """
    def __init__(
        self, 
        base_dataset: EEGDataset, 
        patch_len: Optional[int] = None, 
        mask_ratio: float = 0.5,
        masking_strategy: Optional[BaseMaskingStrategy] = None
    ):
        self.base_dataset = base_dataset
        
        if patch_len is None:
            model_type = base_dataset.config.get('training_params', {}).get('model_type', 'AttnVQ')
            preprocess_params = base_dataset.config.get('model_params', {}).get(model_type, {}).get('preprocess', {})
            patch_len = preprocess_params.get('patch_length', 200)
            
        self.patch_len = patch_len
        self.mask_ratio = mask_ratio
        self.masking_strategy = masking_strategy or RandomMaskingStrategy()
        
        ds_patch_counts = {name: 0 for name in set(self.base_dataset.dataset_names)}
        total_patches = 0
        for i in range(len(self.base_dataset)):
            x, _ = self.base_dataset[i]
            p = x.shape[-1] // self.patch_len
            total_patches += p
            ds_name = self.base_dataset.dataset_names[i]
            ds_patch_counts[ds_name] += p

        print(f"\n--- Masked Pretrain Dataset Summary ---")
        print(f"Grand Total: {len(self.base_dataset)} trials -> {total_patches} patches total")
        print(f"Mask Ratio: {mask_ratio}")
        print(f"---------------------------------------\n")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, y = self.base_dataset[index]
        C, T = x.shape
        num_patches = T // self.patch_len
        
        # Reshape into patches: (C, P, T_patch)
        x_patches = x[:, :num_patches * self.patch_len].reshape(
            C, num_patches, self.patch_len
        )
        
        # Generate Mask
        mask = self.masking_strategy.generate_mask(
            C, num_patches, self.mask_ratio
        )
        
        time_indices = torch.arange(num_patches, dtype=torch.long)
        
        # FFT data: [C, N, F] - Always compute on CPU on-the-fly
        if self.base_dataset.fft_params is not None:
            n_fft = self.base_dataset.fft_params.get('n_fft')
            norm = self.base_dataset.fft_params.get('norm', 'ortho')
            fft_patches = torch.fft.rfft(x_patches, n=n_fft, dim=-1, norm=norm)
        else:
            fft_patches = torch.empty(0)
        
        # Get coordinates: (C, 3)
        coords_idx = self.base_dataset.trial_to_coords_idx[index]
        coords = self.base_dataset.all_coords[coords_idx]
        
        return x_patches, coords, mask, time_indices, y, fft_patches

# --- Factory ---

def build_dataset_from_config(config_dict: Dict, transform: Optional[Callable] = None, mode: str = 'tokenizer') -> Dataset:
    dataset_params = config_dict.get('dataset_params', {})
    model_type = config_dict.get('training_params', {}).get('model_type', 'AttnVQ')
    model_params = config_dict.get('model_params', {}).get(model_type, {})
    preprocess_params = model_params.get('preprocess', {})
    params = model_params.get('tokenizer', {}) # Get model-specific params
    patch_len = preprocess_params.get('patch_length', 200)
    
    loading_tasks = []
    all_channels_found = set()

    for ds_name, ds_args in dataset_params.items():
        data_root = ds_args['dataset_path']
        subject_list = ds_args['subject_to_use']
        
        meta_path = os.path.join(data_root, 'metadata.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        data_metadata = metadata.get('data_metadata', {})
        data_structure = metadata.get('data_structure', {})
        fs_orig = data_metadata.get('Sample_Frequency')
        
        if data_metadata.get('dataset_name') == 'BETA' or data_metadata.get('dataset_name') == 'BETA_3s' or data_metadata.get('dataset_name') == 'BETA_4s':
            loader_cls = BETALoader
        elif data_metadata.get('dataset_name') == 'Dial':
            loader_cls = DialLoader
        elif data_metadata.get('dataset_name') == 'BCICIV':
            loader_cls = BCICIVLoader
        elif data_metadata.get('dataset_name') == 'Inria':
            loader_cls = InriaLoader
        elif data_metadata.get('dataset_name') == 'BCI2000':
            loader_cls = BCI2000Loader
        else:
            if 'BETA' in ds_name: loader_cls = BETALoader
            elif 'BCICIV' in ds_name: loader_cls = BCICIVLoader
            elif 'Inria' in ds_name: loader_cls = InriaLoader
            elif 'BCI2000' in ds_name: loader_cls = BCI2000Loader
            else: loader_cls = DialLoader

        if transform is None:
            ds_transform = build_preprocessing_from_config(config_dict, fs_orig=fs_orig)
        else:
            ds_transform = transform

        loader_config = {
            'dataset_params': ds_args,
            'data_metadata': data_metadata,
            'data_structure': data_structure
        }

        for sub_id in subject_list:
            loading_tasks.append({
                'dataset_name': ds_name,
                'subject_id': sub_id,
                'loader_class': loader_cls,
                'transform': ds_transform,
                'dataset_config': loader_config
            })
            
        for ch_info in data_metadata.get('channels', {}).values():
            all_channels_found.add(ch_info['label'])

    first_ds_key = list(dataset_params.keys())[0]
    target_channels = dataset_params[first_ds_key].get('channels_to_use', ["all"])
    
    if target_channels == "all" or target_channels == ["all"]:
         if not loading_tasks:
             raise RuntimeError(f"No subjects found for dataset splitting. Check your dataset_path and subject_to_use configuration.")
         first_meta = loading_tasks[0]['dataset_config']['data_metadata']
         channel_dict = first_meta.get('channels', {})
         sorted_keys = sorted(channel_dict.keys(), key=lambda k: int(k))
         target_channels = [channel_dict[k]['label'] for k in sorted_keys]

    fft_params = None
    if mode in ['tokenizer', 'pretrain']:
        # Align n_fft with model calculation: fs / freq_resolution
        freq_res = params.get('freq_resolution', 1.0)
        target_fs = preprocess_params.get('target_freq', 200.0)
        n_fft = int(target_fs / freq_res)
        
        fft_params = {
            'patch_len': patch_len,
            'n_fft': n_fft,
            'norm': 'ortho',
            'enabled': preprocess_params.get('precompute_fft', False)
        }

    base_dataset = EEGDataset(
        config=config_dict,
        loading_tasks=loading_tasks,
        desired_channels=target_channels,
        fft_params=fft_params
    )
    
    if mode == 'base':
        return base_dataset
    elif mode == 'tokenizer':
        return TokenizerDataset(base_dataset, patch_len=patch_len)
    elif mode == 'pretrain':
        mask_ratio = preprocess_params.get('mask_ratio', 0.5)
        strategy_name = preprocess_params.get('masking_strategy', 'random')
        
        if strategy_name == 'block':
            strategy = BlockMaskingStrategy(
                row_prob=preprocess_params.get('mask_row_prob', 0.5),
                col_prob=preprocess_params.get('mask_col_prob', 0.5)
            )
        else:
            strategy = RandomMaskingStrategy()
            
        return MaskedPretrainDataset(
            base_dataset, 
            patch_len=patch_len, 
            mask_ratio=mask_ratio,
            masking_strategy=strategy
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")