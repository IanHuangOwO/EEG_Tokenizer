import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Callable, Any
from abc import ABC, abstractmethod

from .loader import BETALoader, DialLoader, BCICIVLoader, InriaLoader, EEGMMIdbLoader
from model.factory import build_preprocessing_from_config

# Channels excluded by default when channels_to_use is "all".
# Set include_non_eeg_channels: true in dataset_params to override.
NON_EEG_CHANNELS = {
    'EOG', 'VEOG', 'HEOG', 'EOG1', 'EOG2',
    'EMG', 'EMG1', 'EMG2',
    'ECG', 'EKG',
    'A1', 'A2', 'M1', 'M2',
    'REF', 'LREF', 'RREF',
    'STI', 'STIM', 'STATUS', 'TRIGGER',
}

# --- Masking Strategies ---

class BaseMaskingStrategy(ABC):
    @abstractmethod
    def generate_mask(self, num_channels: int, num_patches: int, mask_ratio: float) -> torch.Tensor:
        """Returns a boolean mask of shape (num_channels * num_patches,)."""
        pass


class RandomMaskingStrategy(BaseMaskingStrategy):
    def generate_mask(self, num_channels: int, num_patches: int, mask_ratio: float) -> torch.Tensor:
        num_tokens = num_channels * num_patches
        num_masked = int(num_tokens * mask_ratio)
        indices = torch.randperm(num_tokens)
        mask = torch.zeros(num_tokens, dtype=torch.bool)
        mask[indices[:num_masked]] = True
        return mask


class BlockMaskingStrategy(BaseMaskingStrategy):
    """
    Masks entire channels (row) or entire time patches (col) to simulate
    sensor loss or device disconnects, then fills the remainder randomly.
    """
    def __init__(self, row_prob: float = 0.5, col_prob: float = 0.5):
        self.row_prob = row_prob
        self.col_prob = col_prob

    def generate_mask(self, num_channels: int, num_patches: int, mask_ratio: float) -> torch.Tensor:
        mask = torch.zeros((num_channels, num_patches), dtype=torch.bool)
        target_total = int(num_channels * num_patches * mask_ratio)

        num_rows = min(int(np.round(target_total * self.row_prob / num_patches)), num_channels)
        if num_rows > 0:
            mask[torch.randperm(num_channels)[:num_rows], :] = True

        num_cols = min(int(np.round(target_total * self.col_prob / num_channels)), num_patches)
        if num_cols > 0:
            mask[:, torch.randperm(num_patches)[:num_cols]] = True

        remaining = target_total - mask.sum().item()
        if remaining > 0:
            flat = mask.flatten()
            unmasked = torch.where(~flat)[0]
            flat[unmasked[torch.randperm(len(unmasked))[:remaining]]] = True
            mask = flat.reshape(num_channels, num_patches)

        return mask.flatten()


# --- Base Dataset ---

class EEGDataset(Dataset):
    """
    Loads EEG data from multiple subjects/datasets into a unified tensor.
    When assemble_trials=True, flattens each subject's trials into a continuous
    signal and cuts non-overlapping windows of assembly_params['target_length'].
    """
    def __init__(
        self,
        config: Dict,
        loading_tasks: List[Dict[str, Any]],
        desired_channels: List[str],
        fft_params: Optional[Dict] = None,
        assemble_trials: bool = False,
        assembly_params: Optional[Dict] = None
    ):
        self.config = config
        self.channel_names = desired_channels
        self.Nc = len(desired_channels)
        self.assemble_trials = assemble_trials
        self.assembly_params = assembly_params or {}
        self.fft_params = fft_params

        all_data_chunks: List[torch.Tensor] = []
        all_label_chunks: List[torch.Tensor] = []
        all_subject_chunks: List[torch.Tensor] = []
        all_dataset_names: List[str] = []
        all_coords: List[torch.Tensor] = []

        print(f"Loading {len(loading_tasks)} subject-dataset tasks... (assembly={'on' if assemble_trials else 'off'})")
        for task in loading_tasks:
            try:
                ds_name = task['dataset_name']
                subject_id = task['subject_id']
                loader_cls = task['loader_class']
                transform = task['transform']
                ds_config = task['dataset_config']

                ds_indices, target_pos = self._map_channels(desired_channels, ds_config['data_metadata']['channels'])
                loader = loader_cls(config=ds_config, subject_id=subject_id, desired_channel_indices=ds_indices)
                subject_data = loader.get_subject_data()

                if subject_data is None:
                    continue

                raw_data = torch.from_numpy(subject_data['data'])  # (N, C, T)
                N, _, T = raw_data.shape

                padded = torch.zeros((N, self.Nc, T), dtype=torch.float32)
                padded[:, target_pos, :] = raw_data

                if transform is not None:
                    padded = torch.stack([transform(padded[i]) for i in range(N)])

                if self.assemble_trials:
                    padded, labels = self._window_subject_signal(padded, ds_name, subject_id)
                else:
                    labels = torch.from_numpy(subject_data['labels'])

                all_data_chunks.append(padded)
                all_label_chunks.append(labels)
                all_subject_chunks.append(torch.full((len(padded),), subject_id, dtype=torch.long))
                all_dataset_names.extend([ds_name] * len(padded))

                task_coords = torch.zeros((self.Nc, 3), dtype=torch.float32)
                task_coords[target_pos] = torch.from_numpy(subject_data['coords'])
                all_coords.append(task_coords)

            except Exception as e:
                print(f"Failed to load Subject {task.get('subject_id')} ({task.get('dataset_name')}): {e}")

        if not all_data_chunks:
            raise RuntimeError("No datasets were loaded successfully.")

        # Standardize temporal length across all subjects
        max_T = max(d.shape[-1] for d in all_data_chunks)
        standardized = []
        for d in all_data_chunks:
            if d.shape[-1] < max_T:
                pad = torch.zeros((d.shape[0], d.shape[1], max_T - d.shape[-1]), dtype=d.dtype)
                d = torch.cat([d, pad], dim=-1)
            standardized.append(d)

        self.data = torch.cat(standardized)
        self.labels = torch.cat(all_label_chunks)
        self.subject_data = torch.cat(all_subject_chunks)
        self.dataset_names = all_dataset_names
        self.all_coords = all_coords

        self.trial_to_coords_idx = []
        for i, d in enumerate(standardized):
            self.trial_to_coords_idx.extend([i] * d.shape[0])

        print(f"Loaded {len(self.data)} trials, standardized to length {max_T}. Shape: {tuple(self.data.shape)}")

    def _window_subject_signal(self, trials: torch.Tensor, ds_name: str, subject_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flatten all trials into a continuous signal, then cut into non-overlapping
        windows of target_length. Keeps the last chunk (zero-padded) only if it
        fills at least pad_threshold of target_length.
        """
        target_L = self.assembly_params.get('target_length', trials.shape[-1])
        threshold = self.assembly_params.get('pad_threshold', 0.5)

        N, C, T = trials.shape
        signal = trials.reshape(N * T, C).T  # (C, N*T)
        total_T = signal.shape[-1]

        n_complete = total_T // target_L
        remainder = total_T % target_L

        windows = [signal[:, i * target_L:(i + 1) * target_L] for i in range(n_complete)]

        if remainder > 0:
            if remainder >= target_L * threshold:
                last = torch.zeros((C, target_L), dtype=signal.dtype)
                last[:, :remainder] = signal[:, n_complete * target_L:]
                windows.append(last)
                print(f"  [{ds_name} S{subject_id}] Last chunk {remainder}/{target_L}pts kept (padded {target_L - remainder}pts).")
            else:
                print(f"  [{ds_name} S{subject_id}] Last chunk {remainder}/{target_L}pts discarded (below {threshold*100:.0f}% threshold).")

        if not windows:
            raise RuntimeError(f"No windows produced for {ds_name} subject {subject_id} (total_T={total_T}, target_L={target_L}).")

        assembled = torch.stack(windows)
        print(f"  [{ds_name} S{subject_id}] {N} trials × {T}pts → {total_T}pts → {len(assembled)} windows of {target_L}pts.")
        return assembled, torch.zeros(len(assembled), dtype=torch.long)

    def _map_channels(self, desired_channels: List[str], channel_config: Dict) -> Tuple[List[int], List[int]]:
        """Returns (dataset_channel_indices, positions_in_desired_list)."""
        name_to_index = {}
        for key, info in channel_config.items():
            if isinstance(key, str) and key.isdigit() and isinstance(info, dict) and 'label' in info:
                name_to_index[info['label'].upper()] = int(key) - 1  # metadata is 1-indexed

        ds_indices, target_pos = [], []
        for i, name in enumerate(desired_channels):
            if name.upper() in name_to_index:
                ds_indices.append(name_to_index[name.upper()])
                target_pos.append(i)
        return ds_indices, target_pos

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


# --- Dataset Wrappers ---

class TokenizerDataset(Dataset):
    """
    Wraps EEGDataset for AttnVQ tokenizer training.
    Yields: (x, coords, time_indices, label, x_fft)
      x:            [C, N, L]
      coords:       [C, 3]
      time_indices: [N]
      label:        scalar
      x_fft:        [C, F] or empty tensor
    """
    def __init__(self, base_dataset: EEGDataset, patch_len: Optional[int] = None):
        self.base_dataset = base_dataset

        if patch_len is None:
            model_type = base_dataset.config.get('training_params', {}).get('model_type', 'AttnVQ')
            preprocess = base_dataset.config.get('model_params', {}).get(model_type, {}).get('preprocess', {})
            patch_len = preprocess.get('patch_length', 200)

        self.patch_len = patch_len
        total_T = self.base_dataset.data.shape[-1]
        self.num_patches = total_T // patch_len
        remainder = total_T % patch_len

        print(f"Initializing TokenizerDataset: {len(base_dataset)} trials, {self.num_patches} patches per trial.")
        if remainder > 0:
            print(f"  [Truncation] {remainder} samples dropped per trial ({remainder/total_T*100:.1f}%).")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, label = self.base_dataset[index]
        C = x.shape[0]
        N, L = self.num_patches, self.patch_len

        x_patches = x[:, :N * L].reshape(C, N, L)
        coords_idx = self.base_dataset.trial_to_coords_idx[index]
        coords = self.base_dataset.all_coords[coords_idx]
        time_indices = torch.arange(N, dtype=torch.long)

        if self.base_dataset.fft_params is not None:
            n_fft = self.base_dataset.fft_params.get('n_fft')
            norm = self.base_dataset.fft_params.get('norm', 'ortho')
            x_fft = torch.fft.rfft(x_patches.reshape(C, -1), n=n_fft, dim=-1, norm=norm)
        else:
            x_fft = torch.empty(0)

        return x_patches, coords, time_indices, label, x_fft


class MaskedPretrainDataset(Dataset):
    """
    Wraps EEGDataset for masked pretraining.
    Yields: (x_patches, coords, mask, time_indices, label, fft_patches)
      x_patches:    [C, P, L]
      coords:       [C, 3]
      mask:         [C * P] bool
      time_indices: [P]
      label:        scalar
      fft_patches:  [C, P, F] or empty tensor
    """
    def __init__(
        self,
        base_dataset: EEGDataset,
        patch_len: Optional[int] = None,
        mask_ratio: float = 0.5,
        masking_strategy: Optional[BaseMaskingStrategy] = None
    ):
        self.base_dataset = base_dataset
        self.mask_ratio = mask_ratio
        self.masking_strategy = masking_strategy or RandomMaskingStrategy()

        if patch_len is None:
            model_type = base_dataset.config.get('training_params', {}).get('model_type', 'AttnVQ')
            preprocess = base_dataset.config.get('model_params', {}).get(model_type, {}).get('preprocess', {})
            patch_len = preprocess.get('patch_length', 200)

        self.patch_len = patch_len
        total_T = base_dataset.data.shape[-1]
        num_patches = total_T // patch_len
        remainder = total_T % patch_len

        print(f"\n--- MaskedPretrainDataset ---")
        print(f"  {len(base_dataset)} trials | {num_patches} patches/trial | mask_ratio={mask_ratio}")
        if remainder > 0:
            print(f"  [Truncation] {remainder} samples dropped per trial ({remainder/total_T*100:.1f}%).")
        print(f"----------------------------\n")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, y = self.base_dataset[index]
        C, T = x.shape
        P = T // self.patch_len
        L = self.patch_len

        x_patches = x[:, :P * L].reshape(C, P, L)
        mask = self.masking_strategy.generate_mask(C, P, self.mask_ratio)
        time_indices = torch.arange(P, dtype=torch.long)

        if self.base_dataset.fft_params is not None:
            n_fft = self.base_dataset.fft_params.get('n_fft')
            norm = self.base_dataset.fft_params.get('norm', 'ortho')
            fft_patches = torch.fft.rfft(x_patches, n=n_fft, dim=-1, norm=norm)
        else:
            fft_patches = torch.empty(0)

        coords_idx = self.base_dataset.trial_to_coords_idx[index]
        coords = self.base_dataset.all_coords[coords_idx]

        return x_patches, coords, mask, time_indices, y, fft_patches


class FinetuneDataset(Dataset):
    """
    Wraps EEGDataset for supervised finetuning and trial-level inspection.
    Preserves original trial boundaries and labels.
    Yields: (x, coords, label)
      x:      [C, T]
      coords: [C, 3]
      label:  scalar
    """
    def __init__(self, base_dataset: EEGDataset):
        self.base_dataset = base_dataset
        n_classes = len(set(base_dataset.labels.tolist()))
        print(f"Initializing FinetuneDataset: {len(base_dataset)} trials, {n_classes} classes, shape {tuple(base_dataset.data.shape[1:])}.")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, label = self.base_dataset[index]
        coords_idx = self.base_dataset.trial_to_coords_idx[index]
        coords = self.base_dataset.all_coords[coords_idx]
        return x, coords, label


# --- Factory ---

def _resolve_loader(dataset_name: str, ds_name_key: str):
    for name in (dataset_name, ds_name_key):
        if 'BETA' in name:
            return BETALoader
        if 'BCICIV' in name:
            return BCICIVLoader
        if 'Inria' in name:
            return InriaLoader
        if 'EEGMMIdb' in name:
            return EEGMMIdbLoader
        if 'Dial' in name:
            return DialLoader
    return DialLoader  # safe fallback


def _resolve_target_channels(dataset_params: Dict) -> List[str]:
    """
    Determines the unified channel list from the first dataset's config.
    Applies NON_EEG_CHANNELS exclusion when channels_to_use is 'all'.
    """
    first_ds_key = next(iter(dataset_params))
    first_ds_args = dataset_params[first_ds_key]
    channels_to_use = first_ds_args.get('channels_to_use', ['all'])
    include_non_eeg = first_ds_args.get('include_non_eeg_channels', False)

    if channels_to_use not in ('all', ['all']):
        return channels_to_use

    meta_path = os.path.join(first_ds_args['dataset_path'], 'metadata.json')
    with open(meta_path, 'r', encoding='utf-8') as f:
        channel_dict = json.load(f).get('data_metadata', {}).get('channels', {})

    sorted_keys = sorted(
        [k for k in channel_dict if isinstance(k, str) and k.isdigit()],
        key=lambda k: int(k)
    )
    target_channels = []
    for k in sorted_keys:
        ch_info = channel_dict[k]
        if isinstance(ch_info, dict) and 'label' in ch_info:
            label = ch_info['label']
            if include_non_eeg or label.upper() not in NON_EEG_CHANNELS:
                target_channels.append(label)
    return target_channels


def build_dataset_from_config(config_dict: Dict, transform: Optional[Callable] = None, mode: str = 'tokenizer') -> Dataset:
    dataset_params = config_dict.get('dataset_params', {})
    model_type = config_dict.get('training_params', {}).get('model_type', 'AttnVQ')
    model_params = config_dict.get('model_params', {}).get(model_type, {})
    preprocess_params = model_params.get('preprocess', {})
    tokenizer_params = model_params.get('tokenizer', {})
    patch_len = preprocess_params.get('patch_length', 200)

    loading_tasks = []
    for ds_name, ds_args in dataset_params.items():
        meta_path = os.path.join(ds_args['dataset_path'], 'metadata.json')
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        data_metadata = metadata.get('data_metadata', {})
        data_structure = metadata.get('data_structure', {})

        fs_orig = data_metadata['acquisition']['sample_frequency']
        dataset_name = data_metadata.get('dataset_name', ds_name)
        loader_cls = _resolve_loader(dataset_name, ds_name)

        ds_transform = transform if transform is not None else build_preprocessing_from_config(config_dict, fs_orig=fs_orig)

        loader_config = {
            'dataset_params': ds_args,
            'data_metadata': data_metadata,
            'data_structure': data_structure
        }

        for sub_id in ds_args['subject_to_use']:
            loading_tasks.append({
                'dataset_name': ds_name,
                'subject_id': sub_id,
                'loader_class': loader_cls,
                'transform': ds_transform,
                'dataset_config': loader_config
            })

    target_channels = _resolve_target_channels(dataset_params)

    fft_params = None
    if mode in ('tokenizer', 'pretrain'):
        freq_res = tokenizer_params.get('freq_resolution', 1.0)
        target_fs = preprocess_params.get('target_freq', 200.0)
        fft_params = {
            'patch_len': patch_len,
            'n_fft': int(target_fs / freq_res),
            'norm': 'ortho',
        }

    assemble_trials = mode in ('tokenizer', 'pretrain')
    assembly_params = config_dict.get('trial_assembly', {})

    base_dataset = EEGDataset(
        config=config_dict,
        loading_tasks=loading_tasks,
        desired_channels=target_channels,
        fft_params=fft_params,
        assemble_trials=assemble_trials,
        assembly_params=assembly_params
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
        return MaskedPretrainDataset(base_dataset, patch_len=patch_len, mask_ratio=mask_ratio, masking_strategy=strategy)
    elif mode == 'finetune':
        return FinetuneDataset(base_dataset)
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Expected one of: base, tokenizer, pretrain, finetune.")
