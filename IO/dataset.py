import os
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Callable, Any

from .loader import BETALoader, DialLoader, BCICIVLoader, InriaLoader, EEGMMIdbLoader, BCICIV2aLoader, BCICIV2bLoader, GraspAndLiftLoader
from IO.preprocessing import build_preprocessing_from_config
from IO.masking import BaseMaskingStrategy, RandomMaskingStrategy, ComplementaryMaskingStrategy

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

# --- Base Dataset ---

class EEGDataset(Dataset):
    """
    Loads EEG data from multiple subjects/datasets into a unified tensor.
    When assemble_trials=True, flattens each subject's trials into a continuous
    signal and cuts non-overlapping windows of assembly_params['trial_length'].
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
        all_valid_channels: List[torch.Tensor] = []  # per-task [Nc] bool, True = real (not zero-padded) channel
        all_valid_length: List[int] = []             # per-task original T, before cross-subject max_T padding

        print(f"Loading {len(loading_tasks)} subject-dataset tasks... (assembly={'on' if assemble_trials else 'off'})")
        for task in loading_tasks:
            try:
                result = self._load_task(task, desired_channels)
            except Exception as e:
                print(f"Failed to load Subject {task.get('subject_id')} ({task.get('dataset_name')}): {e}")
                continue
            if result is None:
                continue

            all_data_chunks.append(result['data'])
            all_label_chunks.append(result['labels'])
            all_subject_chunks.append(torch.full((len(result['data']),), int(result['subject_id']), dtype=torch.long))
            all_dataset_names.extend([result['dataset_name']] * len(result['data']))
            all_coords.append(result['coords'])
            all_valid_channels.append(result['valid_channels'])
            all_valid_length.append(result['valid_length'])

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
        self.all_valid_channels = all_valid_channels
        self.all_valid_length = all_valid_length

        self.trial_to_coords_idx = []
        for i, d in enumerate(standardized):
            self.trial_to_coords_idx.extend([i] * d.shape[0])

        print(f"Loaded {len(self.data)} trials, standardized to length {max_T}. Shape: {tuple(self.data.shape)}")

    def _load_task(self, task: Dict[str, Any], desired_channels: List[str]) -> Optional[Dict[str, Any]]:
        """
        Loads one subject-dataset task, channel-pads it into self.Nc unified
        channels, applies the preprocessing transform, and (for pretrain)
        windows it into fixed-length trials. Returns None if the subject has
        no data to load.
        """
        ds_name = task['dataset_name']
        subject_id = task['subject_id']
        loader_cls = task['loader_class']
        transform = task['transform']
        ds_config = task['dataset_config']

        ds_indices, target_pos = self._map_channels(desired_channels, ds_config['data_metadata']['channels'])
        loader = loader_cls(config=ds_config, subject_id=subject_id, desired_channel_indices=ds_indices)
        subject_data = loader.get_subject_data()
        if subject_data is None:
            return None

        raw_data = torch.from_numpy(subject_data['data'])  # (N, C, T)
        N, _, T = raw_data.shape

        # Transform (bandpass/resample/normalize) on the REAL channels only, before padding —
        # normalizing after zero-padding folds the zero-filled missing-channel rows into the
        # trial's mean/std (see IO/preprocessing.py's per-trial zscore/robust normalize), which
        # skews scale differently per dataset depending how many channels it's missing relative
        # to canonical_channels (e.g. BCICIV2a is ~2/3 zero-padded channels — that's a much
        # bigger normalization bias than a near-complete dataset like EEGMMIdb).
        if transform is not None:
            raw_data = torch.stack([transform(raw_data[i]) for i in range(N)])
        post_transform_T = raw_data.shape[-1]  # real (non-padded) length for finetune's per-trial mask

        padded = torch.zeros((N, self.Nc, post_transform_T), dtype=torch.float32)
        padded[:, target_pos, :] = raw_data

        if self.assemble_trials:
            padded, labels = self._window_subject_signal(padded, ds_name, subject_id)
        else:
            labels = torch.from_numpy(subject_data['labels'])

        task_coords = torch.zeros((self.Nc, 3), dtype=torch.float32)
        task_coords[target_pos] = torch.from_numpy(subject_data['coords'])

        valid_channels = torch.zeros(self.Nc, dtype=torch.bool)
        valid_channels[target_pos] = True

        return {
            'data': padded,
            'labels': labels,
            'dataset_name': ds_name,
            'subject_id': subject_id,
            'coords': task_coords,
            'valid_channels': valid_channels,
            'valid_length': post_transform_T,
        }

    def _window_subject_signal(self, trials: torch.Tensor, ds_name: str, subject_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flatten all trials into a continuous signal, then cut into non-overlapping
        windows of trial_length. Keeps the last chunk (zero-padded) only if it
        fills at least trial_pad_threshold of trial_length.
        """
        target_L = self.assembly_params.get('trial_length', trials.shape[-1])
        threshold = self.assembly_params.get('trial_pad_threshold', 0.5)

        N, C, T = trials.shape
        # NOT trials.reshape(N*T, C).T — that reinterprets the (N,C,T) memory buffer
        # directly without transposing, scrambling channel and time together. permute
        # first so reshape only merges the N and T axes (already-adjacent after permute),
        # keeping each channel's own timeseries intact and trial-concatenated in order.
        signal = trials.permute(1, 0, 2).reshape(C, N * T)  # (C, N*T)
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
        print(f"  [{ds_name} S{subject_id}] {N} trials x {T}pts -> {total_T}pts -> {len(assembled)} windows of {target_L}pts.")
        return assembled, torch.zeros(len(assembled), dtype=torch.long)

    # Old → canonical label aliases (covers both 10-20 naming conventions)
    _LABEL_ALIASES = {
        'T3': 'T7', 'T4': 'T8',
        'T5': 'P7', 'T6': 'P8',
        'A1': 'TP9', 'A2': 'TP10',
        # BCICIV_1-style intermediate-ring labels -> nearest canonical 10-10 site
        'CFC1': 'FC1', 'CFC2': 'FC2', 'CFC3': 'FC3', 'CFC4': 'FC4',
        'CFC5': 'FC5', 'CFC6': 'FC6', 'CFC7': 'FT7', 'CFC8': 'FT8',
        'CCP1': 'CP1', 'CCP2': 'CP2', 'CCP3': 'CP3', 'CCP4': 'CP4',
        'CCP5': 'CP5', 'CCP6': 'CP6', 'CCP7': 'TP7', 'CCP8': 'TP8',
        'PO1': 'PO3', 'PO2': 'PO4',
    }

    def _normalize_label(self, label: str) -> str:
        up = label.strip().upper()
        return self._LABEL_ALIASES.get(up, up)

    def _map_channels(self, desired_channels: List[str], channel_config: Dict) -> Tuple[List[int], List[int]]:
        """Returns (dataset_channel_indices, positions_in_desired_list)."""
        name_to_index = {}
        for key, info in channel_config.items():
            if isinstance(key, str) and key.isdigit() and isinstance(info, dict) and 'label' in info:
                norm = self._normalize_label(info['label'])
                name_to_index[norm] = int(key) - 1  # metadata is 1-indexed

        ds_indices, target_pos, missing = [], [], []
        for i, name in enumerate(desired_channels):
            norm = self._normalize_label(name)
            if norm in name_to_index:
                ds_indices.append(name_to_index[norm])
                target_pos.append(i)
            else:
                missing.append(name)
        print(f"  [channel map] matched {len(ds_indices)}/{len(desired_channels)}"
              + (f" | zero-padded: {missing}" if missing else ""))
        return ds_indices, target_pos

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


# --- Dataset Wrappers ---

class MaskedPretrainDataset(Dataset):
    """
    Wraps EEGDataset for masked pretraining.
    Yields: (x_patches, coords, mask, time_indices, label, fft_patches, valid_channels)
      x_patches:      [C, P, L]
      coords:         [C, 3]
      mask:           [C * P] bool
      time_indices:   [P]
      label:          scalar
      fft_patches:    [C, P, F] or empty tensor
      valid_channels: [C] bool, True = real (not zero-padded) channel
    """
    def __init__(
        self,
        base_dataset: EEGDataset,
        patch_len: Optional[int] = None,
        mask_ratio: float = 0.5,
        masking_strategy: Optional[BaseMaskingStrategy] = None,
    ):
        self.base_dataset     = base_dataset
        self.masking_strategy = masking_strategy or RandomMaskingStrategy()
        self.mask_ratio       = self.masking_strategy.effective_mask_ratio(mask_ratio)

        if patch_len is None:
            model_type = base_dataset.config.get('training_params', {}).get('pretrain', {}).get('model_type', 'MeFSQ')
            preprocess = base_dataset.config.get('model_params', {}).get(model_type, {}).get('preprocess', {})
            patch_len = preprocess.get('patch_length', 200)

        self.patch_len = patch_len
        total_T     = base_dataset.data.shape[-1]
        num_patches = total_T // patch_len
        remainder   = total_T % patch_len

        # pre-generate one mask per trial so complementary pairs are exact inverses
        self._masks = [
            self.masking_strategy.generate_mask(base_dataset.Nc, num_patches, self.mask_ratio)
            for _ in range(len(base_dataset))
        ]

        strategy_name = type(self.masking_strategy).__name__.replace('MaskingStrategy', '').lower()
        n_effective   = len(base_dataset) * self.masking_strategy.multiplier
        print(f"\n--- MaskedPretrainDataset ---")
        print(f"  {len(base_dataset)} trials | {num_patches} patches/trial | mask_ratio={self.mask_ratio} | strategy={strategy_name}")
        print(f"  effective dataset size: {n_effective}")
        if remainder > 0:
            print(f"  [Truncation] {remainder} samples dropped per trial ({remainder/total_T*100:.1f}%).")
        print(f"----------------------------\n")

    def __len__(self):
        return len(self.base_dataset) * self.masking_strategy.multiplier

    def __getitem__(self, index):
        N = len(self.base_dataset)
        trial_idx, mask = self.masking_strategy.resolve(self._masks, index, N)

        x, y = self.base_dataset[trial_idx]
        C, T = x.shape
        P = T // self.patch_len
        L = self.patch_len

        x_patches    = x[:, :P * L].reshape(C, P, L)
        time_indices = torch.arange(P, dtype=torch.long)

        if self.base_dataset.fft_params is not None:
            n_fft = self.base_dataset.fft_params.get('n_fft')
            norm  = self.base_dataset.fft_params.get('norm', 'ortho')
            fft_patches = torch.fft.rfft(x_patches, n=n_fft, dim=-1, norm=norm)
        else:
            fft_patches = torch.empty(0)

        coords_idx = self.base_dataset.trial_to_coords_idx[trial_idx]
        coords     = self.base_dataset.all_coords[coords_idx]
        valid_channels = self.base_dataset.all_valid_channels[coords_idx]

        return x_patches, coords, mask, time_indices, y, fft_patches, valid_channels


class FinetuneDataset(Dataset):
    """
    Wraps EEGDataset for supervised finetuning and trial-level inspection.
    Preserves original trial boundaries and labels.
    Yields: (x, coords, label, valid_channels, valid_length)
      x:              [C, T]
      coords:         [C, 3]
      label:          scalar
      valid_channels: [C] bool, True = real (not zero-padded) channel
      valid_length:   scalar int, real (non-padded) time length
    """
    def __init__(self, base_dataset: EEGDataset):
        self.base_dataset = base_dataset
        n_classes = len(set(base_dataset.labels.tolist()))
        print(f"Initializing FinetuneDataset: {len(base_dataset)} trials, {n_classes} classes, shape {tuple(base_dataset.data.shape[1:])}.")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, label = self.base_dataset[index]
        task_idx = self.base_dataset.trial_to_coords_idx[index]
        coords = self.base_dataset.all_coords[task_idx]
        valid_channels = self.base_dataset.all_valid_channels[task_idx]
        valid_length = self.base_dataset.all_valid_length[task_idx]
        return x, coords, label, valid_channels, valid_length


# --- Factory ---

def _resolve_loader(dataset_name: str, ds_name_key: str):
    for name in (dataset_name, ds_name_key):
        if 'BETA' in name:
            return BETALoader
        if 'BCICIV2a' in name:
            return BCICIV2aLoader
        if 'BCICIV2b' in name:
            return BCICIV2bLoader
        if 'BCICIV' in name:
            return BCICIVLoader
        if 'Inria' in name:
            return InriaLoader
        if 'GraspAndLift' in name:
            return GraspAndLiftLoader
        if 'EEGMMIdb' in name:
            return EEGMMIdbLoader
        if 'Dial' in name:
            return DialLoader
    return DialLoader  # safe fallback


def _load_montage_channels(name: str) -> List[str]:
    """Looks up a named montage's ordered channel-label list from config/montages.json —
    see docs/agents/adding-a-montage.md. Coordinates in that file are reference/QA data
    only (per-dataset coords are still resolved independently in IO/loader.py); only the
    label order matters here."""
    montage_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'montages.json')
    with open(montage_path, 'r', encoding='utf-8') as f:
        montages = json.load(f)
    if name not in montages:
        raise ValueError(f"Unknown montage '{name}' — not found in {montage_path}. "
                          f"Available: {list(montages.keys())}")
    return [ch['label'] for ch in montages[name]['channels']]


def _resolve_target_channels(dataset_params: Dict, pp: Dict = None) -> List[str]:
    """
    Determines the unified channel list.
    If preprocess_params contains 'canonical_channels', that fixed ordered list is used
    directly — channel index = electrode identity across all datasets. It may be given as
    a literal list (custom, used as-is) or a string naming a montage in
    config/montages.json (see docs/agents/adding-a-montage.md).
    Otherwise falls back to reading from the first dataset's metadata.
    """
    if pp:
        canonical = pp.get('canonical_channels', [])
        if isinstance(canonical, str) and canonical:
            return _load_montage_channels(canonical)
        if canonical:
            return canonical

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


def build_dataset_from_config(config_dict: Dict, transform: Optional[Callable] = None, mode: str = 'pretrain',
                               assemble_trials: Optional[bool] = None) -> Dataset:
    ds_mode        = mode if mode in ('pretrain', 'finetune') else 'pretrain'
    dataset_params = config_dict.get('dataset_params', {}).get(ds_mode, {})
    pp             = config_dict.get('preprocess_params', {})
    patch_len      = pp.get('patch_length', 100)

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

    target_channels = _resolve_target_channels(dataset_params, pp=pp)

    fft_params = None

    if assemble_trials is None:
        assemble_trials = mode in ('pretrain',)  # explicit override: e.g. real per-trial
        # labels for codebook diagnostics (mode='pretrain' normally assembles trials into
        # continuous-signal windows, which discards real labels -- see _window_subject_signal)
    assembly_params = pp

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
    elif mode == 'pretrain':
        mask_pp       = pp.get('mask', {})
        strategy_name = mask_pp.get('masking_strategy', 'random')
        strategy_cfg  = mask_pp.get(strategy_name, {})

        if strategy_name == 'complementary':
            strategy   = ComplementaryMaskingStrategy()
            mask_ratio = ComplementaryMaskingStrategy.MASK_RATIO
        else:
            strategy   = RandomMaskingStrategy()
            mask_ratio = strategy_cfg.get('mask_ratio', 0.5)

        return MaskedPretrainDataset(base_dataset, patch_len=patch_len,
                                     mask_ratio=mask_ratio, masking_strategy=strategy)
    elif mode == 'finetune':
        return FinetuneDataset(base_dataset)
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Expected one of: base, tokenizer, pretrain, finetune.")
