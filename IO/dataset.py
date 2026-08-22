import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Callable, Any

from .loader import load_coords_from_metadata
from IO.preprocessing import build_normalizer_from_config, cache_suffix, slice_patches, window_continuous_signal
from IO.masking import BaseMaskingStrategy, RandomMaskingStrategy, ComplementaryMaskingStrategy, RandomToComplementaryMaskingStrategy

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
        transform = task['transform']
        ds_config = task['dataset_config']

        ds_indices, target_pos = self._map_channels(desired_channels, ds_config['data_metadata']['channels'])

        # Train-time read: compiled cache only (see cache_compile.py) — dataset-specific
        # loading code (datas/<Name>/loader.py) never runs at train time. The cached
        # array holds ALL native channels in metadata.json's index order (compile time
        # keeps every channel, no target-channel subsetting), so ds_indices indexes
        # directly into it, no extra mapping needed.
        dataset_path = ds_config['dataset_params']['dataset_path']
        cache_path = os.path.join(dataset_path, 'cache', f"{subject_id}_{task['cache_suffix']}.npz")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No compiled cache at {cache_path}. Run "
                f"`python cache_compile.py --config config/compile.json` first "
                f"(and make sure compile.json's sample_freq/bandpass_filter match "
                f"this config's preprocess_params)."
            )
        npz = np.load(cache_path)
        data_np = npz['data'][:, ds_indices, :]
        if data_np.shape[0] == 0:
            return None
        coords_np = load_coords_from_metadata(ds_config['data_metadata'], ds_indices)

        raw_data = torch.from_numpy(data_np.astype(np.float32))  # (N, C, T)
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
            target_L = self.assembly_params.get('trial_length', padded.shape[-1])
            threshold = self.assembly_params.get('trial_pad_threshold', 0.5)
            padded, labels = window_continuous_signal(padded, target_L, threshold, ds_name, subject_id)
        else:
            labels = torch.from_numpy(npz['labels'].astype(np.int64))

        task_coords = torch.zeros((self.Nc, 3), dtype=torch.float32)
        task_coords[target_pos] = torch.from_numpy(coords_np.astype(np.float32))

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

def _resolve_default_patch_len(base_dataset: 'EEGDataset') -> int:
    model_type = base_dataset.config.get('training_params', {}).get('pretrain', {}).get('model_type', 'MeFSQ')
    preprocess = base_dataset.config.get('model_params', {}).get(model_type, {}).get('preprocess', {})
    return preprocess.get('patch_length', 200)


class TokenizerDataset(Dataset):
    """
    Wraps EEGDataset for the Tokenizer stage — unmasked, patchified. Unlike
    PretrainDataset there is no masking-strategy machinery here: bool_masked_pos
    is always None for this stage (see train_tokenizer.py), so no mask needs
    generating, tracking, or curriculum-swapping, and __len__ has no multiplier
    (a masking-strategy multiplier like ComplementaryMaskingStrategy's 2x would
    otherwise silently double "epoch" size for a mask this stage never reads).
    Yields the same 7-tuple shape as PretrainDataset for unpack compatibility
    (x_patches, coords, mask, time_indices, label, fft_patches, valid_channels)
    — mask is a constant all-False placeholder, never read downstream.
      x_patches:      [C, P, L]
      coords:         [C, 3]
      mask:           [C * P] bool, always False
      time_indices:   [P]
      label:          scalar
      fft_patches:    [C, P, F] or empty tensor
      valid_channels: [C] bool, True = real (not zero-padded) channel
    """
    def __init__(self, base_dataset: 'EEGDataset', patch_len: Optional[int] = None):
        self.base_dataset = base_dataset
        self.patch_len = patch_len or _resolve_default_patch_len(base_dataset)

        total_T = base_dataset.data.shape[-1]
        num_patches = total_T // self.patch_len
        remainder = total_T % self.patch_len

        print(f"\n--- TokenizerDataset ---")
        print(f"  {len(base_dataset)} trials | {num_patches} patches/trial | unmasked")
        if remainder > 0:
            print(f"  [Truncation] {remainder} samples dropped per trial ({remainder/total_T*100:.1f}%).")
        print(f"----------------------------\n")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, y = self.base_dataset[index]
        x_patches, time_indices = slice_patches(x, self.patch_len)
        C, P, _L = x_patches.shape
        mask = torch.zeros(C * P, dtype=torch.bool)

        if self.base_dataset.fft_params is not None:
            n_fft = self.base_dataset.fft_params.get('n_fft')
            norm  = self.base_dataset.fft_params.get('norm', 'ortho')
            fft_patches = torch.fft.rfft(x_patches, n=n_fft, dim=-1, norm=norm)
        else:
            fft_patches = torch.empty(0)

        coords_idx = self.base_dataset.trial_to_coords_idx[index]
        coords     = self.base_dataset.all_coords[coords_idx]
        valid_channels = self.base_dataset.all_valid_channels[coords_idx]

        return x_patches, coords, mask, time_indices, y, fft_patches, valid_channels


class PretrainDataset(Dataset):
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
            patch_len = _resolve_default_patch_len(base_dataset)

        self.patch_len = patch_len
        total_T     = base_dataset.data.shape[-1]
        self.num_patches = total_T // patch_len
        remainder   = total_T % patch_len

        # pre-generate one mask per trial so complementary pairs are exact inverses
        self._masks = [
            self.masking_strategy.generate_mask(base_dataset.Nc, self.num_patches, self.mask_ratio)
            for _ in range(len(base_dataset))
        ]

        strategy_name = type(self.masking_strategy).__name__.replace('MaskingStrategy', '').lower()
        n_effective   = len(base_dataset) * self.masking_strategy.multiplier
        print(f"\n--- PretrainDataset ---")
        print(f"  {len(base_dataset)} trials | {self.num_patches} patches/trial | mask_ratio={self.mask_ratio} | strategy={strategy_name}")
        print(f"  effective dataset size: {n_effective}")
        if remainder > 0:
            print(f"  [Truncation] {remainder} samples dropped per trial ({remainder/total_T*100:.1f}%).")
        print(f"----------------------------\n")

    def set_masking(self, masking_strategy: BaseMaskingStrategy, mask_ratio: float = 0.5):
        """Swap masking strategy/ratio and regenerate self._masks in place — e.g. for a
        mask-ratio curriculum (ramp a RandomMaskingStrategy up before switching to a fixed
        ComplementaryMaskingStrategy, since the latter structurally ignores any ratio
        argument, see IO/masking.py). Changes __len__ if the new strategy's `multiplier`
        differs from the old one (e.g. random 1x -> complementary 2x) — any DataLoader
        already built against this dataset must be rebuilt afterward, not just re-iterated:
        with persistent_workers=True, worker subprocesses hold their own copy of the
        dataset from when they were spawned and never see this mutation otherwise."""
        self.masking_strategy = masking_strategy
        self.mask_ratio = self.masking_strategy.effective_mask_ratio(mask_ratio)
        self._masks = [
            self.masking_strategy.generate_mask(self.base_dataset.Nc, self.num_patches, self.mask_ratio)
            for _ in range(len(self.base_dataset))
        ]

    def __len__(self):
        return len(self.base_dataset) * self.masking_strategy.multiplier

    def __getitem__(self, index):
        N = len(self.base_dataset)
        trial_idx, mask = self.masking_strategy.resolve(self._masks, index, N)

        x, y = self.base_dataset[trial_idx]
        x_patches, time_indices = slice_patches(x, self.patch_len)

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

def load_montage_channels(name: str) -> List[str]:
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


def resolve_canonical_channels(canonical_channels) -> List[str]:
    """canonical_channels: a montage name (str, looked up via load_montage_channels) or
    an already-literal channel-label list -> list. Centralizes the isinstance check every
    caller of preprocess_params.canonical_channels needs — taking len() of the raw string
    instead silently returns the name's character count (e.g. len("10-10") == 5), not an
    error."""
    if isinstance(canonical_channels, str):
        return load_montage_channels(canonical_channels)
    return canonical_channels


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
        if canonical:
            return resolve_canonical_channels(canonical)

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

        ds_transform = transform if transform is not None else build_normalizer_from_config(config_dict)
        ds_cache_suffix = cache_suffix(pp['sample_freq'], pp['bandpass_filter'])

        loader_config = {
            'dataset_params': ds_args,
            'data_metadata': data_metadata,
            'data_structure': data_structure
        }

        requested_subjects = ds_args['subject_to_use']
        if requested_subjects in (['all'], 'all'):
            all_ids = list(data_structure.keys())
            try:
                requested_subjects = sorted(all_ids, key=int)
            except ValueError:
                requested_subjects = sorted(all_ids)

        for sub_id in requested_subjects:
            loading_tasks.append({
                'dataset_name': ds_name,
                'subject_id': sub_id,
                'transform': ds_transform,
                'dataset_config': loader_config,
                'cache_suffix': ds_cache_suffix,
            })

    target_channels = _resolve_target_channels(dataset_params, pp=pp)

    fft_params = None

    if assemble_trials is None:
        assemble_trials = mode in ('pretrain', 'tokenizer')  # explicit override: e.g. real
        # per-trial labels for codebook diagnostics (pretrain/tokenizer normally assemble
        # trials into continuous-signal windows, which discards real labels -- see
        # IO/preprocessing.py's window_continuous_signal)
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
    elif mode == 'tokenizer':
        return TokenizerDataset(base_dataset, patch_len=patch_len)
    elif mode == 'pretrain':
        mask_pp       = pp.get('mask', {})
        strategy_name = mask_pp.get('masking_strategy', 'random')
        strategy_cfg  = mask_pp.get(strategy_name, {})

        if strategy_name == 'complementary':
            strategy   = ComplementaryMaskingStrategy()
            mask_ratio = ComplementaryMaskingStrategy.MASK_RATIO
        elif strategy_name == 'random_to_complementary':
            strategy = RandomToComplementaryMaskingStrategy(
                target_ratio=ComplementaryMaskingStrategy.MASK_RATIO,
                start_ratio=strategy_cfg.get('start_ratio', 0.1),
                ramp_epochs=strategy_cfg.get('ramp_epochs', 25),
                step_every=strategy_cfg.get('step_every', 5),
            )
            mask_ratio = strategy.effective_mask_ratio(ComplementaryMaskingStrategy.MASK_RATIO)
        else:
            strategy   = RandomMaskingStrategy()
            mask_ratio = strategy_cfg.get('mask_ratio', 0.5)

        return PretrainDataset(base_dataset, patch_len=patch_len,
                               mask_ratio=mask_ratio, masking_strategy=strategy)
    elif mode == 'finetune':
        return FinetuneDataset(base_dataset)
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Expected one of: base, tokenizer, pretrain, finetune.")
