import os
import json
import random
import zlib
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from typing import List, Dict, Optional, Tuple, Callable, Any

from .loader import load_coords_from_metadata, get_standard_coords
from IO.preprocessing import build_normalizer_from_config, cache_suffix, slice_patches, num_patches, window_continuous_signal
from IO.masking import MaskingStrategy, build_masking_strategy_from_config

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

HEAD_RADIUS_M = 0.095   # polar metadata coordinates are projected onto a sphere of this radius (real layout)


def channel_xyz(ch_info: Dict) -> Optional[np.ndarray]:
    """Real-layout template coordinate of one metadata channel (docs/adr/0023): MNE's standard_1020 position for the
    label, else standard_1005, else the polar topomap coordinates projected onto a head sphere (radius 0.5 = the
    equator), else None. A dataset's own 'xyz' (any frame) is read only by coords 'dataset' (own_xyz)."""
    p = get_standard_coords(ch_info.get('label', ''))
    if p is None:
        p = _standard_1005().get(ch_info.get('label', '').strip().lower())   # 10-05 names (FFC1h, TPP9h, AFp3h, ...)
    if p is not None:
        return np.asarray(p, dtype=np.float64)
    c = ch_info.get('coordinates')
    if c:
        th, el = np.deg2rad(c.get('polar_angle_deg', 0.0)), np.pi * c.get('polar_radius', 0.0)
        return HEAD_RADIUS_M * np.array([np.sin(el) * np.sin(th), np.sin(el) * np.cos(th), np.cos(el)])
    return None


def own_xyz(ch_info: Dict, equator_radius: Optional[float]) -> Optional[np.ndarray]:
    """preprocess_params.coords 'dataset' (docs/cards/2026-10-01-dataset-coordinates.md): the position the dataset
    itself records for this channel -- 'xyz' (any frame and unit; aligned later by align_similarity), else its polar
    table when the dataset states its convention (channels.polar_equator_radius, the radius of the equator: EEGLAB
    .loc 0.5, idealised tables e.g. 0.36 / 0.406), projected onto the HEAD_RADIUS_M sphere -- else None."""
    if isinstance(ch_info.get('xyz'), (list, tuple)):
        return np.asarray(ch_info['xyz'], dtype=np.float64)
    c = ch_info.get('coordinates')
    if c and equator_radius:
        th = np.deg2rad(c.get('polar_angle_deg', 0.0))
        el = 0.5 * np.pi * c.get('polar_radius', 0.0) / equator_radius          # angle from the vertex
        return HEAD_RADIUS_M * np.array([np.sin(el) * np.sin(th), np.sin(el) * np.cos(th), np.cos(el)])
    return None


def align_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Least-squares similarity transform (rotation, uniform scale, translation; no reflection -- Umeyama 1991) taking
    the points src [n, 3] onto dst [n, 3] -> (M, mean residual in dst units, t): x -> x @ M.T + t, M = scale * R."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()
    M = scale * R
    t = mu_d - mu_s @ M.T
    resid = float(np.linalg.norm(src @ M.T + t - dst, axis=1).mean())
    return M, resid, t


ALIGN_MAX_RESIDUAL_M = 0.015   # a dataset whose aligned named channels sit farther from the template falls back to it


_STD1005 = None


def _standard_1005() -> Dict[str, np.ndarray]:
    """MNE's standard_1005 positions by lower-case label (standard_1020 lacks the 10-05 half-step sites)."""
    global _STD1005
    if _STD1005 is None:
        import mne
        _STD1005 = {k.lower(): v for k, v in mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos'].items()}
    return _STD1005


IDW_K = 4   # > Nc channels: a missing canonical site = inverse-distance-squared mean of its k nearest good electrodes


def idw_matrix(pos_from: np.ndarray, pos_to: np.ndarray, k: int = IDW_K) -> np.ndarray:
    """Inverse-distance-weighted (1/d^2) interpolation from the k nearest electrodes -> [len(pos_to), len(pos_from)].
    Chosen over spherical splines (docs/cards/2026-09-30-real-coordinates.md check 3: on a 126-channel montage IDW had
    the lower mean and median error, and splines blew up on a noisy electrode)."""
    W = np.zeros((len(pos_to), len(pos_from)))
    for r, p in enumerate(pos_to):
        d2 = ((pos_from - p) ** 2).sum(1)
        nn = np.argsort(d2)[:k]
        w = 1.0 / np.maximum(d2[nn], 1e-12)
        W[r, nn] = w / w.sum()
    return W


# --- Base Dataset ---

def split_pretrain_subjects(dataset_params: Dict, train_fraction: float = 0.9, seed: int = 42) -> Dict[str, Tuple[List[str], List[str]]]:
    """Pretrain subject train/val split -> {dataset: (train subjects, val subjects)}.
    Person-disjoint: datasets recorded from the same people share a cohort
    (metadata.json data_metadata.cohort, default the dataset name; same subject id = same
    person within a cohort), and a person is on one side for every dataset of the cohort.
    Order-independent: every cohort draws from its own RNG (seed, cohort), so adding, removing
    or reordering datasets leaves the other splits unchanged."""
    subjects, cohort_of = {}, {}
    for ds, args in dataset_params.items():
        meta = json.load(open(os.path.join(args['dataset_path'], 'metadata.json'), encoding='utf-8'))
        avail = sorted(meta.get('data_structure', {}).keys())
        req = args['subject_to_use']
        subjects[ds] = avail if req in (['all'], 'all') else [s for s in avail if s in {str(r) for r in req}]
        cohort_of[ds] = meta.get('data_metadata', {}).get('cohort', ds)
    out = {}
    for cohort in sorted(set(cohort_of.values())):
        members = [ds for ds in dataset_params if cohort_of[ds] == cohort]
        persons = sorted({s for ds in members for s in subjects[ds]})
        random.Random(zlib.crc32(f'{seed}/{cohort}'.encode())).shuffle(persons)
        n_train = int(len(persons) * train_fraction)
        if n_train == len(persons) and len(persons) > 1:
            n_train -= 1
        n_train = max(n_train, 1) if persons else 0
        train = set(persons[:n_train])
        for ds in members:
            out[ds] = ([s for s in subjects[ds] if s in train], [s for s in subjects[ds] if s not in train])
    return out


# pretrain: a channel with std below this x the subject's median channel std is treated as padding. 0.10 catches
# dead electrodes and the recording reference (e.g. Cz in the Tsinghua SSVEP sets, mastoids): 309 subject-channels
# over the corpus on 2026-09-26, versus 88 at 0.05 (only the near-zero ones).
FLAT_RATIO = 0.10


class EEGDataset(Dataset):
    """
    Loads EEG data from multiple subjects/datasets into a unified tensor.
    When assemble_trials=True, flattens each subject's trials into a continuous
    signal and cuts non-overlapping windows of assembly_params['window_length'].
    """
    def __init__(
        self,
        config: Dict,
        loading_tasks: List[Dict[str, Any]],
        desired_channels: List[str],
        assemble_trials: bool = False,
        assembly_params: Optional[Dict] = None
    ):
        self.config = config
        self.channel_names = desired_channels
        self.Nc = len(desired_channels)
        # 'grid' (every channel matched to its canonical 10-10 slot by name, the rest dropped) or 'real' (every EEG
        # channel kept with real coordinates; > Nc channels reduced to the canonical sites), docs/adr/0023.
        self.channel_layout = config.get('preprocess_params', {}).get('channel_layout', 'grid')
        if self.channel_layout not in ('grid', 'real'):
            raise ValueError(f"preprocess_params.channel_layout must be grid|real, got {self.channel_layout!r}")
        # 'template' (MNE's position for each channel name) or 'dataset' (real layout only: the dataset's own recorded
        # positions, aligned to MNE's head frame; template where it has none), docs/cards/2026-10-01-dataset-coordinates.md
        self.coords_source = config.get('preprocess_params', {}).get('coords', 'template')
        if self.coords_source not in ('template', 'dataset'):
            raise ValueError(f"preprocess_params.coords must be template|dataset, got {self.coords_source!r}")
        if self.coords_source == 'dataset' and self.channel_layout != 'real':
            raise ValueError("preprocess_params.coords 'dataset' needs channel_layout 'real'")
        self.coord_report = {}
        self._plans = {}
        self.assemble_trials = assemble_trials
        self.assembly_params = assembly_params or {}

        all_data_chunks: List[torch.Tensor] = []
        all_label_chunks: List[torch.Tensor] = []
        all_subject_chunks: List[torch.Tensor] = []
        all_dataset_names: List[str] = []
        all_coords: List[torch.Tensor] = []
        all_valid_channels: List[torch.Tensor] = []  # per-task [Nc] bool, True = real (not zero-padded) channel
        all_named_slots: List[torch.Tensor] = []     # per-task [Nc] bool, True = the slot holds its canonical-name channel
        all_valid_length: List[int] = []             # per-task original T, before cross-subject max_T padding
        all_row_valid_start: List[torch.Tensor] = []  # per-task [N_rows] real-content start, per row
        all_row_valid_end: List[torch.Tensor] = []    # per-task [N_rows] real-content end, per row

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
            all_named_slots.append(result['named_slots'])
            all_valid_length.append(result['valid_length'])
            all_row_valid_start.append(result['row_valid_start'])
            all_row_valid_end.append(result['row_valid_end'])

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
        # Which valid slots hold the channel their canonical name says: everything valid under 'grid'; under 'real'
        # a non-grid channel sits in a free slot. Name-based logic (channel subsampler, backbone_eval's motor-3 ->
        # bci-22 test) reads only these.
        self.all_named_slots = all_named_slots
        self.all_valid_length = all_valid_length
        # Per-ROW (not per-task, unlike all_valid_length above): indexed the same way
        # self.labels/self.data rows are, since real-content bounds vary WITHIN a task
        # (an assembled window's tail pad, or an event-anchored trial's own pre/post pad
        # near a recording edge). data[:, :, row_valid_start[i]:row_valid_end[i]] is real
        # content for row i; everything outside is zero pad. Used by PretrainDataset to
        # keep masking from picking patches out of padded content.
        self.row_valid_start = torch.cat(all_row_valid_start)
        self.row_valid_end = torch.cat(all_row_valid_end)

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

        plan = None
        if self.channel_layout == 'real':
            sub_xyz = None
            if self.coords_source == 'dataset':      # per-subject digitized positions (data_structure.<id>.channel_xyz)
                sub_xyz = (ds_config.get('data_structure', {}).get(str(subject_id)) or {}).get('channel_xyz')
            key = (ds_name, str(subject_id)) if sub_xyz else ds_name
            if key not in self._plans:
                self._plans[key] = self._real_plan(desired_channels, ds_config, sub_xyz, name='/'.join(
                    [ds_name] + ([str(subject_id)] if sub_xyz else [])))
            plan = self._plans[key]
            ds_indices, target_pos = plan['src'], plan['slots']
        else:
            ds_indices, target_pos = self._map_channels(desired_channels, ds_config['data_metadata']['channels'])

        # Train-time read: compiled cache only (see cache_dataset.py) — dataset-specific
        # loading code (datas/<Name>/loader.py) never runs at train time. The cached
        # array holds ALL native channels in metadata.json's index order (compile time
        # keeps every channel, no target-channel subsetting), so ds_indices indexes
        # directly into it, no extra mapping needed.
        dataset_path = ds_config['dataset_params']['dataset_path']
        cache_path = os.path.join(dataset_path, 'cache', f"{subject_id}_{task['cache_suffix']}.npz")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No compiled cache at {cache_path}. Run "
                f"`python cache_dataset.py --config configs/compile.json` first "
                f"(and make sure compile.json's sample_freq/bandpass_filter match "
                f"this config's preprocess_params)."
            )
        npz = np.load(cache_path)
        data_np = npz['data'][:, ds_indices, :]
        if data_np.shape[0] == 0:
            return None
        coords_np = plan['coords'] if plan else load_coords_from_metadata(ds_config['data_metadata'], ds_indices)
        src_names = plan['labels'] if plan else [desired_channels[p] for p in target_pos]

        raw_data = torch.from_numpy(data_np.astype(np.float32))  # (N, C, T)
        N, _, T = raw_data.shape
        # Pretrain only: a channel far flatter than the subject's others (std < FLAT_RATIO x the
        # median channel std, measured BEFORE the per-trial z-score, which would blow its noise
        # up to unit variance) is a dead electrode or the reference -- treated as padding below.
        flat = torch.zeros(len(ds_indices), dtype=torch.bool)
        if self.assemble_trials:
            ch_std = raw_data.transpose(0, 1).reshape(len(ds_indices), -1).std(dim=1)
            flat = ch_std < FLAT_RATIO * ch_std.median()
            if flat.any():
                print(f"  [{ds_name} S{subject_id}] flat channels -> padding: "
                      f"{[src_names[i] for i in flat.nonzero().flatten().tolist()]}")
        named = plan['named'] if plan else [True] * len(target_pos)
        if plan and plan['interp']:
            # > Nc channels: every canonical site, copied where the recording has it (and it is not flat),
            # IDW-interpolated from its good electrodes where not. Dead electrodes are left out of the interpolation.
            W, coords_np = self._interp_matrix(plan, ~flat.numpy(), desired_channels)
            raw_data = torch.einsum('sc,nct->nst', torch.from_numpy(W).float(), raw_data)
            target_pos, named = list(range(self.Nc)), [True] * self.Nc
            flat = torch.zeros(self.Nc, dtype=torch.bool)
        # Per-trial real-content bounds, compiled-rate samples (see cache_dataset.py /
        # IO/loader.py's get_subject_data) -- 'valid_start'/'valid_end' absent (an older
        # cache from before this existed) means "every trial fully real", same default
        # get_subject_data itself uses.
        cache_valid_start = npz['valid_start'].tolist() if 'valid_start' in npz else [0] * N
        cache_valid_end = npz['valid_end'].tolist() if 'valid_end' in npz else [T] * N

        # Transform (bandpass/resample/normalize) on the REAL channels only, before padding —
        # normalizing after zero-padding folds the zero-filled missing-channel rows into the
        # trial's mean/std (see IO/preprocessing.py's per-trial zscore/robust normalize), which
        # skews scale differently per dataset depending how many channels it's missing relative
        # to canonical_channels (e.g. BNCI2014001 is ~2/3 zero-padded channels — that's a much
        # bigger normalization bias than a near-complete dataset like PhysionetMI).
        if transform is not None:
            # Whole (N, C, T) batch in one call -- Normalizer._normalize reduces
            # per-trial (its own mean/std/median, never pooled across N), so this is
            # numerically identical to the old torch.stack([transform(raw_data[i])
            # for i in range(N)]) loop, just without holding N separately-normalized
            # tensors in a Python list before the stack copies them into one buffer
            # (see IO/preprocessing.py's Normalizer docstring).
            raw_data = transform(raw_data)
        post_transform_T = raw_data.shape[-1]  # real (non-padded) length for finetune's per-trial mask

        padded = torch.zeros((N, self.Nc, post_transform_T), dtype=torch.float32)
        padded[:, target_pos, :] = raw_data

        if self.assemble_trials:
            target_L = self.assembly_params.get('window_length', padded.shape[-1])
            padded, labels, row_valid_start, row_valid_end = window_continuous_signal(
                padded, target_L, ds_name, subject_id,
                valid_ranges=list(zip(cache_valid_start, cache_valid_end)),
                min_real_fraction=self.assembly_params.get('window_min_real', 0.5))
            # preprocess_params.window_fraction (pretrain only): keep this fraction of the
            # subject's windows -- the first n_keep of one fixed permutation seeded by
            # (window_fraction_seed, dataset, subject), so every subject stays in the corpus
            # and, at one seed, a smaller fraction is a subset of a larger one (corpus
            # sizes tiny 5% < small 20% < medium 50% < large 100%, 2026-09-24). A dataset's own
            # dataset_params.pretrain.<ds>.window_fraction multiplies in: the corpus builder balances
            # paradigms by capping windows per subject, never by dropping subjects -- the product
            # keeps the nesting across corpus sizes.
            frac = self.assembly_params.get('window_fraction', 1.0) * ds_config['dataset_params'].get('window_fraction', 1.0)
            if frac < 1.0 and len(padded):
                seed = zlib.crc32(f"{self.assembly_params.get('window_fraction_seed', 0)}/{ds_name}/{subject_id}".encode())
                n_keep = max(1, int(round(frac * len(padded))))
                keep = torch.from_numpy(np.sort(np.random.default_rng(seed).permutation(len(padded))[:n_keep]))
                padded, labels = padded[keep], labels[keep]
                row_valid_start, row_valid_end = row_valid_start[keep], row_valid_end[keep]
        else:
            labels = torch.from_numpy(npz['labels'].astype(np.int64))
            # Every row here is one real, untouched-by-assembly trial -- use the cache's
            # own per-trial bounds directly (cross-subject max_T padding, applied later in
            # EEGDataset.__init__, is handled separately via all_valid_length/FinetuneDataset).
            row_valid_start = torch.tensor(cache_valid_start, dtype=torch.long)
            row_valid_end = torch.tensor(cache_valid_end, dtype=torch.long)

        task_coords = torch.zeros((self.Nc, 3), dtype=torch.float32)
        task_coords[target_pos] = torch.from_numpy(coords_np.astype(np.float32))

        valid_channels = torch.zeros(self.Nc, dtype=torch.bool)
        valid_channels[target_pos] = True
        if flat.any():
            dead = [target_pos[i] for i in flat.nonzero().flatten().tolist()]
            valid_channels[dead] = False
            padded[:, dead] = 0
            task_coords[dead] = 0
        named_slots = torch.zeros(self.Nc, dtype=torch.bool)
        named_slots[[p for p, n in zip(target_pos, named) if n]] = True
        named_slots &= valid_channels

        return {
            'data': padded,
            'labels': labels,
            'dataset_name': ds_name,
            'subject_id': subject_id,
            'coords': task_coords,
            'valid_channels': valid_channels,
            'named_slots': named_slots,
            'valid_length': post_transform_T,
            'row_valid_start': row_valid_start,
            'row_valid_end': row_valid_end,
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

    def _real_plan(self, desired_channels: List[str], ds_config: Dict, sub_xyz=None, name: str = '') -> Dict:
        """channel_layout 'real' (docs/adr/0023): which cached channels to read and where they go. A channel whose
        label (with the 10-20 aliases) is a canonical name keeps that slot, exactly as under 'grid'; every other EEG
        channel with a known position fills a free slot. More than Nc channels -> 'interp': all of them feed
        _interp_matrix. -> {src, slots, labels, coords, named, interp, pos}."""
        chans = ds_config['data_metadata']['channels']
        include_non_eeg = ds_config['dataset_params'].get('include_non_eeg_channels', False)
        keys = sorted((k for k in chans if isinstance(k, str) and k.isdigit()), key=int)
        slot_of = {self._normalize_label(n): s for s, n in enumerate(desired_channels)}
        src, labels, pos, slot, dropped = [], [], [], [], []
        for k in keys:
            info = chans[k]
            label = info.get('label', '') if isinstance(info, dict) else ''
            s = slot_of.get(self._normalize_label(label))
            if s is not None and s in slot:
                s = None                                       # two labels for one site (e.g. T3 and T7)
            if s is None and not include_non_eeg and label.upper() in NON_EEG_CHANNELS:
                continue
            p = channel_xyz(info) if isinstance(info, dict) else None
            if p is None:
                dropped.append(label)
                continue
            src.append(int(k) - 1); labels.append(label); pos.append(p); slot.append(s)
        if dropped:
            print(f"  [channel map] no position, dropped: {dropped}")
        if self.coords_source == 'dataset':
            pos = self._dataset_positions(chans, src, labels, pos, sub_xyz, name)
        # canonical-slot order first (as 'grid' reads them, so a canonical-only dataset normalises bit-identically:
        # the per-trial z-score sums channels in read order), the other channels after in metadata order
        order = sorted(range(len(src)), key=lambda j: (slot[j] is None, slot[j] if slot[j] is not None else j))
        src, labels, pos, slot = ([v[j] for j in order] for v in (src, labels, pos, slot))
        named = [s is not None for s in slot]
        pos = np.array(pos)
        interp = len(src) > self.Nc
        if not interp:
            free = iter(s for s in range(self.Nc) if s not in slot)
            slot = [s if s is not None else next(free) for s in slot]
        print(f"  [channel map real] {len(src)} EEG channels, {sum(named)} on canonical slots"
              + (f", > {self.Nc}: interpolated to the canonical sites" if interp else ""))
        return {'src': src, 'slots': slot, 'labels': labels, 'coords': pos.astype(np.float32), 'named': named,
                'interp': interp, 'pos': pos}

    def _dataset_positions(self, chans: Dict, src: List[int], labels: List[str], pos: List[np.ndarray], sub_xyz,
                           name: str) -> List[np.ndarray]:
        """coords 'dataset': each channel's own recorded position (own_xyz, or the subject's channel_xyz row), all of a
        dataset's (or subject's) own positions mapped onto MNE's head frame by one similarity transform fitted on its
        channels that also have a template position; channels without an own position keep the template one. A fit
        residual above ALIGN_MAX_RESIDUAL_M -> the whole dataset keeps the template (reported)."""
        eq = chans.get('polar_equator_radius')
        own = []
        for i, lab in zip(src, labels):
            info = chans.get(str(i + 1), {})
            p = np.asarray(sub_xyz[i], dtype=np.float64) if sub_xyz else own_xyz(info, eq)
            own.append(p)
        tmpl = [channel_xyz({'label': lab}) for lab in labels]
        fit = [j for j, (o, t) in enumerate(zip(own, tmpl)) if o is not None and t is not None]
        n_own = sum(o is not None for o in own)
        if n_own == 0:
            return pos
        if len(fit) < 4:
            self.coord_report[name] = {'own': n_own, 'fit': len(fit), 'residual_mm': None, 'used': False}
            print(f"  [coords dataset] {name}: {n_own} own positions but {len(fit)} named for the fit -> template")
            return pos
        M, resid, t = align_similarity(np.array([own[j] for j in fit]), np.array([tmpl[j] for j in fit]))
        used = resid <= ALIGN_MAX_RESIDUAL_M
        self.coord_report[name] = {'own': n_own, 'fit': len(fit), 'residual_mm': resid * 1000, 'used': used,
                                   'scale': float(np.cbrt(abs(np.linalg.det(M))))}
        print(f"  [coords dataset] {name}: {n_own} own positions, aligned on {len(fit)} named, mean residual "
              f"{resid * 1000:.1f} mm" + ("" if used else " > limit -> template"))
        if not used:
            return pos
        return [o @ M.T + t if o is not None else p for o, p in zip(own, pos)]

    def _interp_matrix(self, plan: Dict, good: np.ndarray, desired_channels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """> Nc channels -> (W [Nc, n_src] mapping the recording to the Nc canonical sites, coords [Nc, 3]). A site
        the recording has (and whose electrode is good) is copied (one-hot row, the electrode's own position); every
        other site is an inverse-distance-weighted row (idw_matrix) from the good electrodes, at the site's standard
        position."""
        n = len(plan['src'])
        W = np.zeros((self.Nc, n))
        coords = np.zeros((self.Nc, 3))
        have = {s: j for j, s in enumerate(plan['slots']) if s is not None and good[j]}
        missing = [s for s in range(self.Nc) if s not in have]
        for s, j in have.items():
            W[s, j], coords[s] = 1.0, plan['pos'][j]
        if missing:
            std = np.array([get_standard_coords(desired_channels[s]) for s in missing], dtype=np.float64)
            gi = np.flatnonzero(good)
            W[np.ix_(missing, gi)] = idw_matrix(plan['pos'][gi], std)
            coords[missing] = std
        return W, coords.astype(np.float32)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


# --- Sanity checks ---

def sanity_check_base(base_dataset: 'EEGDataset') -> None:
    """
    Verifies EEGDataset's parallel per-trial/per-task arrays stayed aligned
    through loading/padding/windowing. Call once after construction, before
    training starts — cheap (no data copies), catches an alignment bug (wrong
    label/coords/subject_id paired with a trial) that would otherwise silently
    corrupt training instead of crashing anywhere obvious.
    """
    n = len(base_dataset)
    if n == 0:
        raise ValueError("EEGDataset has 0 trials.")

    for name, arr in (('labels', base_dataset.labels), ('subject_data', base_dataset.subject_data),
                       ('dataset_names', base_dataset.dataset_names),
                       ('trial_to_coords_idx', base_dataset.trial_to_coords_idx)):
        if len(arr) != n:
            raise ValueError(f"EEGDataset.{name} has length {len(arr)}, expected {n} (== len(data)).")

    n_tasks = len(base_dataset.all_coords)
    for name, arr in (('all_valid_channels', base_dataset.all_valid_channels),
                       ('all_valid_length', base_dataset.all_valid_length)):
        if len(arr) != n_tasks:
            raise ValueError(f"EEGDataset.{name} has length {len(arr)}, expected {n_tasks} (== len(all_coords)).")

    max_idx = max(base_dataset.trial_to_coords_idx)
    if max_idx >= n_tasks:
        raise ValueError(f"trial_to_coords_idx references task {max_idx}, but only {n_tasks} tasks were loaded.")

    for coords in base_dataset.all_coords:
        if coords.shape != (base_dataset.Nc, 3):
            raise ValueError(f"A coords entry has shape {tuple(coords.shape)}, expected ({base_dataset.Nc}, 3).")
    for vc in base_dataset.all_valid_channels:
        if vc.shape != (base_dataset.Nc,):
            raise ValueError(f"A valid_channels entry has shape {tuple(vc.shape)}, expected ({base_dataset.Nc},).")

    # dataset_names and trial_to_coords_idx are built in the same per-task loop
    # (EEGDataset.__init__), so every trial mapped to task index T must report the
    # same dataset_name every time it recurs -- a scramble (wrong zip/append order)
    # breaks this even though the plain length checks above still pass.
    task_dataset_name: Dict[int, str] = {}
    for i in range(n):
        task_idx = base_dataset.trial_to_coords_idx[i]
        name = base_dataset.dataset_names[i]
        seen = task_dataset_name.setdefault(task_idx, name)
        if seen != name:
            raise ValueError(
                f"Trial {i}: dataset_name {name!r} disagrees with an earlier trial mapped "
                f"to the same task index {task_idx} ({seen!r}) — trial_to_coords_idx and "
                f"dataset_names are out of sync."
            )

    if not torch.isfinite(base_dataset.data).all():
        raise ValueError("EEGDataset.data contains NaN/Inf — check upstream cache/normalize.")

    print(f"[sanity check] EEGDataset: {n} trials, {n_tasks} tasks — "
          f"data/labels/coords/subject/dataset_name alignment OK.")


def sanity_check_wrapper(dataset) -> None:
    """
    Pulls the first and last item through __getitem__ on the actual training
    dataset (PretrainDataset/FinetuneDataset) — catches an
    index-mapping bug in the wrapper itself (e.g. a masking-strategy resolve()
    or trial_to_coords_idx lookup gone wrong) before training starts, rather
    than a cryptic mid-epoch crash or, worse, silently wrong data with no
    crash at all.
    """
    n = len(dataset)
    if n == 0:
        raise ValueError(f"{type(dataset).__name__} has 0 items.")
    for idx in sorted({0, n - 1}):
        item = dataset[idx]
        for t in item:
            if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
                raise ValueError(f"{type(dataset).__name__}[{idx}] contains NaN/Inf.")
    print(f"[sanity check] {type(dataset).__name__}: {n} items, first/last __getitem__ OK.")


# --- Dataset Wrappers ---

def _resolve_default_patch_len(base_dataset: 'EEGDataset') -> int:
    model_type = base_dataset.config.get('training_params', {}).get('pretrain', {}).get('model_type', 'MeSAE')
    preprocess = base_dataset.config.get('model_params', {}).get(model_type, {}).get('preprocess', {})
    return preprocess.get('patch_length', 200)


class PretrainDataset(Dataset):
    """
    Wraps EEGDataset for masked pretraining.
    Yields: (x_patches, coords, mask, time_indices, label, valid_channels)
      x_patches:      [C, P, L]
      coords:         [C, 3]
      mask:           [C * P] bool
      time_indices:   [P]
      label:          scalar
      valid_channels: [C] bool, True = real (not zero-padded) channel
    """
    def __init__(
        self,
        base_dataset: EEGDataset,
        patch_len: Optional[int] = None,
        patch_stride: Optional[int] = None,
        masking_strategy: Optional[MaskingStrategy] = None,
    ):
        self.base_dataset = base_dataset
        if patch_len is None:
            patch_len = _resolve_default_patch_len(base_dataset)
        self.patch_len = patch_len
        self.patch_stride = patch_stride or patch_len
        self.num_patches = num_patches(base_dataset.data.shape[-1], self.patch_len, self.patch_stride)

        # Per-trial (channel, patch) validity -- excludes zero-padded channels AND, for an
        # assembled window's zero tail (window_continuous_signal), patches that start past
        # the window's real content. Fixed after construction; every mask is drawn inside it.
        self._valid_masks = self._build_valid_masks()
        self.set_masking(masking_strategy or build_masking_strategy_from_config({}))

        n = len(base_dataset)
        print(f"\n--- PretrainDataset ---")
        print(f"  {n} trials | {self.num_patches} patches/trial (patch_len={self.patch_len}, "
              f"patch_stride={self.patch_stride}) | {self.masking_strategy.describe()}")
        print(f"----------------------------\n")

    def _build_valid_masks(self):
        """Per-trial [Nc*num_patches] bool -- True at (channel, patch) positions that are
        BOTH a real (not zero-padded) channel and a genuinely-real patch position (a
        patch whose start sample falls within [row_valid_start, row_valid_end) -- outside
        that, whether an assembled window's tail pad or an event-anchored trial's own
        leading/trailing pad near a recording edge, it's padding, see
        IO/preprocessing.py's window_continuous_signal / cut_event_window). Masks are never
        drawn outside it, so masking never spends its budget on content that's already
        known-zero."""
        bd = self.base_dataset
        patch_starts = torch.arange(self.num_patches) * self.patch_stride  # [P]
        masks = []
        for trial_idx in range(len(bd)):
            coords_idx = bd.trial_to_coords_idx[trial_idx]
            valid_channels = bd.all_valid_channels[coords_idx]           # [Nc] bool
            valid_patch = (patch_starts >= bd.row_valid_start[trial_idx]) \
                & (patch_starts < bd.row_valid_end[trial_idx])           # [P] bool
            masks.append((valid_channels.unsqueeze(1) & valid_patch.unsqueeze(0)).reshape(-1))
        return masks

    def set_masking(self, masking_strategy: MaskingStrategy):
        """(Re)draw every trial's mask ([C*N] bool) from the strategy's current epoch (the
        training loop calls this every masked epoch, see train_pretrain.py). Any DataLoader
        built on this dataset must be rebuilt afterwards (with persistent_workers, workers hold
        their own copy of the dataset from spawn time).

        masking_strategy.subsampler (ChannelSubsampler, optional): per trial, a sparse montage
        to keep; removed channels leave the valid set BEFORE the mask is drawn, and
        __getitem__ turns them into padding (zero signal, not valid, no loss)."""
        self.masking_strategy = masking_strategy
        subsampler = masking_strategy.subsampler
        bd = self.base_dataset
        if subsampler is not None and not hasattr(self, '_montage_idx'):
            norm = bd._normalize_label
            name_to_idx = {norm(n): i for i, n in enumerate(bd.channel_names)}
            self._montage_idx = [[name_to_idx[norm(n)] for n in load_montage_channels(m) if norm(n) in name_to_idx]
                                 for m in subsampler.montages]
        self._keep, self._masks = [], []
        for i in range(len(bd)):
            valid, keep = self._valid_masks[i], None
            if subsampler is not None:
                ti = bd.trial_to_coords_idx[i]
                keep = subsampler.sample(bd.all_valid_channels[ti] & bd.all_named_slots[ti], self._montage_idx)
                if keep is not None:
                    valid = valid & keep.repeat_interleave(self.num_patches)
            self._keep.append(keep)
            coords = bd.all_coords[bd.trial_to_coords_idx[i]]
            self._masks.append(masking_strategy.generate(bd.Nc, self.num_patches, valid, coords,
                                                          subsampled=keep is not None))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, trial_idx):
        mask = self._masks[trial_idx]

        x, y = self.base_dataset[trial_idx]
        x_patches, time_indices = slice_patches(x, self.patch_len, self.patch_stride)

        coords_idx = self.base_dataset.trial_to_coords_idx[trial_idx]
        coords     = self.base_dataset.all_coords[coords_idx]
        valid_channels = self.base_dataset.all_valid_channels[coords_idx]
        keep = self._keep[trial_idx]
        if keep is not None:                       # subsampled window: removed channels become padding
            x_patches = x_patches * keep[:, None, None]
            coords = coords * keep[:, None]
            valid_channels = valid_channels & keep

        return x_patches, coords, mask, time_indices, y, valid_channels


class MontageBatchSampler(Sampler):
    """Batches of one montage (real-channel set) each, so a batch can drop the canonical
    channels none of its windows has (train_pretrain.py's _unpack_batch): most datasets fill
    only part of the 64-channel montage (Dreyer2023 27), and padded channels cost as much
    compute as real ones. Windows are shuffled within each montage, cut into batches, and
    the batch order is shuffled (shuffle=False: montages in order, windows in order)."""
    def __init__(self, dataset: PretrainDataset, batch_size: int, shuffle: bool):
        bd = dataset.base_dataset
        groups: Dict[tuple, List[int]] = {}
        for i in range(len(bd)):
            key = tuple(bd.all_valid_channels[bd.trial_to_coords_idx[i]].nonzero().flatten().tolist())
            groups.setdefault(key, []).append(i)
        self.groups, self.batch_size, self.shuffle = list(groups.values()), batch_size, shuffle

    def _batches(self):
        out = []
        for g in self.groups:
            g = [g[j] for j in torch.randperm(len(g)).tolist()] if self.shuffle else g
            out += [g[k:k + self.batch_size] for k in range(0, len(g), self.batch_size)]
        return [out[j] for j in torch.randperm(len(out)).tolist()] if self.shuffle else out

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return sum(-(-len(g) // self.batch_size) for g in self.groups)


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
    """Looks up a named montage's ordered channel-label list from configs/montages.json —
    see docs/agents/adding-a-montage.md. Coordinates in that file are reference/QA data
    only (per-dataset coords are still resolved independently in IO/loader.py); only the
    label order matters here."""
    montage_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'montages.json')
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
    configs/montages.json (see docs/agents/adding-a-montage.md).
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
    patch_stride   = pp.get('patch_stride', patch_len)

    loading_tasks = []
    for ds_name, ds_args in dataset_params.items():
        meta_path = os.path.join(ds_args['dataset_path'], 'metadata.json')
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        data_metadata = metadata.get('data_metadata', {})
        data_structure = metadata.get('data_structure', {})

        ds_transform = transform if transform is not None else build_normalizer_from_config(config_dict)
        ds_cache_suffix = cache_suffix(pp['sample_freq'], pp['bandpass_filter'],
                                        pp.get('pre_event_seconds', 0.0), pp.get('post_event_seconds', 0.0))

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

    if assemble_trials is None:
        assemble_trials = mode == 'pretrain'  # explicit override: e.g. real
        # per-trial labels for codebook diagnostics (pretrain/tokenizer normally assemble
        # trials into continuous-signal windows, which discards real labels -- see
        # IO/preprocessing.py's window_continuous_signal)
    assembly_params = pp

    base_dataset = EEGDataset(
        config=config_dict,
        loading_tasks=loading_tasks,
        desired_channels=target_channels,
        assemble_trials=assemble_trials,
        assembly_params=assembly_params
    )
    sanity_check_base(base_dataset)

    if mode == 'base':
        return base_dataset
    elif mode == 'pretrain':
        # The tokenizer phase ignores masks; train_pretrain.py re-applies the strategy (with
        # its epoch) from the first masked epoch on.
        strategy = build_masking_strategy_from_config(pp.get('mask', {}))
        ds = PretrainDataset(base_dataset, patch_len=patch_len, patch_stride=patch_stride,
                             masking_strategy=strategy)
        sanity_check_wrapper(ds)
        return ds
    elif mode == 'finetune':
        ds = FinetuneDataset(base_dataset)
        sanity_check_wrapper(ds)
        return ds
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Expected one of: base, pretrain, finetune.")
