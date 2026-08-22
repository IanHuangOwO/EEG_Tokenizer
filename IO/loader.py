import os
import numpy as np
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
        return get_standard_coords(ch_name)

    def _load_coords_from_metadata(self) -> np.ndarray:
        """Default coord loader — see module-level load_coords_from_metadata()."""
        return load_coords_from_metadata(self.data_metadata, self.channel_indices)

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


def get_standard_coords(ch_name: str) -> Optional[np.ndarray]:
    try:
        import mne
        if not hasattr(get_standard_coords, '_positions'):
            montage = mne.channels.make_standard_montage('standard_1020')
            get_standard_coords._positions = montage.get_positions()['ch_pos']
        positions = get_standard_coords._positions
        keys = {k.upper(): k for k in positions.keys()}
        if ch_name.upper() in keys:
            return positions[keys[ch_name.upper()]]
    except Exception:
        pass
    return None


def load_coords_from_metadata(data_metadata: Dict, channel_indices: List[int]) -> np.ndarray:
    """
    Default coord loader — tries MNE standard_1020 first, then falls back to
    polar coords from metadata. Returns zeros for unknown channels. Shared by
    BaseSubjectLoader (compile-time per-dataset loaders) and IO/dataset.py's
    cache-read path (train time) — same channel-coordinate contract either way.
    """
    channel_config = data_metadata.get('channels', {})
    sorted_keys = sorted(
        [k for k in channel_config.keys() if isinstance(k, str) and k.isdigit()],
        key=lambda k: int(k)
    )
    coords_list = []
    for idx in channel_indices:
        ch_info = channel_config.get(sorted_keys[idx]) if idx < len(sorted_keys) else None
        label = ch_info.get('label', 'Unknown') if isinstance(ch_info, dict) else 'Unknown'

        mne_coords = get_standard_coords(label)
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


def resolve_dataset_loader(dataset_path: str):
    """Dynamically imports datas/<Name>/loader.py's `Loader` class — used only
    by cache_compile.py; train-time reads the compiled cache directly in
    IO/dataset.py, no loader class involved. Directory presence of loader.py
    IS the registration; there is no separate registry to keep in sync."""
    import importlib.util
    mod_path = os.path.join(dataset_path, 'loader.py')
    if not os.path.exists(mod_path):
        raise FileNotFoundError(f"No {mod_path} — see docs/agents/adding-a-dataset.md Step 6.")
    spec = importlib.util.spec_from_file_location(f'dataset_loader_{os.path.basename(dataset_path)}', mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Loader
