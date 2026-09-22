import os
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any

from IO.preprocessing import cut_event_window


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
        # Global compile-time setting (config/compile.json's compile_params, passed into
        # every loader's dataset_params by cache_dataset.py -- same mechanism dataset_path
        # already uses). 0.0 default = old behavior (window starts exactly at the event,
        # zero-length post) for every loader that hasn't opted in. Only meaningful for a
        # trigger/annotation-anchored loader that actually reads it (BCICIV1_Train,
        # BNCI2014001, Inria_Train as of this comment) -- see docs/model-analysis-checklist.md
        # for the per-dataset headroom measured before enabling this. In NATIVE sample-rate
        # samples wherever a loader converts to pts (self.sample_freq, not the compile
        # target) -- cache_dataset.py rescales the resulting valid_ranges to the compiled
        # rate itself, see get_subject_data.
        self.pre_event_seconds = self.dataset_params.get('pre_event_seconds', 0.0)
        self.post_event_seconds = self.dataset_params.get('post_event_seconds', 0.0)
        # Set by a loader's _load_data() (via _segment_by_annotations or its own cutting) to
        # a list of N (start, end) NATIVE-rate-sample pairs, one per trial this subject
        # produced, marking real (non-padded) content -- see IO/preprocessing.py's
        # cut_event_window. None (default, and every loader that never sets this) means
        # "every trial fully real", read by get_subject_data below.
        self._last_valid_ranges = None

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

    def _segment_by_annotations(
        self, raw, code_to_label: Dict[str, int], channel_indices: List[int],
        pre_event_pts: int, post_event_pts: int,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Cuts [C, pre_event_pts+post_event_pts] windows around each MNE annotation whose
        description is a key of code_to_label, via cut_event_window (always pads a
        trial that runs off the recording's start/end, never drops one -- see that
        function's docstring; this replaces the old drop-if-insufficient-headroom
        behavior). Shared by the GDF/EDF event-marker loaders (PhysionetMI, BNCI2014001,
        BNCI2014004) -- sets self._last_valid_ranges (NATIVE-rate samples) alongside the
        returned trials/labels so get_subject_data can carry real-vs-padded content
        through to the compiled cache. Measured real pre-event headroom per dataset
        (min ~3.5s+ in the loaders that use this) is in
        docs/model-analysis-checklist.md.
        """
        import mne
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        label_for_code = {event_id[k]: v for k, v in code_to_label.items() if k in event_id}
        if not label_for_code:
            return [], []

        data_np = raw.get_data(picks=channel_indices)
        trials, labels, valid_ranges = [], [], []
        for event_pts, _, code in events:
            label = label_for_code.get(code, -1)
            if label == -1:
                continue
            window, vs, ve = cut_event_window(data_np, event_pts, pre_event_pts, post_event_pts)
            if ve <= vs:
                continue  # entirely outside the recording -- degenerate, see cut_event_window
            trials.append(window)
            labels.append(label)
            valid_ranges.append((vs, ve))
        self._last_valid_ranges = valid_ranges
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
        self._last_valid_ranges = None  # reset -- only a loader that calls
                                         # _segment_by_annotations (or sets it itself)
                                         # below overrides this before returning
        data, labels = self._load_data()
        if data is None:
            return None
        n = len(data)
        valid_ranges = self._last_valid_ranges if self._last_valid_ranges is not None \
            else [(0, data.shape[-1])] * n
        return {
            'data': data.astype(np.float32),
            'labels': labels.astype(np.int64),
            'coords': self._load_coords().astype(np.float32),
            'subject_id': self.subject_id,
            # NATIVE-rate (self.sample_freq) sample indices -- cache_dataset.py rescales
            # to the compiled rate before saving (see BandpassResample changing sample
            # count). [(0, T)]*n (every trial fully real) for every loader that doesn't
            # set self._last_valid_ranges -- i.e. all of them except the event-anchored
            # ones (see IO/preprocessing.py's cut_event_window / _segment_by_annotations).
            'valid_ranges': valid_ranges,
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
    by cache_dataset.py; train-time reads the compiled cache directly in
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
