import json
import os
import warnings
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
        # Global compile-time setting (configs/compile.json's compile_params, passed into
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


# --- MOABB-backed loading ------------------------------------------------------------
# MOABB-backed dataset loading (compile-time only, like every datas/<Name>/loader.py).
#
# For a dataset MOABB covers, datas/<Name>/loader.py is just `class Loader(MoabbLoader): pass`
# and gen_metadata.py calls write_moabb_metadata(). MOABB does the download and the file
# parsing (events, sessions, runs); everything after -- trial cutting with the global
# pre/post_event_seconds, bandpass/resample, channel unification -- stays ours
# (cache_dataset.py). MOABB's paradigm/evaluation layers are deliberately not used.
#
# Raw files live in datas/<Name>/raw/ in MOABB's own layout (MNE-<code>-data/...):
# moabb_dataset() redirects MOABB's storage-path lookup there.

_RAW_DIR = None


def _get_dataset_path(sign, path=None):
    return path if path is not None else _RAW_DIR


def moabb_dataset(class_name: str, dataset_path: str, **kwargs):
    """MOABB dataset whose downloads land in <dataset_path>/raw/."""
    global _RAW_DIR
    import sys
    import moabb.datasets
    ds = getattr(moabb.datasets, class_name)(**kwargs)
    # ponytail: module-global redirect of MOABB's storage-path lookup. Its MNE_DATASETS_<SIGN>_PATH
    # keys use a per-module sign (e.g. every BNCI dataset shares "BNCI"), not ds.code, and
    # several modules import get_dataset_path by name -- so patch every copy. One raw dir at
    # a time: fine for cache_dataset.py's per-dataset loop, not for two datasets in one process
    # concurrently.
    _RAW_DIR = os.path.abspath(os.path.join(dataset_path, 'raw'))
    for name, mod in list(sys.modules.items()):
        if name.startswith('moabb.datasets') and hasattr(mod, 'get_dataset_path'):
            mod.get_dataset_path = _get_dataset_path
    return ds


def subject_runs(ds, subject: int):
    """[(session, run, mne.io.Raw)] for one subject, downloading it if needed."""
    # ponytail: private _get_single_subject_data instead of get_data -- get_data's session
    # filter drops Lee2019's first session (selected_sessions (1, 2) vs session keys
    # '0'/'1', MOABB 1.5.0), and it also skips MOABB's cache layer we don't need.
    # Switch to get_data if a MOABB upgrade removes the private method.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sessions = ds._get_single_subject_data(subject)
    return [(s, r, raw) for s, runs in sessions.items() for r, raw in runs.items()]


class MoabbLoader(BaseSubjectLoader):
    """
    Cuts one trial per labelled event. Default: anchor = event + dataset.interval[0] (the
    MI/ERP onset MOABB defines), window = [anchor - pre_event_seconds, anchor +
    post_event_seconds] (post falls back to the interval length when unset). A dataset
    whose trials don't fit the global pre/post (P300 flashes, back-to-back PhysionetMI
    trials) sets metadata moabb.window = [t0, t1]: seconds relative to the raw event,
    used instead. moabb.continuous = true (pretrain-only datasets whose trials overlap,
    e.g. P300 flashes ~0.1 s apart) ignores events and cuts every run into non-overlapping
    window_size_seconds windows with label 0. Channels are picked by name (metadata
    'original_label'); labels come from metadata targets' 'moabb_event'.
    """
    def __init__(self, config: Dict, subject_id: int, desired_channel_indices: List[int]):
        super().__init__(config, subject_id, desired_channel_indices)
        self.moabb_subject = int(self._require_subject(subject_id)['moabb_subject'])
        m = self.data_metadata['moabb']
        self.ds = moabb_dataset(m['class'], self.data_root, **m.get('kwargs', {}))
        ch = self.data_metadata['channels']
        self.pick_names = [ch[str(i + 1)]['original_label'] for i in desired_channel_indices]
        t = self.data_metadata['targets']
        self.event_to_label = {t[k]['moabb_event']: int(k) for k in t if k.isdigit()}
        self.window = m.get('window')
        self.continuous = m.get('continuous', False)

    def _load_data(self):
        import mne
        lo, hi = self.ds.interval
        trials, labels, ranges = [], [], []
        for _, _, raw in subject_runs(self.ds, self.moabb_subject):
            self._resample_if_needed(raw)
            sf = raw.info['sfreq']
            if self.continuous:
                data = raw.get_data(picks=self.pick_names).astype(np.float32)
                win = int(round(self.standard_window * sf))
                n = data.shape[1] // win
                if n:
                    trials += list(data[:, :n * win].reshape(data.shape[0], n, win).transpose(1, 0, 2))
                    labels += [0] * n
                    ranges += [(0, win)] * n
                continue
            if self.window:
                shift, pre, post = 0, int(round(-self.window[0] * sf)), int(round(self.window[1] * sf))
            else:
                shift = int(round(lo * sf))
                pre = int(round(self.pre_event_seconds * sf))
                post = int(round((self.post_event_seconds or (hi - lo)) * sf))
            # Raw straight from _get_single_subject_data: events are on the stim channel
            # (MOABB's get_data pipeline is what would turn them into annotations).
            if mne.pick_types(raw.info, stim=True).size:
                events = mne.find_events(raw, shortest_event=0, verbose=False)
            else:
                events, _ = mne.events_from_annotations(raw, event_id=self.ds.event_id, verbose=False)
            code_to_label = {self.ds.event_id[k]: v for k, v in self.event_to_label.items()}
            data = raw.get_data(picks=self.pick_names).astype(np.float32)
            for sample, _, code in events:
                if code not in code_to_label:
                    continue
                anchor = sample - raw.first_samp + shift
                window, vs, ve = cut_event_window(data, anchor, pre, post)
                if ve <= vs:
                    continue
                trials.append(window)
                labels.append(code_to_label[code])
                ranges.append((vs, ve))
        if not trials:
            return None, None
        self._last_valid_ranges = ranges
        return np.stack(trials), np.array(labels, dtype=np.int64)


def write_moabb_metadata(root: str, name: str, class_name: str, dataset_info: Dict,
                         target_labels: Dict[str, str], kwargs: Dict = None,
                         window: List[float] = None, continuous_seconds: float = None) -> Dict:
    """
    Writes <root>/metadata.json from MOABB: EEG channel names and sample rate from the
    first subject's first run (downloads it if needed), subjects from ds.subject_list,
    targets = MOABB event names. target_labels maps event name -> readable label; its key
    order sets the label indices (so a migrated dataset can keep its old label order),
    events it leaves out follow in sorted order.
    """
    import mne
    kwargs = kwargs or {}
    ds = moabb_dataset(class_name, root, **kwargs)
    raw = subject_runs(ds, ds.subject_list[0])[0][2]
    eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    events = [e for e in target_labels if e in ds.event_id] + sorted(set(ds.event_id) - set(target_labels))
    lo, hi = ds.interval
    # Where the MOABB event sits inside each compiled trial, in seconds from trial start --
    # read by tools.analysis.lookup_event_onset_sample for the event line on time-axis
    # plots. window=[t0, t1] is relative to the raw event (-> -t0); the default window is
    # centred on event + interval[0] (the MI/stimulus onset MOABB defines) with the global
    # pre_event_seconds before it; continuous windows have no event.
    if continuous_seconds:
        onset = None
    elif window:
        onset = 0.0 - window[0]  # 0.0 - x, not -x: no '-0.0' in metadata
    else:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'configs', 'compile.json')) as f:
            onset = json.load(f)['compile_params'].get('pre_event_seconds', 0.0)
    meta = {
        "data_metadata": {
            "dataset_name": name,
            "dataset_info": dataset_info,
            **({"event_onset_seconds": onset} if onset is not None else {}),
            "moabb": {"class": class_name, "kwargs": kwargs, "code": ds.code,
                      **({"window": window} if window else {}),
                      **({"continuous": True} if continuous_seconds else {})},
            "acquisition": {
                "sample_frequency": raw.info['sfreq'],
                "window_size_seconds": continuous_seconds or ((window[1] - window[0]) if window else hi - lo),
                "num_subjects": len(ds.subject_list),
            },
            "targets": {"count": 1, "type": "pretrain_dummy",
                        "0": {"label": "dummy (continuous window, no task label)", "moabb_event": None}}
            if continuous_seconds else {"count": len(events), "type": ds.paradigm,
                        **{str(i): {"label": target_labels.get(e, e), "moabb_event": e}
                           for i, e in enumerate(events)}},
            "channels": {"count": len(eeg), **{str(i + 1): {"label": n, "original_label": n}
                                               for i, n in enumerate(eeg)}},
        },
        "data_structure": {str(s): {"moabb_subject": s} for s in ds.subject_list},
    }
    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
    print(f"wrote {name}/metadata.json: {len(ds.subject_list)} subjects, {len(eeg)} EEG channels, "
          f"{raw.info['sfreq']} Hz, events {events}")
    return meta


def fetch_moabb(dataset_path: str, retries: int = 10) -> None:
    """Downloads every subject of a MOABB-backed dataset (metadata.json already written)
    into <dataset_path>/raw/. Resumable: complete files are skipped. Retries on the
    connection drops the big mirrors (wasabi, figshare) throw."""
    import time
    meta = json.load(open(os.path.join(dataset_path, 'metadata.json')))
    m = meta['data_metadata']['moabb']
    ds = moabb_dataset(m['class'], dataset_path, **m.get('kwargs', {}))
    raw_dir = os.path.abspath(os.path.join(dataset_path, 'raw'))
    for s in ds.subject_list:
        for attempt in range(retries):
            try:
                ds.data_path(s, path=raw_dir)
                break
            except Exception as e:
                print(f"subject {s} attempt {attempt} failed: {e}", flush=True)
                time.sleep(30)
        else:
            print(f"subject {s} FAILED after {retries} attempts", flush=True)
            continue
        print(f"subject {s} ok", flush=True)
