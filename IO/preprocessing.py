import numpy as np
import scipy.signal
import torch
from typing import Optional, Tuple


class BandpassResample:
    """Compile-time only (cache_compile.py) — bandpass filter + resample to a
    fixed sample_freq. Baked into the cache once; never runs at train time."""
    def __init__(self, original_freq, sample_freq=200, l_freq=None, h_freq=None, notch_freq=None):
        self.original_freq = original_freq
        self.sample_freq = sample_freq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        if self.l_freq is not None and self.h_freq is not None:
            # h_freq must be strictly < Nyquist (original_freq/2) — clamp rather than
            # error out, since a shared config bandpass_filter can legitimately exceed
            # a low-native-rate dataset's Nyquist (e.g. h_freq=100 vs a 200Hz dataset).
            nyquist = self.original_freq / 2
            h_freq = min(self.h_freq, nyquist - 1e-6)
            if h_freq != self.h_freq:
                print(f"  [Warning] h_freq={self.h_freq} >= Nyquist ({nyquist}) at "
                      f"original_freq={self.original_freq} — clamped to {h_freq:.4f}.")
            sos = scipy.signal.butter(4, [self.l_freq, h_freq], btype='bandpass', fs=self.original_freq, output='sos')
            x = scipy.signal.sosfiltfilt(sos, x, axis=-1)

        if self.notch_freq is not None:
            b, a = scipy.signal.iirnotch(self.notch_freq, 30.0, fs=self.original_freq)
            x = scipy.signal.filtfilt(b, a, x, axis=-1)

        if self.sample_freq != self.original_freq:
            new_num_samples = int(x.shape[-1] * self.sample_freq / self.original_freq)
            x = scipy.signal.resample(x, new_num_samples, axis=-1)

        # sosfiltfilt/resample can hand back a negative-strided view (e.g. no
        # resample needed, filtfilt's internal reversal not copied out) — torch
        # refuses those, so force a contiguous copy before handoff.
        return torch.from_numpy(np.ascontiguousarray(x)).float()


class Normalizer:
    """Online (train-time) — normalization only. Bandpass/resample already
    baked into the compiled cache by BandpassResample, so the training
    pipeline never carries that logic. Batched: __call__ takes (N, C, T) and
    normalizes each of the N trials independently (its own mean/std/median,
    not one pooled across the batch) -- same per-trial semantics a Python
    loop of N single-trial calls would give, but computed in one vectorized
    pass instead. IO/dataset.py's _load_task used to do exactly that loop
    (torch.stack([transform(raw_data[i]) for i in range(N)])), which for a
    large subject (e.g. EEGMMIdb's ~362 trials/subject) held N separately-
    normalized tensors in a Python list simultaneously before torch.stack
    copied them into one buffer -- a real, avoidable memory spike across a
    whole dataset's worth of subjects, not just a speed cost (see check_model.py
    OOM investigation, docs/model-analysis-checklist.md)."""
    def __init__(self, normalization_type='fixed'):
        self.normalization_type = str(normalization_type).lower() if normalization_type else 'none'

        valid_norms = ['fixed', 'zscore', 'robust', 'none']
        if self.normalization_type not in valid_norms:
            raise ValueError(f"normalization_type must be one of {valid_norms}")

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return torch.from_numpy(self._normalize(x)).float()

    def _normalize(self, x):
        # x: (N, C, T) -- reduce over (C, T) per trial (axis 0), never pooled across N.
        if self.normalization_type == 'fixed':
            return x / 100.0
        elif self.normalization_type == 'zscore':
            mean = np.mean(x, axis=(-2, -1), keepdims=True)
            std  = np.std(x, axis=(-2, -1), keepdims=True)
            return (x - mean) / (std + 1e-8)
        elif self.normalization_type == 'robust':
            median = np.median(x, axis=(-2, -1), keepdims=True)
            q75 = np.percentile(x, 75, axis=(-2, -1), keepdims=True)
            q25 = np.percentile(x, 25, axis=(-2, -1), keepdims=True)
            return (x - median) / ((q75 - q25) + 1e-8)
        return x


def build_normalizer_from_config(config: dict) -> Normalizer:
    signal_params = config.get('preprocess_params', {'normalization_type': 'zscore'})
    return Normalizer(normalization_type=signal_params['normalization_type'])


def build_bandpass_resample_from_config(config: dict, fs_orig: Optional[float] = None) -> BandpassResample:
    signal_params = config.get('preprocess_params', {
        'sample_freq': 200, 'bandpass_filter': {'l_freq': 0.1, 'h_freq': 80.0}
    })
    bandpass = signal_params['bandpass_filter']

    if fs_orig is None:
        if 'data_metadata' not in config or 'Sample_Frequency' not in config['data_metadata']:
            raise ValueError("Config must contain 'data_metadata.Sample_Frequency' when fs_orig is not provided.")
        fs_orig = config['data_metadata']['Sample_Frequency']

    return BandpassResample(
        original_freq=fs_orig,
        sample_freq=signal_params['sample_freq'],
        l_freq=bandpass['l_freq'],
        h_freq=bandpass['h_freq'],
    )


def cache_suffix(sample_freq, bandpass_filter: dict, pre_event_seconds: float = 0.0,
                  post_event_seconds: float = 0.0) -> str:
    """Derives the compiled-cache filename suffix from the params baked into it —
    shared by cache_compile.py (writes) and IO/dataset.py (reads), so a config
    change that alters either just misses the cache instead of silently reading
    stale data. pre_event_seconds/post_event_seconds only add to the suffix when
    either is nonzero (the default 0/0 keeps existing filenames — and existing
    caches — untouched); once set, EVERY dataset's cache filename changes, same as
    a bandpass_filter change would, even datasets whose own loader ignores these
    params (they're a global compile-time setting, not a per-dataset one — see
    cache_compile.py / IO/loader.py's cut_event_window)."""
    l_freq, h_freq = bandpass_filter['l_freq'], bandpass_filter['h_freq']
    suffix = f"fs{sample_freq}_bp{l_freq}-{h_freq}"
    if pre_event_seconds or post_event_seconds:
        suffix += f"_pre{pre_event_seconds:g}_post{post_event_seconds:g}"
    return suffix


def cut_event_window(data: np.ndarray, event_pts: int, pre_pts: int, post_pts: int
                      ) -> Tuple[np.ndarray, int, int]:
    """data: [C, T] one continuous recording, event_pts: sample index of the real
    event/trigger within it -> (window [C, pre_pts+post_pts], valid_start, valid_end).

    Cuts [event_pts-pre_pts, event_pts+post_pts) around the anchor, ALWAYS returning a
    window of exactly pre_pts+post_pts samples — zero-padded on whichever side(s) run
    past the recording's actual start/end, never dropped (no threshold to clear, unlike
    the old window_continuous_signal design this generalizes — see its docstring).
    valid_start/valid_end mark the real (non-padded) content within the returned window
    (window[:, valid_start:valid_end] is real; everything outside is zero pad) — feeds
    IO/dataset.py's per-row validity, which keeps masking/loss from treating pad as
    signal (see docs/model-analysis-checklist.md).

    Padding here only covers running off the EDGE OF THE RECORDING — it does NOT
    protect against reading real content that belongs to a DIFFERENT, adjacent
    trial/event in the middle of a continuous recording (that's a labeling problem,
    not a bounds problem — see datas/EEGMMIdb/loader.py's docstring for why that
    dataset doesn't use this at all)."""
    C, T = data.shape
    total = pre_pts + post_pts
    raw_start = event_pts - pre_pts
    raw_end = event_pts + post_pts
    window = np.zeros((C, total), dtype=data.dtype)
    src_start, src_end = max(0, raw_start), min(T, raw_end)
    if src_end <= src_start:
        return window, 0, 0  # entirely outside the recording -- degenerate, caller should skip
    dst_start = src_start - raw_start
    dst_end = dst_start + (src_end - src_start)
    window[:, dst_start:dst_end] = data[:, src_start:src_end]
    return window, dst_start, dst_end


def num_patches(total_T: int, patch_len: int, patch_stride: Optional[int] = None) -> int:
    """Patch count slice_patches would produce for a signal of length total_T —
    shared so callers that need P before calling slice_patches (e.g. to size a
    mask) can't drift out of sync with the real formula."""
    patch_stride = patch_stride or patch_len
    return (total_T - patch_len) // patch_stride + 1 if total_T >= patch_len else 0


def slice_patches(x: torch.Tensor, patch_len: int, patch_stride: Optional[int] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """x: [..., T] (any leading dims — [C,T] per-sample or [B,C,T] batched) ->
    (x_patches [..., P, L], time_indices [P], unbatched). patch_stride=None (or
    equal to patch_len) gives the original non-overlapping behavior; smaller
    strides overlap consecutive patches within the same Window. Uses unfold
    (a view, not a copy, for the non-overlap case) so it stays generic on
    leading dims. Drops any remainder < patch_len. Shared by IO/dataset.py's
    PretrainDataset (per-sample) and train_finetune.py's
    FinetuneCollate (batched)."""
    patch_stride = patch_stride or patch_len
    T = x.shape[-1]
    if T < patch_len:
        x_patches = x[..., :0].reshape(*x.shape[:-1], 0, patch_len)
    else:
        # unfold returns a strided view (not a reshape-compatible copy) — .contiguous()
        # so downstream code (fft, model forward, collate) doesn't have to know that.
        x_patches = x.unfold(-1, patch_len, patch_stride).contiguous()  # [..., P, patch_len]
    time_indices = torch.arange(x_patches.shape[-2], dtype=torch.long)
    return x_patches, time_indices


def window_continuous_signal(trials: torch.Tensor, target_L: int, ds_name: str, subject_id,
                              valid_ranges=None
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flatten all trials into a continuous signal, then cut into non-overlapping windows
    of target_L. ALWAYS pads the final partial chunk now (previously conditional on a
    window_pad_threshold — removed; every window gets used, none silently dropped for
    running a few samples short). Moved out of EEGDataset so the base dataset class
    stays focused on load/channel-map/normalize; this is pure slicing.

    valid_ranges: optional list of N (start, end) pairs, one per INPUT trial, marking
    real (non-padded) content within that trial (see IO/loader.py's cut_event_window —
    only event-anchored loaders produce these; None, the default, means every input
    trial is fully real). Propagated through concatenation+rechunking as a per-sample
    validity mask, then reduced back to one (valid_start, valid_end) per OUTPUT window —
    safe because the real region within any one window is always contiguous here: either
    an input trial is fully valid (the None/default case, or any complete input trial
    that isn't the very first/last one touching a recording edge), or trial_len already
    equals target_L (the event-anchored case, one input trial maps 1:1 onto one output
    window, no splicing of two different trials' validity within a single window).

    Returns (assembled [n_windows, C, target_L], labels [n_windows] (dummy, always 0 —
    assembled windows don't carry a real per-trial label), valid_start [n_windows] long,
    valid_end [n_windows] long — window[:, valid_start:valid_end] is real content,
    everything outside is zero pad. See docs/model-analysis-checklist.md for why
    IO/dataset.py's PretrainDataset needs this (masking must not treat pad as signal).
    """
    N, C, T = trials.shape
    if valid_ranges is None:
        valid_ranges = [(0, T)] * N
    # NOT trials.reshape(N*T, C).T — that reinterprets the (N,C,T) memory buffer
    # directly without transposing, scrambling channel and time together. permute
    # first so reshape only merges the N and T axes (already-adjacent after permute),
    # keeping each channel's own timeseries intact and trial-concatenated in order.
    signal = trials.permute(1, 0, 2).reshape(C, N * T)  # (C, N*T)
    total_T = signal.shape[-1]

    valid_mask = np.zeros(total_T, dtype=bool)
    for i, (vs, ve) in enumerate(valid_ranges):
        valid_mask[i * T + vs: i * T + ve] = True

    def _bounds(seg):
        idx = np.flatnonzero(seg)
        return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0)

    n_complete = total_T // target_L
    remainder = total_T % target_L

    windows, valid_starts, valid_ends = [], [], []
    for i in range(n_complete):
        windows.append(signal[:, i * target_L:(i + 1) * target_L])
        vs, ve = _bounds(valid_mask[i * target_L:(i + 1) * target_L])
        valid_starts.append(vs); valid_ends.append(ve)

    if remainder > 0:
        last = torch.zeros((C, target_L), dtype=signal.dtype)
        last[:, :remainder] = signal[:, n_complete * target_L:]
        windows.append(last)
        vs, ve = _bounds(valid_mask[n_complete * target_L: n_complete * target_L + remainder])
        valid_starts.append(vs); valid_ends.append(ve)
        print(f"  [{ds_name} S{subject_id}] Last chunk {remainder}/{target_L}pts kept (padded {target_L - remainder}pts).")

    if not windows:
        raise RuntimeError(f"No windows produced for {ds_name} subject {subject_id} (total_T={total_T}, target_L={target_L}).")

    assembled = torch.stack(windows)
    print(f"  [{ds_name} S{subject_id}] {N} trials x {T}pts -> {total_T}pts -> {len(assembled)} windows of {target_L}pts.")
    return (assembled, torch.zeros(len(assembled), dtype=torch.long),
            torch.tensor(valid_starts, dtype=torch.long), torch.tensor(valid_ends, dtype=torch.long))
