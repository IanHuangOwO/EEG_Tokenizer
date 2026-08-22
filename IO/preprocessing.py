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
    pipeline never carries that logic."""
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
        if self.normalization_type == 'fixed':
            return x / 100.0
        elif self.normalization_type == 'zscore':
            mean = np.mean(x)
            std  = np.std(x)
            return (x - mean) / (std + 1e-8)
        elif self.normalization_type == 'robust':
            median = np.median(x)
            q75, q25 = np.percentile(x, [75, 25])
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


def cache_suffix(sample_freq, bandpass_filter: dict) -> str:
    """Derives the compiled-cache filename suffix from the params baked into it —
    shared by cache_compile.py (writes) and IO/dataset.py (reads), so a config
    change that alters either just misses the cache instead of silently reading
    stale data."""
    l_freq, h_freq = bandpass_filter['l_freq'], bandpass_filter['h_freq']
    return f"fs{sample_freq}_bp{l_freq}-{h_freq}"


def slice_patches(x: torch.Tensor, patch_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """x: [..., T] (any leading dims — [C,T] per-sample or [B,C,T] batched) ->
    (x_patches [..., P, L], time_indices [P], unbatched). Drops any remainder <
    patch_len. Shared by IO/dataset.py's TokenizerDataset/PretrainDataset
    (per-sample) and train_finetune.py's FinetuneCollate (batched)."""
    T = x.shape[-1]
    P = T // patch_len
    x_patches = x[..., :P * patch_len].reshape(*x.shape[:-1], P, patch_len)
    time_indices = torch.arange(P, dtype=torch.long)
    return x_patches, time_indices


def window_continuous_signal(trials: torch.Tensor, target_L: int, threshold: float,
                              ds_name: str, subject_id) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten all trials into a continuous signal, then cut into non-overlapping
    windows of target_L. Keeps the last chunk (zero-padded) only if it fills at
    least `threshold` of target_L. Moved out of EEGDataset so the base dataset
    class stays focused on load/channel-map/normalize; this is pure slicing.
    """
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
