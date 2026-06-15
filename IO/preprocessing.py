import numpy as np
import scipy.signal
import torch
from typing import Optional


class EEGPreprocessing:
    def __init__(
        self,
        original_freq,
        target_freq=200,
        l_freq=None,
        h_freq=None,
        notch_freq=None,
        normalization_type='fixed'
    ):
        self.original_freq = original_freq
        self.target_freq = target_freq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq

        self.normalization_type = str(normalization_type).lower() if normalization_type else 'none'

        valid_norms = ['fixed', 'zscore', 'robust', 'none']
        if self.normalization_type not in valid_norms:
            raise ValueError(f"normalization_type must be one of {valid_norms}")

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        if self.l_freq is not None and self.h_freq is not None:
            sos = scipy.signal.butter(4, [self.l_freq, self.h_freq], btype='bandpass', fs=self.original_freq, output='sos')
            x = scipy.signal.sosfiltfilt(sos, x, axis=-1)

        if self.notch_freq is not None:
            b, a = scipy.signal.iirnotch(self.notch_freq, 30.0, fs=self.original_freq)
            x = scipy.signal.filtfilt(b, a, x, axis=-1)

        if self.target_freq != self.original_freq:
            new_num_samples = int(x.shape[-1] * self.target_freq / self.original_freq)
            x = scipy.signal.resample(x, new_num_samples, axis=-1)

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


def build_preprocessing_from_config(config: dict, fs_orig: Optional[float] = None) -> EEGPreprocessing:
    signal_params = config.get('preprocess_params', {
        'target_freq': 200, 'l_freq': 0.1, 'h_freq': 80.0, 'normalization_type': 'zscore'
    })

    if fs_orig is None:
        if 'data_metadata' not in config or 'Sample_Frequency' not in config['data_metadata']:
            raise ValueError("Config must contain 'data_metadata.Sample_Frequency' when fs_orig is not provided.")
        fs_orig = config['data_metadata']['Sample_Frequency']

    return EEGPreprocessing(
        original_freq=fs_orig,
        target_freq=signal_params['target_freq'],
        l_freq=signal_params['l_freq'],
        h_freq=signal_params['h_freq'],
        normalization_type=signal_params['normalization_type'],
    )
