import numpy as np
import scipy.signal
import torch

class RecurrentVQProcessing:
    """
    Preprocessing pipeline for RecurrentVQ.
    """
    def __init__(
        self, 
        original_freq,
        target_freq=200,
        l_freq=None,
        h_freq=None,
        notch_freq=None,
        normalization_type='fixed'
    ):
        """
        Args:
            original_freq (float): Sampling frequency of the input data.
            target_freq (float): Desired sampling frequency (default: 200 Hz).
            l_freq (float): Low cut-off frequency for bandpass. Default is None (no low-pass).
            h_freq (float): High cut-off frequency for bandpass. Default is None (no high-pass).
            notch_freq (float): Frequency to notch filter. Set to None to disable.
            normalization_type (str): 'fixed' (div by 100), 'zscore', 'robust', or 'none'.
        """
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
        """
        Args:
            x (torch.Tensor or np.ndarray): EEG signal of shape (Channels, Time).
            
        Returns:
            torch.Tensor: Preprocessed signal of shape (Channels, New_Time).
        """
        # Ensure working with numpy for scipy operations
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            
        # 1. Filtering (Bandpass + Notch)
        # Note: We apply filters along the last axis (Time)
        
        # Bandpass
        if self.l_freq is not None and self.h_freq is not None:
            sos = scipy.signal.butter(4, [self.l_freq, self.h_freq], btype='bandpass', fs=self.original_freq, output='sos')
            x = scipy.signal.sosfiltfilt(sos, x, axis=-1)
            
        # Notch (Only if notch_freq is set)
        if self.notch_freq is not None:
            # Quality factor Q = 30 is typical for notch
            b, a = scipy.signal.iirnotch(self.notch_freq, 30.0, fs=self.original_freq)
            x = scipy.signal.filtfilt(b, a, x, axis=-1)

        # 2. Resampling
        if self.target_freq != self.original_freq:
            num_samples = x.shape[-1]
            new_num_samples = int(num_samples * self.target_freq / self.original_freq)
            x = scipy.signal.resample(x, new_num_samples, axis=-1)

        # 3. Normalization
        x = self._normalize(x)

        # Return as Float Tensor
        return torch.from_numpy(x).float()
    
    def _normalize(self, x):
        """
        Applies selected normalization strategy per sample.
        Input x: (Channels, Time)
        """
        if self.normalization_type == 'fixed':
            # LaBraM Standard: Divide by 100
            return x / 100.0
        
        elif self.normalization_type == 'zscore':
            # Z-Score per channel: (x - mean) / std
            mean = np.mean(x, axis=-1, keepdims=True)
            std = np.std(x, axis=-1, keepdims=True)
            return (x - mean) / (std + 1e-8)
            
        elif self.normalization_type == 'robust':
            # Robust Scaler: (x - median) / IQR
            median = np.median(x, axis=-1, keepdims=True)
            q75, q25 = np.percentile(x, [75 ,25], axis=-1, keepdims=True)
            iqr = q75 - q25
            return (x - median) / (iqr + 1e-8)
            
        # 'none' falls through here
        return x