"""Band-filtered orig-vs-recon time series plot, one row per channel."""

import os
import numpy as np
import matplotlib.pyplot as plt


_DEFAULT_BANDS = {
    'Delta': (0.5,  4),
    'Theta': (4,    8),
    'Alpha': (8,   13),
    'Beta':  (13,  30),
    'Gamma': (30, 100),
}


def _canonical_bands(l_freq=None, h_freq=None, band_edges=None, fs=None):
    """Delta/Theta/Alpha/Beta/Gamma edges (overridable via training_params.visualize.bands
    in config, default _DEFAULT_BANDS), clipped to [l_freq, h_freq] when given (the
    preprocessing bandpass, see preprocess_params) — a band entirely outside that range
    is dropped rather than plotted as filtered-out-but-labeled-real content (e.g. Gamma's
    default 30-80Hz upper edge exceeds a common h_freq=75 bandpass; unclipped, MNE would
    filter for 75-80Hz content the preprocessing bandpass already removed upstream, and
    the resulting near-flat trace could misread as a real "no gamma" finding instead of
    "already filtered out before the model ever saw it")."""
    raw = band_edges or _DEFAULT_BANDS
    bands = {'Raw': None}
    nyquist = fs / 2 if fs else None
    for name, (lo, hi) in raw.items():
        if l_freq is not None:
            lo = max(lo, l_freq)
        if h_freq is not None:
            hi = min(hi, h_freq)
        if nyquist is not None:
            hi = min(hi, nyquist * 0.999)  # MNE IIR rejects h_freq >= Nyquist
        if lo >= hi:
            continue
        bands[f'{name} ({lo:g}-{hi:g})'] = (lo, hi)
    return bands


def visualize_reconstruction(train_batch, val_batch, epoch,
                             output_dir='output/visualization/reconstruction',
                             channel_names=None,
                             subject_id=None, trial_idx=None,
                             mask=None, patch_len=100, patch_stride=None, tag='',
                             fs=200.0, l_freq=None, h_freq=None, band_edges=None,
                             event_onset_sec=None, valid_start=None, valid_end=None):
    """
    Band-filtered orig vs recon for all channels of one val sample.
    Rows: channels. Cols: Raw / Delta / Theta / Alpha / Beta / Gamma.
    Masked patches highlighted in red per channel.
    mask: [C, N] bool numpy array or None. Patch p covers samples
    [p * patch_stride, p * patch_stride + patch_len) -- patch_stride defaults to patch_len
    (non-overlapping); with overlapping patches (stride < len) placing patch p at
    p * patch_len stretched the shading to (len/stride)x the real trial length.
    fs: sample rate in Hz (preprocess_params.sample_freq) — used for both the time axis
    and the band-filter cutoffs; defaults to 200.0 only for callers that don't pass one.
    l_freq/h_freq: preprocess_params bandpass — clips the canonical band edges to what the
    preprocessing bandpass actually preserved, see _canonical_bands.
    event_onset_sec: real-trial event onset in seconds (see
    BaseEpochChecker._lookup_event_onset), or None — drawn as a vertical dashed line on
    every panel when given, `is not None` (0 is a real onset, e.g. a trial with no
    pre-event buffer — see docs/model-analysis-checklist.md), never omitted just because
    it's falsy.
    valid_start/valid_end: sample indices (see BaseEpochChecker._lookup_valid_range) —
    real content lies in [valid_start, valid_end), everything outside is compile-time
    zero-pad. orig is already exactly zero there (cache_dataset.py re-zeros post-filter),
    but recon is the model's own output, which has no reason to be zero on pad it was
    never trained to reconstruct meaningfully — and both get re-filtered per band here
    (_band_filter), which would ring across that zero/nonzero edge same as the cache-time
    bug this mirrors (see cache_dataset.py's re-zero comment) if not re-zeroed AFTER
    filtering, not before. When given, both orig and recon are forced flat (zero) outside
    [valid_start, valid_end) on every band column, pre- and post-pad alike, symmetric.
    """
    os.makedirs(output_dir, exist_ok=True)

    val_orig, val_recon = val_batch
    if val_orig is None:
        return

    orig  = val_orig[0].detach().cpu().numpy()
    recon = val_recon[0].detach().cpu().numpy()
    C = orig.shape[0]
    n = min(orig.shape[-1], recon.shape[-1])
    orig, recon = orig[:, :n], recon[:, :n]
    t = np.arange(n) / fs
    stride = patch_stride or patch_len

    vs = max(0, min(valid_start, n)) if valid_start is not None else None
    ve = max(0, min(valid_end,   n)) if valid_end   is not None else None

    bands = _canonical_bands(l_freq, h_freq, band_edges, fs=fs)

    n_rows, n_cols = C, len(bands)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 1.2),
                             sharex=True, constrained_layout=True)
    if C == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Reconstruction (Val) — Epoch {epoch}",
                 fontsize=14, fontweight='bold')

    for col, (band_name, freqs) in enumerate(bands.items()):
        for row in range(C):
            ax = axes[row, col]
            yo, yr = _band_filter(orig[row], recon[row], freqs, fs)
            if vs is not None:
                yo[:vs] = 0.0
                yr[:vs] = 0.0
            if ve is not None:
                yo[ve:] = 0.0
                yr[ve:] = 0.0
            ax.plot(t, yo, color='#666666', lw=0.5, alpha=0.7)
            ax.plot(t, yr, 'r--', lw=0.5, alpha=0.8)
            # shade masked patches
            if mask is not None:
                ch_mask = mask[row] if mask.ndim == 2 else mask  # [N]
                for p_idx, is_masked in enumerate(ch_mask):
                    if is_masked:
                        t0 = p_idx * stride / fs
                        t1 = (p_idx * stride + patch_len) / fs
                        ax.axvspan(t0, t1, color='red', alpha=0.15, linewidth=0)
            if event_onset_sec is not None:
                ax.axvline(event_onset_sec, color='k', ls='--', lw=0.8, alpha=0.8)
            ax.set_xlim(0, n / fs)
            ax.set_yticks([])
            ax.grid(True, alpha=0.08)
            if row == 0:
                ax.set_title(band_name, fontsize=8, fontweight='bold')
            if col == 0:
                ch_label = channel_names[row] if channel_names else f'Ch {row}'
                ax.set_ylabel(ch_label, fontsize=5, rotation=0, labelpad=28, va='center')
            if row < C - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Time (s)", fontsize=6)

    prefix   = f"sub{subject_id}_trial{trial_idx}_" if subject_id is not None else ""
    ep_tag   = f"ep{epoch:04d}_" if epoch is not None else ""
    path = os.path.join(output_dir, f"{prefix}{ep_tag}{tag}recon_signal.png")
    plt.savefig(path, dpi=80, bbox_inches='tight')
    plt.close()
    return path


def _band_filter(orig, recon, freqs, fs=200.0):
    """Band-filter orig and recon using MNE IIR filter. Returns (orig_filtered, recon_filtered)."""
    if freqs is None:
        return orig, recon
    try:
        import mne
        l_f, h_f = freqs
        yo = mne.filter.filter_data(orig.reshape(1, -1).astype(np.float64),
                                    fs, l_f, h_f, method='iir', verbose=False)[0]
        yr = mne.filter.filter_data(recon.reshape(1, -1).astype(np.float64),
                                    fs, l_f, h_f, method='iir', verbose=False)[0]
        return yo, yr
    except Exception as e:
        print(f"[timeseries] band filter {freqs} failed, plotting zeros: {e}")
        return np.zeros_like(orig), np.zeros_like(recon)
