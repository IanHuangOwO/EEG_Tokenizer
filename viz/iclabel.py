"""ICLabel classification of stamps — treats each used stamp as a pseudo-IC and runs
the ICLabel network (mne-icalabel) on IC-style features built from the stamp's own
model objects:

- topography  = the stamp's signed per-channel mixing column (viz.extract's amp_topo,
  valid channels only) — exactly the object ICLabel reads a scalp map from,
- activation  = the stamp's per-patch decoded waveform (a*D_hat + b*Hilbert(D_hat) at
  the strongest channel's gain pair), stitched across the trial's patches with zeros
  where the stamp wasn't selected,
- autocorrelation/PSD computed from that activation by mne-icalabel's own feature
  code (bypassing its ICA-object plumbing — the helpers only read n_components_ off
  the ICA and sfreq/length/montage off the Raw, so a shim + a zero-data RawArray with
  the real channel montage is sufficient and keeps all of ICLabel's own
  normalization).

CAVEAT: ICLabel was trained on ICA components of continuous (>= seconds), 1-100 Hz,
CAR-referenced EEG. A stamp is a 0.5 s dictionary template with a sparse stitched
activation from ONE trial — the class distribution is an interpretability hint
("this stamp smells like line noise / eye / brain"), not a calibrated probability.

Import/model failures degrade to None — the gallery renders without the row rather
than the checker crashing on an optional dependency.
"""

import warnings

import numpy as np

ICLABEL_CLASSES = ['Brain', 'Muscle', 'Eye', 'Heart', 'LineN', 'ChanN', 'Other']


def stamp_iclabel_probs(ch_pos, sfreq, mixing, activities):
    """ch_pos: [Cv, 3] valid channels' 3D coords (any consistent head frame),
    sfreq: sampling rate, mixing: [Cv, Q] signed per-channel mixing columns (one per
    stamp), activities: [Q, T] stitched activation time courses.
    -> probs [Q, 7] (ICLABEL_CLASSES order) or None when mne-icalabel is unavailable
    or anything in the pipeline fails (deliberately silent-degrading, see module
    docstring)."""
    try:
        import mne
        from mne_icalabel.iclabel.features import _eeg_autocorr, _eeg_rpsd, _eeg_topoplot
        from mne_icalabel.iclabel.network import run_iclabel
        from types import SimpleNamespace

        Cv, Q = mixing.shape
        T = activities.shape[1]
        ch_names = [f'E{i}' for i in range(Cv)]
        # zero-data Raw: the feature helpers never read the data (topo comes from
        # `mixing`, psd/autocorr from `activities`) — only sfreq/times/montage matter.
        info = mne.create_info(ch_names, sfreq, 'eeg')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = mne.io.RawArray(np.zeros((Cv, T)), info, verbose='error')
            # scale positions to a ~9.5 cm head radius; the loc conversion is
            # angle-based so the absolute scale barely matters, this just keeps MNE's
            # sanity checks quiet.
            pos = np.asarray(ch_pos, dtype=float)
            pos = pos / (np.linalg.norm(pos, axis=1).max() + 1e-9) * 0.095
            montage = mne.channels.make_dig_montage(
                ch_pos={n: p for n, p in zip(ch_names, pos)}, coord_frame='head')
            raw.set_montage(montage, verbose='error')

            ica_shim = SimpleNamespace(n_components_=Q)
            topo = _eeg_topoplot(raw, np.asarray(mixing, dtype=float), ch_names)
            act = np.asarray(activities, dtype=float)
            psd = _eeg_rpsd(raw, ica_shim, act)
            autocorr = _eeg_autocorr(raw, ica_shim, act)
            # backend explicit: this mne-icalabel version's own default ('pytorch')
            # fails its own validation, which only accepts 'torch'/'onnx'/None.
            probs = run_iclabel(topo, psd, autocorr, backend='torch')
        return np.asarray(probs)
    except Exception as e:  # optional feature: never take the checker down with it
        warnings.warn(f'stamp ICLabel skipped: {type(e).__name__}: {e}')
        return None
