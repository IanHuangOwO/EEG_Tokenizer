"""
Cho2017, loaded through MOABB's Cho2017 (1.5.0) -- metadata.json generator (downloads subject 1).
Then `python datas/pretrain/Cho2017/fetch.py` downloads the rest (resumable).

Motor imagery L/R hand (GigaDB 100295): 52 subjects, 64 EEG ch (+4 EMG dropped) at 512 Hz, 200-240 trials. Window: global pre/post around the cue (MOABB interval [0, 3]).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

ROOT = os.path.dirname(os.path.abspath(__file__))


def senloc_xyz(meta):
    """Every subject's .mat carries eeg.senloc [64, 3]: that subject's digitized electrode positions in cm, rows in
    the EEG channel order (differ between subjects by 1.7-3.5 cm, checked 2026-10-01). Stored per subject in metres
    (data_structure.<id>.channel_xyz, own frame); coords 'dataset' aligns each subject to MNE's head frame."""
    import glob
    import scipy.io as sio
    n_ch = sum(k.isdigit() for k in meta["data_metadata"]["channels"])
    for sid, entry in meta["data_structure"].items():
        f = glob.glob(os.path.join(ROOT, "raw", "**", "mat_data", f"s{int(sid):02d}.mat"), recursive=True)
        if not f:
            continue
        loc = sio.loadmat(f[0], squeeze_me=True, struct_as_record=False)["eeg"].senloc
        assert loc.shape == (n_ch, 3), (sid, loc.shape)
        entry["channel_xyz"] = [[round(float(c) / 100, 6) for c in row] for row in loc]
    meta["data_metadata"]["channels"]["coordinates_source"] = "eeg.senloc of each subject's .mat (per subject, own frame)"

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Cho2017", "Cho2017",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Cho2017.html",
        "file_format": "via MOABB",
        "description": 'Motor imagery L/R hand (GigaDB 100295): 52 subjects, 64 EEG ch (+4 EMG dropped) at 512 Hz, 200-240 trials. Window: global pre/post around the cue (MOABB interval [0, 3]).',
    },
    target_labels={},
    kwargs={},
    window=None,
    continuous_seconds=None,
    postprocess=senloc_xyz,
)
