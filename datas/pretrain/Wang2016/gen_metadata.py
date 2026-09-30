"""
Wang2016, loaded through MOABB's Wang2016 (1.5.0) -- metadata.json generator (downloads subject 1).
Then `python datas/pretrain/Wang2016/fetch.py` downloads the rest (resumable).

Tsinghua SSVEP benchmark: 34 subjects, 64 EEG ch at 250 Hz, 40 targets x 6 blocks. MOABB stores 6 s epochs back to back (0.5 s cue, 5 s flicker, 0.5 s tail), so the window is fixed to the flicker, [0.5, 5.5] s.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

ROOT = os.path.dirname(os.path.abspath(__file__))


def loc_coordinates(meta):
    """raw/64-channels.loc: the benchmark's own channel file (bci.med.tsinghua.edu.cn/upload/yijun/64-channels.loc,
    fetched 2026-10-01; MOABB's zenodo copy leaves it out): index, polar angle, radius (EEGLAB convention, 0.5 = the
    equator), label."""
    table = {}
    for line in open(os.path.join(ROOT, "raw", "64-channels.loc")):
        f = line.split()
        if len(f) >= 4:
            table[f[3].upper()] = (float(f[1]), float(f[2]))
    ch = meta["data_metadata"]["channels"]
    for k, v in ch.items():
        if k.isdigit():
            ang, rad = table[v["label"].upper()]
            v["coordinates"] = {"polar_angle_deg": ang, "polar_radius": rad}
    ch["coordinates_source"] = "raw/64-channels.loc (Tsinghua benchmark)"
    ch["polar_equator_radius"] = 0.5

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Wang2016", "Wang2016",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Wang2016.html",
        "file_format": "via MOABB",
        "description": 'Tsinghua SSVEP benchmark: 34 subjects, 64 EEG ch at 250 Hz, 40 targets x 6 blocks. MOABB stores 6 s epochs back to back (0.5 s cue, 5 s flicker, 0.5 s tail), so the window is fixed to the flicker, [0.5, 5.5] s.',
    },
    target_labels={},
    kwargs={},
    window=[0.5, 5.5],
    continuous_seconds=None,
    postprocess=loc_coordinates,
)
