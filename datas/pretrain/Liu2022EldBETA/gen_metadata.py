"""
Liu2022EldBETA, loaded through MOABB's Liu2022EldBETA (1.5.0) -- metadata.json generator (downloads subject 1).
Then `python datas/pretrain/Liu2022EldBETA/fetch.py` downloads the rest (resumable).

eldBETA: SSVEP from 100 ELDERLY subjects, 9 targets x 7 blocks, 64 EEG ch at 1000 Hz. Window [0, 5] s from stimulus onset (5 s flicker).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import write_moabb_metadata

ROOT = os.path.dirname(os.path.abspath(__file__))


def tsv_xyz(meta):
    """raw/electrodes.tsv: the BIDS electrodes.tsv inside the dataset's own archives (extracted 2026-10-01 from
    sub-001 ses-01; identical across the 7 sessions of sub-001 and in sub-023): x / y / z in mm, x to the nose, y to
    the left ear. Stored in metres in that frame; coords 'dataset' aligns it to MNE's head frame."""
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(ROOT, "raw", "electrodes.tsv"))]
    table = {r[0].upper(): [round(float(c) / 1000, 7) for c in r[1:4]] for r in rows[1:] if len(r) >= 4}
    ch = meta["data_metadata"]["channels"]
    for k, v in ch.items():
        if k.isdigit() and v["label"].upper() in table:
            v["xyz"] = table[v["label"].upper()]
    ch["coordinates_source"] = "raw/electrodes.tsv (BIDS, from the dataset archives; own frame, metres)"

write_moabb_metadata(
    os.path.dirname(os.path.abspath(__file__)), "Liu2022EldBETA", "Liu2022EldBETA",
    dataset_info={
        "source_url": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.Liu2022EldBETA.html",
        "file_format": "via MOABB",
        "description": 'eldBETA: SSVEP from 100 ELDERLY subjects, 9 targets x 7 blocks, 64 EEG ch at 1000 Hz. Window [0, 5] s from stimulus onset (5 s flicker).',
    },
    target_labels={},
    kwargs={},
    window=[0.0, 5.0],
    continuous_seconds=None,
    postprocess=tsv_xyz,
)
