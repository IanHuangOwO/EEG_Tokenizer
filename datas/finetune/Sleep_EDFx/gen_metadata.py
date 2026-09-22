"""
Sleep_EDFx (PhysioNet "Sleep-EDF Database Expanded", Sleep Cassette cohort
only) -- metadata.json generator. See docs/agents/adding-a-dataset.md Step 3.

Source verified directly (2026-09-22) via physionet.org/content/sleep-edfx/
1.0.0/ (SHA256SUMS.txt for the authoritative file list -- 153 PSG +
153 Hypnogram files, the on-page directory listing under-counts due to
pagination) plus one real PSG/Hypnogram pair's header (mne.io.read_raw_edf /
mne.read_annotations).

Sleep Cassette (SC) cohort only -- PhysioNet's own summary text claims "78
subjects", but the real file listing (SHA256SUMS.txt, authoritative --
counted directly, not assumed) has only 52 unique subject numbers (00-76,
sparse, most missing entirely) across 153 PSG recordings, i.e. most of
those 52 have 2 nights, a few 1. Table 14's "78" and PhysioNet's "78" appear
to both be wrong/stale relative to what v1.0.0 actually ships -- flagged,
not silently "corrected" to match either external claim. Healthy adults
aged 25-101, recorded 1987-1991. Does NOT include the separate Sleep
Telemetry (ST) cohort (44 recordings, different study/population).

2 bipolar EEG derivations (Fpz-Cz, Pz-Oz -- NOT single-electrode positions,
so they don't resolve via MNE's standard_1020 montage; coordinates fall
back to zero, same as any unrecognized channel name), 100 Hz. Each PSG file
pairs with a Hypnogram file sharing the same 7-char prefix (e.g.
"SC4001E0-PSG.edf" / "SC4001EC-Hypnogram.edf" -- the 8th character is a
scorer code that varies unpredictably per file, matched by prefix here, not
assumed fixed).

Sleep stages scored per the 1968 Rechtschaffen & Kales manual (verified
directly in one file's annotations: W/1/2/3/4/R/?) -- mapped to Table 14's
5-class scheme (W/N1/N2/N3/REM) by merging stages 3+4 into N3 (standard
AASM-era convention) and dropping '?' (unscored) epochs. PRE/POST-SLEEP WAKE
CROP: the raw recordings include hours of pre-lights-off and post-wake-up
"W" padding (one inspected file: 8.5 continuous hours of W before any other
stage) -- loader.py crops to the real sleep period +/- 30 minutes before
the first and after the last non-W epoch, the standard convention for this
exact dataset (used in the original Sleep-EDF papers and MNE's own
sleep-physionet tutorial), not an invented cutoff.
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(__file__)

DATASET_INFO = {
    "source_url": "https://physionet.org/content/sleep-edfx/1.0.0/",
    "file_format": "EDF (PSG) / EDF+ (Hypnogram annotations)",
    "description": (
        "52 healthy subjects (Sleep Cassette cohort -- PhysioNet's own "
        "summary text says 78, the real file listing says 52, see "
        "dataset_info.notes), whole-night polysomnography, 2 EEG "
        "derivations (Fpz-Cz, Pz-Oz), 5-class sleep stage classification "
        "(W/N1/N2/N3/REM)."
    ),
    "task_type": "sleep_staging",
    "reference": "Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberye JJL "
                 "(2000). Analysis of a sleep-dependent neuronal feedback "
                 "loop: the slow-wave microcontinuity of the EEG. IEEE-BME "
                 "47(9):1185-1194. Also: Goldberger et al. (2000) PhysioBank, "
                 "PhysioToolkit, and PhysioNet. Circulation 101(23):e215-e220.",
    "notes": (
        "Real subject count is 52 (counted directly from SHA256SUMS.txt's "
        "153 PSG recordings), not the 78 claimed by both PhysioNet's own "
        "summary text and Table 14 -- flagged, not silently matched to "
        "either. Sleep Cassette (SC) cohort only, NOT Sleep Telemetry (ST) -- see "
        "this file's module docstring. 30s epochs, R&K stages 3+4 merged "
        "into N3 (AASM convention), '?' (unscored) and epochs outside the "
        "cropped sleep window dropped. Epoch window hardcoded in loader.py "
        "(30s @ 100Hz = 3000 samples), NOT compile.json's global pre/"
        "post_event_seconds. Channels are bipolar derivations (not single "
        "electrode sites) -- coordinates unresolvable, fall back to zero."
    ),
}

CHANNELS = {
    "1": {"label": "Fpz-Cz"},
    "2": {"label": "Pz-Oz"},
}

TARGETS = {
    "count": 5,
    "type": "sleep_stage",
    "0": {"label": "W (wake)"},
    "1": {"label": "N1"},
    "2": {"label": "N2"},
    "3": {"label": "N3 (R&K stages 3+4 merged)"},
    "4": {"label": "REM"},
}


def build_data_structure(raw_dir):
    """Pairs each *-PSG.edf with the *-Hypnogram.edf sharing its 7-char
    prefix (e.g. 'SC4001E') -- the 8th char (scorer code) varies per file,
    not a fixed convention, so matched by prefix rather than assumed."""
    psg_files = sorted(glob.glob(os.path.join(raw_dir, "SC*-PSG.edf")))
    hyp_files = sorted(glob.glob(os.path.join(raw_dir, "SC*-Hypnogram.edf")))
    hyp_by_prefix = {os.path.basename(f)[:7]: os.path.basename(f) for f in hyp_files}

    structure = {}
    for psg_path in psg_files:
        fname = os.path.basename(psg_path)
        m = re.match(r"SC4(\d{2})(\d)E0-PSG\.edf$", fname)
        if not m:
            continue
        sub, night = m.group(1), m.group(2)
        prefix = fname[:7]
        hyp_fname = hyp_by_prefix.get(prefix)
        if hyp_fname is None:
            print(f"  [Warning] {fname}: no matching Hypnogram file, skipping this night")
            continue
        sub_key = str(int(sub))
        structure.setdefault(sub_key, {"files": []})
        structure[sub_key]["files"].append({
            "psg": f"raw/{fname}", "hypnogram": f"raw/{hyp_fname}",
        })
    return structure


def main():
    raw_dir = os.path.join(ROOT, "raw")
    structure = build_data_structure(raw_dir)

    meta = {
        "data_metadata": {
            "dataset_name": "Sleep_EDFx",
            "dataset_info": DATASET_INFO,
            "acquisition": {
                "sample_frequency": 100,
                "window_size_seconds": 30.0,
                "num_subjects": len(structure),
            },
            "targets": TARGETS,
            "channels": {"count": len(CHANNELS), "system": "Bipolar derivations (not a standard montage)", **CHANNELS},
        },
        "data_structure": structure,
    }

    out_path = os.path.join(ROOT, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=4)
    n_nights = sum(len(v["files"]) for v in structure.values())
    print(f"wrote {out_path}: {len(structure)} subjects, {n_nights} nights")


if __name__ == "__main__":
    main()
