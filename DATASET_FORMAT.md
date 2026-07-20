# Adding a new EEG dataset

How to convert a raw dataset (zip download, per-subject files, whatever
format it ships in) into the standard `datas/<name>/` layout used by this
repo. Reference implementations (each has a `datas/_gen_<name>_metadata.py`
generator, see Step 3): `BETA_4s`/`BETA_3s` (single .mat file per subject),
`Dial` (split signal/label .mat files), `Inria_Train`/`Inria_Test` (CSV,
shared label file across subjects), `EEGMMIdb` (EDF, multi-run folder per
subject), `BCICIV1_Train`/`BCICIV1_Test` (.mat with real digitized channel
coordinates), `BCICIV2a`/`BCICIV2b` (GDF, event-marker driven),
`GraspAndLift_Train` (continuous multi-label events collapsed to a dummy
pretrain-only label).

## Step 0: research the dataset before writing anything

Don't guess metadata — look it up. Before touching `metadata.json`, find and
read:
- The dataset's official page / paper / PhysioNet-Kaggle-OpenNeuro listing
  (`source_url`).
- Sample rate, number of channels, channel montage/layout, number of
  subjects, trial/window length, number of classes and what each class means.
  These usually come from the paper's Methods section or a README shipped
  with the download — don't infer sample rate from file size or guess class
  meaning from label ints.
- The exact channel order and naming used in the raw files (montage diagrams,
  a `channels.tsv`/`electrodes.txt` if present, or the paper's electrode
  figure). Get this wrong and `_map_channels()` in `IO/dataset.py` will
  silently zero-pad/misalign channels against other datasets.
- Label/event encoding: what each stimulus code, marker value, or annotation
  string actually means (e.g. EEGMMIdb's `T0/T1/T2` annotations only make
  sense after reading PhysioNet's task description).

Put what you find into `dataset_info` (`source_url`, `description`,
`reference`, `task_type`, and any dataset-specific notes worth preserving —
see `EEGMMIdb`'s `electrode_note` field) so the next person doesn't have to
redo this research.

## Step 1: stage the raw files

Raw downloads are usually a zip/tar archive, or a pile of per-subject
archives. Decompress into `datas/<DatasetName>/` (or a subfolder under it)
before writing `metadata.json` — `data_structure` paths are relative to
`dataset_path` and must point at real extracted files, not archives.

```bash
# zip
python -c "import zipfile; zipfile.ZipFile('raw.zip').extractall('datas/MyDataset')"
# tar / tar.gz
python -c "import tarfile; tarfile.open('raw.tar.gz').extractall('datas/MyDataset')"
# per-subject archives (common on PhysioNet/OpenNeuro): loop and extract each
for f in downloads/*.zip; do python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall('datas/MyDataset')" "$f"; done
```

Keep whatever internal layout the archive already has (per-subject folders,
per-session subfolders) — `metadata.json`'s `data_structure` just needs a
relative path/set of paths per subject, it doesn't force a particular
directory shape (see the three shapes in Step 4). Don't commit the original
archive files themselves once extracted — only the files `data_structure`
references need to stick around.

After extracting, spot-check that the "data" files are actually data, not
pointer stubs — a GitHub-hosted mirror of a git-annex/GIN/LFS-backed dataset
(e.g. `HighGamma`'s `high-gamma-dataset-master.zip`) can extract cleanly into
real-looking `.mat`/`.edf` filenames that are actually ~100-byte text files
like `../../.git/annex/objects/<hash>/<hash>` — the real content lives on a
separate content server and needs a dedicated fetch (`gin get`/`git-annex
get`, or the repo's documented per-file download links), not just `git clone`
or a GitHub zip download. `file <extracted>` or checking the file size against
what the raw signal shape implies catches this before you sink time into a
loader for data that was never actually there.

## Step 2: directory layout

```
datas/<DatasetName>/
    metadata.json
    <extracted raw files, any sub-layout>
```

`metadata.json` is the only required file at the top level.

## Step 3: `metadata.json` schema

Always write a small Python generator script, `datas/_gen_<name>_metadata.py`,
that produces `metadata.json` — don't hand-write the JSON directly. It's the
reproducible source of truth: subject lists come from a real directory
listing instead of being transcribed by hand (typo-prone, goes stale the
moment a file is added/removed), and anything computable from the raw files
(coordinates, per-subject class labels, etc.) gets computed instead of typed.
See `_gen_bciciv1_metadata.py` (reads `nfo.xpos`/`nfo.ypos` out of a `.mat`
file, converts cartesian → polar), `_gen_inria_metadata.py` (walks `signals/`
to group per-subject session lists, reads `ChannelsLocation.csv`), and
`_gen_beta_metadata.py` / `_gen_eegmmidb_metadata.py` (directory-listing-driven
`data_structure`, hand-researched channel constants). Re-run the script and
commit its output rather than hand-editing `metadata.json` out of sync with it.

Two top-level keys: `data_metadata` (dataset description) and
`data_structure` (per-subject file index).

```jsonc
{
    "data_metadata": {
        "dataset_name": "MyDataset",
        "dataset_info": {                 // free-form, human-facing — fill in from Step 0 research
            "source_url": "...",
            "file_format": "MATLAB (.mat)",  // or "CSV", "EDF+", "GDF", ...
            "description": "...",
            "task_type": "Motor Imagery",
            "reference": "...",              // paper citation
            "contact": "..."
        },
        "acquisition": {
            "sample_frequency": 250,          // REQUIRED. Hz, used for filtering/resampling.
            "window_size_seconds": 4.0,       // REQUIRED if the loader needs to cut fixed windows
                                               // (see BaseSubjectLoader.target_points). Omit if
                                               // raw files already contain pre-cut trials of fixed length.
            "num_subjects": 55,               // informational
            "num_runs_per_subject": 4         // informational
        },
        "targets": {
            "count": 40,                      // REQUIRED. Number of classes.
            "type": "ssvep",                  // free-form: ssvep / motor_imagery / p300_speller / ...
            "0": { "label": "8.6 Hz", "stimulus_frequency_hz": 8.6 },
            "1": { "label": "8.8 Hz", "stimulus_frequency_hz": 8.8 }
            // ... one entry per class, 0-indexed, matching the label ints your loader emits.
            // "description" per class is optional but recommended when the class meaning
            // isn't self-evident from the label (see Inria_Train, EEGMMIdb above).
        },
        "channels": {
            "count": 64,
            "system": "10-20 International System",   // free-form
            "1": {
                "label": "FP1",                        // REQUIRED. Standard 10-20/10-10 name where
                                                         // possible — _map_channels() (IO/dataset.py)
                                                         // matches datasets to each other by this label
                                                         // (case-insensitive, with old->new 10-20 aliases,
                                                         // e.g. T3->T7, A1->TP9; see _LABEL_ALIASES).
                "original_label": "Fp1.",               // OPTIONAL: raw file's native channel name, if it
                                                         // differs (dots/case from EDF headers, etc — see
                                                         // EEGMMIdb). Not read by code, kept for traceability.
                "coordinates": {                        // REQUIRED (unless you rely on MNE's standard_1020
                                                         // montage — see below).
                    "polar_angle_deg": -17.926,
                    "polar_radius": 0.51499
                }
            }
            // ... 1-indexed, one entry per channel, in the SAME order as the raw signal's channel axis
        }
    },
    "data_structure": {
        // see Step 4 — shape depends on how the raw dataset stores files
    }
}
```

Notes:
- `channels.<n>.coordinates` uses **polar** coordinates (angle in degrees,
  radius normalized ~0-1, standard EEG topomap convention), NOT 3D xyz.
  `BaseSubjectLoader._load_coords_from_metadata()` converts to `[x, y, z]`.
  If `label` matches an MNE `standard_1020` montage channel name, MNE's real
  3D position is used instead and `coordinates` becomes a fallback used only
  when MNE lookup fails or MNE isn't installed (see `EEGMMIdb`, which omits
  `coordinates` entirely for midline channels like `FCZ`/`CPZ` that MNE
  always resolves). Filling in `coordinates` anyway is recommended for
  non-standard channel names, or if you want reproducible topomaps without
  depending on MNE being installed.
- Any channel labelled inside `NON_EEG_CHANNELS` in `IO/dataset.py` (EOG, EMG,
  ECG, reference/mastoid, trigger/status channels) is auto-excluded when a
  downstream config uses `channels_to_use: ["all"]`. Set
  `include_non_eeg_channels: true` in that dataset's `dataset_params` entry to
  keep them (`Inria_Train` keeps its `EOG` channel in metadata for this
  reason — excluded by default, includable on demand).
- `targets.<n>` keys must be 0-indexed and dense (`0..count-1`) — this is the
  label space your loader's `_load_data()` must emit.

## Step 4: `data_structure` shapes

Pick whichever matches how the raw files are actually organized — the
loader's `__init__` reads whatever keys it expects, so the shape is a
contract between `metadata.json` and your loader class, not something the
rest of the pipeline enforces.

**A. Single file per subject** (`BETA_4s`) — signal + labels combined in one file:
```jsonc
"16": { "file": "signals_labels/S16.mat" }
```

**B. Split signal/label files, one pair per subject** (`Dial`):
```jsonc
"1": { "signals": "signals/DataSub_1.mat", "labels": "labels/LabSub_1.mat" }
```

**B'. Split signal/label files, label file SHARED across subjects** (`Inria_Train`)
— common for Kaggle-style competitions with one master label CSV:
```jsonc
"2": { "signals": "signals/Data_S02_Sess01.csv", "labels": "TrainLabels.csv" }
```
The loader is responsible for filtering the shared label file down to the
rows belonging to this subject/session (see `InriaLoader._load_data`, which
matches on a session-ID substring in the `IdFeedBack` column).

**C. Multiple raw files per subject** (`EEGMMIdb`) — e.g. one file per
recording run/session:
```jsonc
"1": { "folder": "S001", "runs": ["S001R01.edf", "S001R02.edf", "..."] }
```
The loader loads and concatenates events across all runs
(`EEGMMIdbLoader._load_data`).

Subject keys need not be contiguous or start at 1 — `BETA_4s` only has
subjects 16-70, `Inria_Train` skips several subject numbers entirely.

## Step 5: signal array conventions

Whatever the raw file format, `_load_data()` must return `(data, labels)`
where:
- `data`: `np.ndarray` shape `(N, C, T)` — N trials, C = `len(desired_channel_indices)`
  (already channel-subsetted, in the metadata's channel order), T = raw
  samples per trial (need not match `window_size_seconds` * `sample_frequency`
  exactly, but should be consistent across trials in the array).
- `labels`: `np.ndarray` shape `(N,)`, `int`, 0-indexed, dense within
  `[0, targets.count)`.

`get_subject_data()` (base class) then casts dtypes, attaches channel coords,
and returns the dict consumed by `IO/dataset.py`.

## Step 6: writing the loader class

Add a class to `IO/loader.py`, subclassing `BaseSubjectLoader`
(`IO/loader.py:8`). Two abstract methods, `_load_coords` and `_load_data`.
A `TemplateLoader` skeleton with the boilerplate already filled in is at the
bottom of `IO/loader.py` — copy it, rename, fill in `_load_data`.

```python
class MyDatasetLoader(BaseSubjectLoader):
    def __init__(self, config, subject_id, desired_channel_indices):
        super().__init__(config, subject_id, desired_channel_indices)
        subject_str = str(subject_id)
        structure = config['data_structure']
        if subject_str not in structure:
            raise ValueError(f"Subject {subject_id} not found in data structure.")
        self.file_path = os.path.join(self.data_root, structure[subject_str]['file'].lstrip('./'))

    def _load_coords(self) -> np.ndarray:
        return self._load_coords_from_metadata()   # standard — reuse unless you have real digitized coords

    def _load_data(self):
        if not os.path.exists(self.file_path):
            return None, None                       # loader must be able to signal "missing subject"
        # ... read raw file, subset self.channel_indices, build labels ...
        return eeg_data, labels
```

### Raw file formats — what to read them with

| Format          | Library                          | Notes |
|------------------|-----------------------------------|-------|
| `.mat` (MATLAB)  | `scipy.io.loadmat`                | Struct fields come back as nested numpy structured arrays — index like `mat['data']['EEG'][0, 0]` (see `BETALoader`). |
| `.csv`           | `pandas.read_csv`                 | Common for Kaggle-style exports: one column per channel + a marker/label column (see `InriaLoader`). |
| `.edf` / `.edf+`  | `mne.io.read_raw_edf(path, preload=True)` | Use `raw.get_data(picks=self.channel_indices)`; resample via `raw.resample(self.sample_freq)` if native rate differs from metadata. Events via `mne.events_from_annotations(raw)` (see `EEGMMIdbLoader`). |
| `.gdf`           | `mne.io.read_raw_gdf(path, preload=True)` | Same API shape as `read_raw_edf` above — GDF stores its own event/annotation table, read via `mne.events_from_annotations` same as EDF. Common for BCI Competition IV datasets (2a/2b). |
| `.bdf` (BioSemi) | `mne.io.read_raw_bdf(path, preload=True)` | Same MNE API shape as EDF/GDF. |
| `.fif` (MNE-native) | `mne.io.read_raw_fif(path, preload=True)` | Same MNE API shape. |

Things the existing loaders show you need to handle per format:
- **Single-file-per-subject, all trials pre-blocked** (`BETALoader`): reshape
  `(C, T, Blocks, Targets)` -> `(N, C, T)` and synthesize labels
  `[0]*blocks + [1]*blocks + ...` since class order is implicit in array
  layout.
- **Split signal/label files** (`DialLoader`): load both, truncate to
  `min(len)` if mismatched, remap 1-indexed labels to 0-indexed.
- **Split files with a shared/master label file** (`InriaLoader`): filter the
  shared label table down to this subject's rows by matching a session-ID
  substring; fall back to a `SampleSubmission.csv`-style file if the subject
  has no rows in the primary label file (e.g. held-out test subjects).
- **Test/holdout split with no published ground truth** (competition test
  sets, e.g. Kaggle's Inria BCI Challenge — `test.zip` ships signals only,
  no label file, and the true labels were never released): don't guess real
  labels and don't silently drop the split — `_load_data()` needs *some*
  label file to key trial-cutting off of, or the loader returns `None, None`
  for every subject and the split silently contributes zero trials. Write a
  placeholder label file (e.g. `Inria_Test/_gen_dummy_labels.py`: count real
  event markers per file, emit `Prediction=0` for each) so trials still get
  cut, and record in `dataset_info.notes`/a dedicated note that the labels
  are dummies. Safe for self-supervised pretraining (label values unused);
  never use such a split for supervised finetune/eval of the labeled task.
- **Continuous recording + event markers, fixed trial length**
  (`BCICIVLoader`, `InriaLoader`): use `self.standard_window` /
  `self.sample_freq` to compute `trial_len` in samples, slice fixed windows
  starting at each marker position, drop any trial whose window runs past
  the end of the recording. Remap arbitrary/bipolar label encodings
  (`{-1,+1}`, 1-indexed) to dense 0-indexed via `{v: i for i, v in enumerate(np.unique(raw_labels))}`.
- **Multi-run folder + annotation-based segmentation** (`EEGMMIdbLoader`):
  loop over each run file, `mne.io.read_raw_edf` / `read_raw_gdf`, resample
  to `self.sample_freq` if the file's native rate differs,
  `mne.events_from_annotations`, map annotation codes to class ints, and
  concatenate trials across all runs for the subject.
- **Continuous multi-label event annotations** (`GraspAndLiftLoader` — Kaggle
  Grasp-and-Lift EEG: 6 binary event columns per sample, overlapping in
  time): this doesn't fit the one-dense-int-label-per-trial contract
  (`labels: np.ndarray` shape `(N,)`, Step 5) at all — there is no single
  "class" per window. Don't force a lossy single-label reduction unless you
  actually need the labels for something; if the dataset's only current use
  is self-supervised pretraining, do what `GraspAndLiftLoader` does: chop
  each continuous recording into fixed non-overlapping `standard_window`-size
  windows, assign dummy label `0` to all of them, and read-but-discard the
  real event columns. Set `targets.count: 1` (not the real class count) so
  the placeholder is honest, and record in `dataset_info.notes` that a real
  multi-label loader would need writing before this split is usable for
  supervised event-detection finetune/eval.

Always wrap per-subject/per-file failures so one bad file doesn't kill the
whole loading run — return `None, None` from `_load_data()` rather than
raising; `EEGDataset.__init__` (`IO/dataset.py:118-163`) already
try/excepts around each task and just skips it with a printed warning.

## Step 7: wiring into the IO pipeline

Two edits, both in `IO/dataset.py`:

1. Import the class:
   ```python
   from .loader import BETALoader, DialLoader, BCICIVLoader, InriaLoader, EEGMMIdbLoader, MyDatasetLoader
   ```
2. Register it in `_resolve_loader()` (`IO/dataset.py:381`) — matches on
   substring of either the config key or `data_metadata.dataset_name`:
   ```python
   def _resolve_loader(dataset_name: str, ds_name_key: str):
       for name in (dataset_name, ds_name_key):
           ...
           if 'MyDataset' in name:
               return MyDatasetLoader
       return DialLoader  # fallback
   ```

Nothing else needs to change — `build_dataset_from_config()` reads
`metadata.json`, resolves the loader by name match, and folds the dataset
into the same channel-unified, cross-subject-padded tensor as every other
dataset.

## Step 8: config wiring (to actually use it)

Add an entry under `dataset_params.pretrain` or `dataset_params.finetune` in
your config JSON:

```jsonc
"dataset_params": {
    "pretrain": {
        "MyDataset": {
            "dataset_path": "datas/MyDataset",
            "subject_to_use": ["all"],     // or an explicit list of subject-id ints/strs
            "channels_to_use": ["all"]     // or an explicit ordered channel-label list
        }
    }
}
```

- `subject_to_use: ["all"]` is resolved against `data_structure`'s keys at
  train time (`train_pretrain.py:167-177` / `train_finetune.py`) — shuffled
  with a fixed seed and split into train/val by `subject_to_use` **before**
  `build_dataset_from_config` ever runs, so subjects never leak across the
  split.
- `channels_to_use: ["all"]` on the **first** dataset listed determines the
  unified channel space (`_resolve_target_channels`, `IO/dataset.py:396`);
  other datasets get their channels remapped onto it by label
  (`_map_channels`), zero-padding anything they don't have. Alternatively set
  `preprocess_params.canonical_channels` to a fixed ordered list to pin the
  channel space explicitly across all datasets.

## Step 9: sanity-check

```bash
python -c "
import json
from IO.dataset import build_dataset_from_config
config = json.load(open('config/config_pretrain.json'))
config['dataset_params']['pretrain'] = {
    'MyDataset': {'dataset_path': 'datas/MyDataset', 'subject_to_use': ['all'], 'channels_to_use': ['all']}
}
ds = build_dataset_from_config(config, mode='base')
print(len(ds), ds.data.shape)
"
```
Watch the printed `[channel map] matched X/Y` line — low match counts mean
your channel labels don't line up with the 10-20 naming convention
(check `_LABEL_ALIASES` / `_normalize_label` in `IO/dataset.py`).

## Currently unconverted raw datasets in `./datas`

These folders exist but have no `metadata.json` yet (as of this writing):
`Siena`.

`HighGamma` is blocked, not just unconverted: `high-gamma-dataset-master.zip`
only contains git-annex pointer stubs (see the Step 1 gotcha above), no real
`.mat` data. Needs a real fetch from the GIN repo
(https://web.gin.g-node.org/robintibor/high-gamma-dataset) before there's
anything to format.

`GraspAndLift_Train` is converted (`datas/_gen_graspandlift_metadata.py`,
`GraspAndLiftLoader` — see the continuous-multi-label-events bullet above).
`GraspAndLift_Val` is still empty — the Kaggle competition's held-out
series 9-10 were never downloaded, so there's nothing to convert there yet.

`BCICIV2a`/`BCICIV2b` are converted — see `BCICIV2aLoader`/`BCICIV2bLoader`
in `IO/loader.py` for GDF + event-marker reference examples (`BCICIV2b`
also shows the multi-run-per-subject shape C, concatenating the 3 training
sessions per subject).

`BCICIV1_Train`/`BCICIV1_Test` are converted too — `BCICIVLoader` (generic,
shape A single-file-per-subject) already handled this format out of the
box, including the "no `mrk` in eval data" case (falls back to evenly-spaced
windows, dummy label 0 — see the missing-test-labels bullet above). Only
`datas/_gen_bciciv1_metadata.py` and the two `metadata.json` needed writing.
Two quirks worth knowing if you touch this dataset again: (1) raw subject
ids are letters `a`-`g`, remapped to ints `1`-`7` in `data_structure`
(`EEGDataset.__init__` hard-requires `int(subject_id)`) with the original
letter kept as `orig_id`; (2) each subject's 2 MI classes are a
subject-specific pick from {left hand, right hand, foot} — dense labels 0/1
are still globally consistent (0 = raw `y=-1`, 1 = raw `y=+1`), but their
physical meaning varies per subject, recorded in each `data_structure`
entry's `classes` field.
