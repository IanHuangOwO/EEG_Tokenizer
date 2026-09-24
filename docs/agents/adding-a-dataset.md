# Adding a new EEG dataset

How to convert a raw dataset (zip download, per-subject files, whatever
format it ships in) into the standard `datas/<name>/` layout used by this
repo. Reference implementations (each has a `datas/<name>/gen_metadata.py`
generator, see Step 3): `BETA_4s`/`BETA_3s` (single .mat file per subject),
`Nakanishi2015` (split signal/label .mat files), `Inria_Train`/`Inria_Test` (CSV,
shared label file across subjects), `PhysionetMI` (EDF, multi-run folder per
subject), `BCICIV1_Train`/`BCICIV1_Test` (.mat with real digitized channel
coordinates), `BNCI2014001`/`BNCI2014004` (GDF, event-marker driven),
`GraspAndLift_Train` (continuous multi-label events collapsed to a dummy
pretrain-only label), `DEAP` (Python pickle, `pickle.load(..., encoding='latin1')`
for a Python-2-pickled file; continuous multi-dimensional ratings thresholded into
discrete classes — a genuine label-scheme decision, not a guessable raw-format
fact, see its `loader.py`).

Training never reads these raw files directly — a separate **compile step**
(`cache_dataset.py`, see Step 7) bandpass-filters/resamples each subject once
and writes a per-subject `.npz` cache; `IO/dataset.py` reads only that cache
at train time. Everything through Step 6 below is about getting the raw
format wired up for that compile step, not about wiring into training
directly.

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
  string actually means (e.g. PhysionetMI's `T0/T1/T2` annotations only make
  sense after reading PhysioNet's task description).

Put what you find into `dataset_info` (`source_url`, `description`,
`reference`, `task_type`, and any dataset-specific notes worth preserving —
see `PhysionetMI`'s `electrode_note` field) so the next person doesn't have to
redo this research.

## Step 1: stage the raw files

Raw downloads are usually a zip/tar archive, or a pile of per-subject
archives. Decompress into `datas/<DatasetName>/raw/` (or a subfolder under it)
before writing `metadata.json` — `data_structure` paths are relative to
`dataset_path` (`datas/<DatasetName>/`) and must point at real extracted
files under `raw/`, not archives.

```bash
# zip
python -c "import zipfile; zipfile.ZipFile('raw.zip').extractall('datas/MyDataset/raw')"
# tar / tar.gz
python -c "import tarfile; tarfile.open('raw.tar.gz').extractall('datas/MyDataset/raw')"
# per-subject archives (common on PhysioNet/OpenNeuro): loop and extract each
for f in downloads/*.zip; do python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall('datas/MyDataset/raw')" "$f"; done
```

Keep whatever internal layout the archive already has (per-subject folders,
per-session subfolders) — `metadata.json`'s `data_structure` just needs a
`raw/`-prefixed relative path/set of paths per subject, it doesn't force a
particular directory shape (see the three shapes in Step 4). Don't commit the
original archive files themselves once extracted — only the files
`data_structure` references need to stick around.

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
    loader.py              # dataset-specific loader, see Step 6
    gen_metadata.py        # metadata.json generator, see Step 3
    raw/                   # extracted raw files, any sub-layout
    cache/                 # cache_dataset.py output, see Step 7 — not committed
```

`metadata.json` is the only required file at the top level.

## Step 3: `metadata.json` schema

Always write a small Python generator script, `datas/<DatasetName>/gen_metadata.py`,
that produces `metadata.json` — don't hand-write the JSON directly. It's the
reproducible source of truth: subject lists come from a real directory
listing instead of being transcribed by hand (typo-prone, goes stale the
moment a file is added/removed), and anything computable from the raw files
(coordinates, per-subject class labels, etc.) gets computed instead of typed.
`datas/_template_gen_metadata.py` has the boilerplate/schema already filled
in — copy it, fill in `DATASET_INFO`/`CHANNELS`/`TARGETS`/`build_data_structure`.
See `BCICIV1_Train/gen_metadata.py` (reads `nfo.xpos`/`nfo.ypos` out of a
`.mat` file, converts cartesian → polar), `Inria_Train/gen_metadata.py`
(walks `raw/signals/` to group per-subject session lists, reads
`ChannelsLocation.csv`), and `BETA_4s/gen_metadata.py` /
`PhysionetMI/gen_metadata.py` (directory-listing-driven `data_structure`,
hand-researched channel constants). Re-run the script and commit its output
rather than hand-editing `metadata.json` out of sync with it.

Every path written into `data_structure` must be `raw/`-prefixed (relative to
`dataset_path`, i.e. `datas/<DatasetName>/`) — see Step 1.

Some datasets ship train/test or A/B splits as sibling directories
(`BCICIV1_Train`+`BCICIV1_Test`, `BNCI2014001`+`BNCI2014004`, `Inria_Train`+`Inria_Test`)
— each still gets its own independent `gen_metadata.py`/`loader.py`, even
when the two splits share the same channel layout/paper. Every `datas/<Name>/`
directory is a standalone dataset as far as the pipeline is concerned; nothing
reaches across to a sibling directory.

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
            // isn't self-evident from the label (see Inria_Train, PhysionetMI above).
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
                                                         // PhysionetMI). Not read by code, kept for traceability.
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
  when MNE lookup fails or MNE isn't installed (see `PhysionetMI`, which omits
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
"16": { "file": "raw/signals_labels/S16.mat" }
```

**B. Split signal/label files, one pair per subject** (`Nakanishi2015`):
```jsonc
"1": { "signals": "raw/signals/DataSub_1.mat", "labels": "raw/labels/LabSub_1.mat" }
```

**B'. Split signal/label files, label file SHARED across subjects** (`Inria_Train`)
— common for Kaggle-style competitions with one master label CSV:
```jsonc
"2": { "signals": "raw/signals/Data_S02_Sess01.csv", "labels": "raw/TrainLabels.csv" }
```
The loader is responsible for filtering the shared label file down to the
rows belonging to this subject/session (see `datas/pretrain/Inria_Train/loader.py`'s
`Loader._load_data`, which matches on a session-ID substring in the
`IdFeedBack` column).

**C. Multiple raw files per subject** (`PhysionetMI`) — e.g. one file per
recording run/session:
```jsonc
"1": { "folder": "raw/S001", "runs": ["S001R01.edf", "S001R02.edf", "..."] }
```
The loader loads and concatenates events across all runs
(`datas/finetune/PhysionetMI/loader.py`'s `Loader._load_data`).

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

Create `datas/MyDataset/loader.py`, with a class named exactly `Loader`
(that fixed name, plus this file's location, is the whole discovery
contract — no registry to edit anywhere else) subclassing `BaseSubjectLoader`
(`IO/loader.py`). Two abstract methods, `_load_coords` and `_load_data`.
`datas/_template_loader.py` has the boilerplate already filled in — copy it,
fill in `_load_data`.

This class only ever runs during Step 7's compile step, never at train time —
train time reads the compiled `.npz` cache directly in `IO/dataset.py`'s
`EEGDataset._load_task`, regardless of which dataset it is (no loader class
involved on that path).

```python
# datas/MyDataset/loader.py
from IO.loader import BaseSubjectLoader

class Loader(BaseSubjectLoader):
    def __init__(self, config, subject_id, desired_channel_indices):
        super().__init__(config, subject_id, desired_channel_indices)
        entry = self._require_subject(subject_id)   # looks up data_structure[str(subject_id)], raises if missing
        self.file_path = self._resolve(entry['file'])  # joins onto dataset_path, strips a leading './'
        # Split signal/label style (Nakanishi2015):  self._resolve(entry['signals']) / self._resolve(entry['labels'])
        # Multi-run style (PhysionetMI):       [self._resolve(os.path.join(entry['folder'], r)) for r in entry['runs']]

    # _load_coords defaults to self._load_coords_from_metadata() (BaseSubjectLoader) —
    # override only if you have real digitized coordinates (see BCICIV1_Train/loader.py).

    def _load_data(self):
        if not self._existing([self.file_path]):
            return None, None                       # loader must be able to signal "missing subject"
        # ... read raw file, subset self.channel_indices, build labels ...
        return eeg_data, labels
```

`datas/_template_loader.py` and `datas/_template_gen_metadata.py` are the copy-paste
starting points and stay in sync with the real `BaseSubjectLoader` API — copy from
there, not from this snippet, if the two ever drift.

### Raw file formats — what to read them with

| Format          | Library                          | Notes |
|------------------|-----------------------------------|-------|
| `.mat` (MATLAB)  | `scipy.io.loadmat`                | Struct fields come back as nested numpy structured arrays — index like `mat['data']['EEG'][0, 0]` (see `datas/pretrain/BETA_4s/loader.py`). |
| `.csv`           | `pandas.read_csv`                 | Common for Kaggle-style exports: one column per channel + a marker/label column (see `datas/pretrain/Inria_Train/loader.py`). |
| `.edf` / `.edf+`  | `mne.io.read_raw_edf(path, preload=True)` | Use `raw.get_data(picks=self.channel_indices)`; resample via `raw.resample(self.sample_freq)` if native rate differs from metadata. Events via `mne.events_from_annotations(raw)` (see `datas/finetune/PhysionetMI/loader.py`). |
| `.gdf`           | `mne.io.read_raw_gdf(path, preload=True)` | Same API shape as `read_raw_edf` above — GDF stores its own event/annotation table, read via `mne.events_from_annotations` same as EDF. Common for BCI Competition IV datasets (2a/2b). |
| `.bdf` (BioSemi) | `mne.io.read_raw_bdf(path, preload=True)` | Same MNE API shape as EDF/GDF. |
| `.fif` (MNE-native) | `mne.io.read_raw_fif(path, preload=True)` | Same MNE API shape. |

Things the existing loaders show you need to handle per format:
- **Single-file-per-subject, all trials pre-blocked** (`datas/pretrain/BETA_4s/loader.py`):
  reshape `(C, T, Blocks, Targets)` -> `(N, C, T)` and synthesize labels
  `[0]*blocks + [1]*blocks + ...` since class order is implicit in array
  layout.
- **Split signal/label files** (`datas/finetune/Nakanishi2015/loader.py`): load both, truncate
  to `min(len)` if mismatched, remap 1-indexed labels to 0-indexed.
- **Split files with a shared/master label file** (`datas/pretrain/Inria_Train/loader.py`):
  filter the shared label table down to this subject's rows by matching a
  session-ID substring; fall back to a `SampleSubmission.csv`-style file if
  the subject has no rows in the primary label file (e.g. held-out test
  subjects).
- **Test/holdout split with no published ground truth** (competition test
  sets, e.g. Kaggle's Inria BCI Challenge — `test.zip` ships signals only,
  no label file, and the true labels were never released): don't guess real
  labels and don't silently drop the split — `_load_data()` needs *some*
  label file to key trial-cutting off of, or the loader returns `None, None`
  for every subject and the split silently contributes zero trials. Generate
  a placeholder label file lazily from inside `loader.py`'s `Loader.__init__`
  (count real event markers per file, emit `Prediction=0` for each) so trials
  still get cut without a separate manual script to remember to run — see
  `datas/pretrain/Inria_Test/loader.py`'s `_generate_dummy_labels()`, called only when
  the expected label file is missing — and record in `dataset_info.notes`/a
  dedicated note that the labels are dummies. Safe for self-supervised
  pretraining (label values unused); never use such a split for supervised
  finetune/eval of the labeled task.
- **Continuous recording + event markers, fixed trial length**
  (`datas/pretrain/BCICIV1_Train/loader.py`, `datas/pretrain/Inria_Train/loader.py`): use
  `self.standard_window` / `self.sample_freq` to compute `trial_len` in
  samples, slice fixed windows starting at each marker position, drop any
  trial whose window runs past the end of the recording. Remap
  arbitrary/bipolar label encodings (`{-1,+1}`, 1-indexed) to dense 0-indexed
  via `{v: i for i, v in enumerate(np.unique(raw_labels))}`.
- **Multi-run folder + annotation-based segmentation** (`datas/finetune/PhysionetMI/loader.py`):
  loop over each run file, `mne.io.read_raw_edf` / `read_raw_gdf`, resample
  to `self.sample_freq` if the file's native rate differs,
  `mne.events_from_annotations`, map annotation codes to class ints, and
  concatenate trials across all runs for the subject.
- **Continuous ratings, no native discrete label** (`datas/pretrain/DEAP/loader.py` — DEAP's
  per-trial valence/arousal/dominance/liking are 1-9 continuous self-report scores,
  not classes): don't silently pick a threshold/scheme — this is a real modeling
  decision (which dimensions, how many classes, what split point) with downstream
  consequences, not something to infer from the raw format. Ask, record the answer
  in `dataset_info.notes` (which dimensions/threshold were used and why), and
  compute the discrete label in `loader.py` from the raw continuous values (never
  hand-edit a derived label into `metadata.json`).
- **Continuous multi-label event annotations** (`datas/pretrain/GraspAndLift_Train/loader.py`
  — Kaggle Grasp-and-Lift EEG: 6 binary event columns per sample, overlapping
  in time): this doesn't fit the one-dense-int-label-per-trial contract
  (`labels: np.ndarray` shape `(N,)`, Step 5) at all — there is no single
  "class" per window. Don't force a lossy single-label reduction unless you
  actually need the labels for something; if the dataset's only current use
  is self-supervised pretraining, do what this loader does: chop each
  continuous recording into fixed non-overlapping `standard_window`-size
  windows, assign dummy label `0` to all of them, and read-but-discard the
  real event columns. Set `targets.count: 1` (not the real class count) so
  the placeholder is honest, and record in `dataset_info.notes` that a real
  multi-label loader would need writing before this split is usable for
  supervised event-detection finetune/eval.

Always wrap per-subject/per-file failures so one bad file doesn't kill the
whole loading run — return `None, None` from `_load_data()` rather than
raising; `EEGDataset.__init__` (`IO/dataset.py`) already
try/excepts around each task and just skips it with a printed warning.

## Step 7: compile the cache

No registration step — `datas/MyDataset/loader.py` existing IS the
registration (`IO/loader.py`'s `resolve_dataset_loader()` dynamically imports
it by directory convention). Instead, run the compile step:

1. Add an entry to `configs/compile.json`:
   ```jsonc
   "datasets": {
       "MyDataset": { "dataset_path": "datas/MyDataset" }
   }
   ```
   (no `subject_to_use`/`channels_to_use` here — compile always does every
   subject, every native channel; `sample_freq`/`bandpass_filter` at the top
   of `compile.json` MUST match `config.json`'s `preprocess_params` or the
   cache won't be found at train time.)
2. Run it:
   ```bash
   python cache_dataset.py --config configs/compile.json --dataset MyDataset
   ```
3. Confirm `datas/MyDataset/cache/<subject>_fs<...>_bp<...>.npz` files were
   written, one per subject with real data — `[MyDataset SX] (N, C, T) -> ...`
   lines printed during compile.
4. Verification is baked into the same command and runs automatically right
   after compiling — no separate step needed. It checks shape/label ranges,
   dead (zero-variance) channels, and that the bandpass filter actually
   attenuated power above its cutoff (Welch PSD in-band vs out-of-band), and
   prints `[MyDataset SX] OK` or `FAIL:` with the reason per subject. A
   non-zero exit code means something failed — read the `FAIL:` lines before
   wiring the dataset into a training config.
   - `--no-verify` skips this (compile only, old `cache_dataset.py` behaviour).
   - `--verify-only` skips compiling and just checks an existing cache
     (`--subjects N` limits how many, default 3) — useful to re-check a cache
     without recompiling, e.g. after editing `metadata.json` by hand.
   - `--deep` (either mode) also re-runs the raw loader + `BandpassResample`
     fresh and diffs byte-for-byte against the cache — slower (parses raw
     files again), but the strongest check: confirms the cache isn't stale
     relative to the current `loader.py`/compile code.

Training never imports `datas/MyDataset/loader.py` — `build_dataset_from_config()`
(`IO/dataset.py`) always reads the `.npz` cache directly, which just needs to
exist.

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
  unified channel space (`_resolve_target_channels`, `IO/dataset.py`);
  other datasets get their channels remapped onto it by label
  (`_map_channels`), zero-padding anything they don't have. Alternatively set
  `preprocess_params.canonical_channels` to a fixed ordered list to pin the
  channel space explicitly across all datasets.

## Step 9: sanity-check

```bash
python -c "
import json
from IO.dataset import build_dataset_from_config
config = json.load(open('configs/pretrain.template.json'))
config['dataset_params']['pretrain'] = {
    'MyDataset': {'dataset_path': 'datas/MyDataset', 'subject_to_use': ['all'], 'channels_to_use': ['all']}
}
ds = build_dataset_from_config(config, mode='base')
print(len(ds), ds.data.shape)
"
```
Watch the printed `[channel map] matched X/Y` line — low match counts mean
your channel labels don't line up with the 10-20 naming convention
(check `_LABEL_ALIASES` / `_normalize_label` in `IO/dataset.py`). This reads
through the compiled cache — if Step 7 wasn't run, `EEGDataset._load_task`
raises a clear `FileNotFoundError` naming the missing `.npz` path instead of
parsing raw files.

## Currently unconverted raw datasets in `./datas`

`Siena` is blocked, not just unconverted: `datas/finetune/Siena/archive.zip` (as
staged, 2026-09-22) contains a single flat file, `Siena_Sleep_EEG_Data.csv` —
944,640 rows x 20 channel columns (`Fp1,F3,C3,P3,O1,F7,T3,T5,Fc1,Fc5,Cp1,Cp5,
F9,Fz,Cz,Pz,Pf2,F4,C4,P4`) + one binary `diagnosis` column (536,320 rows `1`,
408,320 rows `0`). This is very likely a Kaggle re-export of PhysioNet's
*Siena Scalp EEG Database* (a seizure dataset, despite the "Sleep" filename —
Siena's real corpus has no sleep-staging task) with the original per-subject/
per-recording structure stripped out. Missing, not guessable per Step 0:
- **No subject-ID column or file split** — every row is concatenated into one
  table with no boundary marker. Per-subject caching and the
  subject-level train/val split this repo relies on (`CLAUDE.md`: "data never
  leaks between subjects") cannot be done from this file as-is; inventing
  arbitrary row-count chunks and calling them "subjects" would be fabricated
  metadata, not real subject boundaries.
- **No documented sample rate.** PhysioNet's real Siena corpus is 512 Hz, but
  this CSV doesn't state whether/how it was resampled — don't assume 512 Hz
  without a source page or README confirming it.
- **`Pf2` is not a standard 10-20/10-10 label** (`IO/dataset.py`'s
  `_LABEL_ALIASES` won't resolve it) — likely a renamed reference/ground
  channel; needs the original channel key to map correctly.
- The zip has no accompanying README/documentation, so none of the above can
  be resolved from the file itself.

Needed before this can be converted: the original source page (Kaggle listing
or PhysioNet), or the real per-subject PhysioNet EDF files
(https://physionet.org/content/siena-scalp-eeg/) instead of this flattened
CSV.

`HighGamma` is blocked, not just unconverted: `high-gamma-dataset-master.zip`
only contains git-annex pointer stubs (see the Step 1 gotcha above), no real
`.mat` data. Needs a real fetch from the GIN repo
(https://web.gin.g-node.org/robintibor/high-gamma-dataset) before there's
anything to format.

`GraspAndLift_Train` is converted (`datas/pretrain/GraspAndLift_Train/gen_metadata.py`,
`datas/pretrain/GraspAndLift_Train/loader.py` — see the continuous-multi-label-events
bullet above). `GraspAndLift_Test` only has `test.zip` (the Kaggle
competition's held-out series 9-10) — still unextracted/unconverted.

`BNCI2014001`/`BNCI2014004` are converted — see `datas/finetune/BNCI2014001/loader.py`/
`datas/finetune/BNCI2014004/loader.py` for GDF + event-marker reference examples
(`BNCI2014004` also shows the multi-run-per-subject shape C, concatenating the
3 training sessions per subject).

`BCICIV1_Train`/`BCICIV1_Test` are converted too — `datas/pretrain/BCICIV1_Train/loader.py`
(generic, shape A single-file-per-subject) already handled this format out of
the box, including the "no `mrk` in eval data" case (falls back to
evenly-spaced windows, dummy label 0 — see the missing-test-labels bullet
above). Only `datas/pretrain/BCICIV1_Train/gen_metadata.py` (writes both
directories' `metadata.json`, see Step 3) needed writing. Two quirks worth
knowing if you touch this dataset again: (1) raw subject
ids are letters `a`-`g`, remapped to ints `1`-`7` in `data_structure`
(`EEGDataset.__init__` hard-requires `int(subject_id)`) with the original
letter kept as `orig_id`; (2) each subject's 2 MI classes are a
subject-specific pick from {left hand, right hand, foot} — dense labels 0/1
are still globally consistent (0 = raw `y=-1`, 1 = raw `y=+1`), but their
physical meaning varies per subject, recorded in each `data_structure`
entry's `classes` field.
