# Adding a montage (standard or custom)

`config/montages.json` and the `IO/dataset.py` lookup branch described here are live.
Currently populated: `10-20` (21ch, classic Jasper 1958 subset) and `10-10` (64ch, this
project's original canonical channel set) — both built from real
`mne.channels.make_standard_montage('standard_1020')` cartesian coordinates. See
`config.json`'s `preprocess_params.canonical_channels: "10-10"`.

## What a montage is, here

A montage is a named, ordered list of canonical channel labels (+ optional per-channel
reference coordinates) used for **cross-dataset channel unification**
(`build_dataset_from_config` in `IO/dataset.py`, see `CLAUDE.md`'s "Multi-dataset
training" section): the first dataset's channel space is fixed to this list, order and
all, and every other dataset's channels are mapped onto it — present channels matched by
name (via `EEGDataset._normalize_label`/`_map_channels`, which already uppercases and
applies an old-naming alias table), missing ones zero-padded.

A montage is **not** the source of truth for per-dataset channel coordinates fed to the
model (`coords [B, C, 3]`) — those are resolved per-dataset at load time in
`IO/loader.py` (`BaseSubjectLoader._load_coords_from_metadata`): MNE's
`make_standard_montage('standard_1020')` tried first, falling back to that dataset's own
`metadata.json` `polar_angle_deg`/`polar_radius`. A montage's own coordinates are
reference/documentation/QA data — useful for sanity-checking a dataset's reported
positions against the standard, not required for the pipeline to run.

## File: `config/montages.json`

One JSON object, keyed by montage name:

```json
{
  "10-20": {
    "description": "Classic 21-electrode Jasper 1958 system",
    "source": "mne:standard_1020",
    "channels": [
      {
        "label": "Fp1",
        "polar_angle_deg": -18.0,
        "polar_radius": 0.511,
        "cartesian": { "x": -0.0294, "y": 0.0983, "z": 0.0271 },
        "region": "frontal",
        "hemisphere": "L"
      }
    ]
  },
  "10-10": { "description": "...", "source": "mne:standard_1020", "channels": [ "..." ] }
}
```

Per-channel fields:

| field             | required | notes                                                              |
|--------------------|----------|---------------------------------------------------------------------|
| `label`            | yes      | canonical name, matched via the same normalize/alias path as datasets |
| `polar_angle_deg`  | no       | matches the convention already used by every `datas/*/metadata.json` |
| `polar_radius`     | no       | same                                                                 |
| `cartesian`        | no       | `{x,y,z}`, when the source provides real 3D (e.g. MNE)              |
| `region`           | no       | frontal / central / parietal / occipital / temporal — for future QA/filtering |
| `hemisphere`       | no       | `L` / `R` / `Z` (midline)                                            |

Top-level `description`/`source` are documentation only — `source` should say where the
coordinates came from (`"mne:standard_1020"`, `"mne:standard_1005"`,
`"derived-from:10-10"`, a specific dataset name, or `"manual"`), so a later reviewer can
tell a verified reference from a hand-entered guess.

## `config.json` wiring

`preprocess_params.canonical_channels` accepts either form:

```json
"canonical_channels": "10-10"
```
— string, resolved by name against `config/montages.json`.

```json
"canonical_channels": ["Fp1", "Fp2", "F7", "..."]
```
— inline list, unchanged from today's behavior, no `montages.json` entry required. Use
this for one-off/experimental channel sets that don't deserve a named, reusable montage.

`IO/dataset.py`'s existing canonical-channels resolution (around
`build_dataset_from_config`, currently reading `pp.get('canonical_channels', [])`
directly as a list) gains one branch: if the value is a `str`, load
`config/montages.json` and pull that key's `channels[*].label` list, in order; if it's
already a `list`, use it as-is (today's behavior, untouched).

## Procedure: adding a STANDARD montage (10-5, an equipment layout, etc.)

1. **Source coordinates from a verified reference, not memory.** For 10-20/10-10/10-05,
   use `mne.channels.make_standard_montage('standard_1020')` (10-20/10-10) or
   `'standard_1005'` (10-05, dense) — MNE ships real digitized/simulated positions for
   these. For a manufacturer layout (BioSemi 64/128/256, EGI/Philips HydroCel
   128/256/257), MNE also ships several of these by name
   (`mne.channels.get_builtin_montages()` lists what's available) — check there before
   typing anything by hand.
2. Write a small one-off script (not part of the training pipeline) that builds the
   montage, calls `.get_positions()['ch_pos']` for cartesian coordinates, and writes the
   `config/montages.json` entry — set `source` to say exactly which MNE montage/version
   was used.
3. If the standard introduces channel names not already covered by
   `EEGDataset._LABEL_ALIASES` (`IO/dataset.py`) — old-style names, intermediate-ring
   names like BCICIV1's `CFC1`/`CCP1` — add the alias mapping there so any dataset using
   the old name still matches.
4. Add the montage name to the tracked-montages list below.

## Tracked montages

| name    | channels | source                          | notes                                             |
|---------|----------|----------------------------------|----------------------------------------------------|
| `10-20` | 21       | `mne:standard_1020` (subset)     | classic Jasper 1958 electrodes + A1/A2 mastoids     |
| `10-10` | 64       | `mne:standard_1020`              | this project's original hand-curated canonical set  |

Beyond the 10-x family, `mne.channels.get_builtin_montages()` (MNE 1.11) also ships
manufacturer layouts not currently tracked here: BioSemi (16/32/64/128/160/256),
EGI/HydroCel (32/64/65/128/129/256/257), easycap variants, mgh60/70, and a few fNIRS
optode layouts (`artinis-*`) — add any of these the same way if a dataset needs one.

## Procedure: adding a CUSTOM montage (project-specific subset)

1. Pull the channel list (and optionally `polar_angle_deg`/`polar_radius`) straight from
   the relevant dataset's own `datas/<name>/metadata.json` `data_metadata.channels` dict
   — that file is already the authoritative source for that dataset's own channels, no
   need to touch MNE.
2. Set `source` to the dataset name it came from (e.g. `"datas/Dial/metadata.json"`).
3. `region`/`hemisphere`/`cartesian` are optional — fill in only what you actually know;
   omit rather than guess.

## Verification

After adding or changing a montage and pointing `canonical_channels` at it, run the
usual dataset build path and check `EEGDataset._map_channels`'s printed line for every
configured dataset:

```
[channel map] matched X/Y | zero-padded: [...]
```

Confirm the matched count is what you expect for each dataset (a full match for the
dataset the montage was built from; partial + an expected zero-padded list for others
being unified onto it) — an unexpectedly low match count usually means a naming
mismatch that needs an alias, not a data problem.
