# 0023: Real channel coordinates instead of the name-matched 10-10 grid

- **Status:** reverted (user, 2026-09-30). The code was removed (IO/dataset.py, tools/analysis/backbone_eval.py back to
  their state before 951e3c5); `grid` is the only layout. Why: see Outcome below.
- **Context:** `IO/dataset.py` mapped every dataset onto the 64 slots of the canonical 10-10 montage by channel name
  (`_map_channels`): a channel whose label is not one of the 64 names (EGI `E1..E129`, BioSemi `A1..D32`, 10-05 names
  such as `FFC1h`, extra sites like `Iz`) was dropped, and every kept channel sat at its grid slot. The new pretraining
  datasets (docs/reports / datas/DATASETS.md, 2026-09-30) are chosen for montage and device diversity and accurate
  positions, which the grid discards. The model never needed the grid: it reads channels only through their
  coordinates (Fourier coordinate embedding, per-block relative spatial bias) and a validity mask, and has no
  per-slot parameter.

## Decision

`preprocess_params.channel_layout`: `grid` (the old mapping, the default, so every existing run and config reproduces)
or `real`:

- **<= 64 EEG channels:** every EEG channel is kept, with its real coordinates. A channel whose label is a canonical
  10-10 name stays in that name's slot (so a dataset with only canonical names loads exactly as under `grid`); every
  other EEG channel fills a free slot. Non-EEG channels are excluded as before.
- **> 64 EEG channels:** reduced to the 64 canonical sites (option a): a site the recording has by name is copied
  exactly; a site it lacks is filled by inverse-distance-weighted interpolation (1/d^2, the 4 nearest good
  electrodes) at the site's standard position. Spherical splines were the first choice and failed the card's test
  (check 3: higher mean error, blow-ups on a noisy electrode). Computed at load time from the cached native
  channels (option: load time), so compiled caches are unchanged.
- **Coordinates** (real layout): digitized `xyz` in metadata (metres, MNE head frame) first, then MNE's
  standard_1020 position for the label, then standard_1005 (10-05 half-step names such as FFC1h), then the metadata's polar coordinates projected onto a 95 mm head sphere
  (the grid layout's flat polar fallback is kept there unchanged). A channel with none of these is dropped with a
  warning.
- **Named slots:** each task records which slots hold the channel their canonical name says (`all_named_slots`);
  name-based logic (the pretraining channel subsampler, `backbone_eval`'s motor-3 -> bci-22 test) uses only those.

## Consequences

- New datasets' `gen_metadata.py` should write digitized `xyz` when the source has it (BIDS `electrodes.tsv`,
  converted to metres in the MNE head frame).
- Under `real`, datasets that had non-grid channels gain them (and > 64-channel datasets change from a name subset
  to 64 sites with interpolation), so the corpus changes; datasets with only canonical names do not.
- Finetune feature caches are keyed by `preprocess_params`, so a `real` run never reuses a `grid` cache.
- Verified by docs/cards/2026-09-30-real-coordinates.md.

## Outcome (2026-09-30, reverted)

- Two tiny backbones on `real` (seeds 1, 2; combined head, 3 finetune seeds each) lost BNCI2014001 loso by 10 points
  (39.9 vs 49.9, every held-out subject lower, head train balanced accuracy ~46% vs ~69%) and BNCI2014004 loso by 3
  (74.2 vs 77.2); few-shot and BNCI2014008 were level (001 few-shot +4.3).
- "Real coordinates" was a misnomer: no pretrain or finetune metadata records digitized positions, so every channel was
  placed by MNE's template for its name under both layouts. What `real` changed was the channel set: extra non-grid
  channels in 11 datasets and IDW-interpolated canonical sites for Schirrmeister2017. Which of the two caused the loss
  was not tested.
- The polar tables some metadata carries (BETA: EEGLAB .loc, equator 0.5; Inria: idealised 10-20, equator 0.36;
  GraspAndLift: a borrowed template table, nonlinear) use different conventions and are templates too, not measurements.
- Measured positions would come from datasets that ship them (BIDS electrodes.tsv), written into metadata as `xyz`;
  revisit with those. Runs archived in output/archive/2026-09-30_real_layout/.
