# Card: real layout with each dataset's own electrode coordinates (tiny, 3 seeds)

Written 2026-10-01, before any code or run.

- **Change:** the `real` channel layout of ADR 0023 (every EEG channel kept; > 64 channels reduced to the 64 canonical
  sites, a site the recording lacks filled by IDW from its 4 nearest electrodes), with positions taken from **each
  dataset's own coordinates** instead of MNE's template for the channel name. ADR 0023 never did this: no metadata
  recorded positions, so every channel sat at MNE's template position (its outcome section).
- **Question:** does a backbone pretrained on the datasets' own electrode positions (plus their non-grid channels)
  transfer better or worse than the grid backbone? The earlier real-vs-grid comparison is void (head mismatch), so
  this is also the first fair real-layout comparison.

## Coordinates, per channel, in this order

1. **Dataset's own positions** (`xyz` in metadata.json, metres, MNE head frame), from, per dataset:
   - a coordinate file shipped with the data (BIDS `electrodes.tsv`, `.loc` / `.elc` / `.sfp` / `.ced` / `.locs`,
     a digitization in the raw files, a channel-location CSV), fetched from the dataset's source when not on disk;
   - or the polar table already in metadata.json (BETA, Inria), converted to xyz at load time with that dataset's
     stated convention (`polar_equator_radius`: BETA 0.5, Inria 0.36).
   GraspAndLift's polar table is ours (copied from a shared template), not the dataset's: it does not count.
2. **Frame alignment:** a dataset's positions come in its own frame and units. They are mapped onto MNE's head
   frame by one similarity transform per dataset (rotation + uniform scale + translation, least squares) fitted on
   its channels that have standard 10-10 / 10-05 names; the fit residual is reported per dataset (> 15 mm mean:
   flagged and the dataset falls back to the template). The dataset's own relative geometry is kept.
3. **Fallback:** a dataset with no coordinate source of its own uses MNE's template positions (standard_1020, then
   standard_1005), as in ADR 0023.

**Provenance rules.** Every coordinate already in a metadata.json (the BETA, Inria and GraspAndLift polar tables) is
re-checked against the dataset's actual source (the authors' location file, paper figure or official download): a
table that matches is kept, one that does not is corrected, one that cannot be traced to the source is flagged
"unverified" in the table below and in the report (and not used as the dataset's own). Newly fetched or corrected
coordinates go into the dataset's `gen_metadata.py` (read from the source file it ships or we fetch into `raw/`),
and metadata.json is regenerated from it, never hand-edited; the generator records the source file and its
convention.

Step 1 (before any training) fills in the table below by checking every tiny-corpus pretrain dataset and the three
finetune sets (BNCI2014001 / 004 / 008) for a coordinate source; the finetune sets are expected to have none (MOABB
attaches template montages) and then keep the template, exactly as the grid runs saw them.

| dataset | own coordinate source | verified against source | channels with own position | alignment residual |
|---|---|---|---|---|
| BETA_4s / BETA_3s | `data.suppl_info.chan` in every .mat (authors' table, EEGLAB polar, equator 0.5) | yes; identical over 8 subjects. Our old copy matched except Cz (source 0 / 0). CB1 / CB2 are 0 / 0 in the source (vertex placeholder for cerebellar sites): no position, dropped | 60 of 62 EEG | 5.5 mm |
| Wang2016 | `64-channels.loc`, the Tsinghua benchmark's file (fetched from bci.med.tsinghua.edu.cn; MOABB's copy lacks it), EEGLAB polar | yes (official file). Flag: M1 / M2 are left-right swapped in it (M1 at +90 deg); both are non-EEG and never read | 62 (incl. CB1 / CB2) | 6.4 mm |
| Liu2022EldBETA | BIDS `electrodes.tsv` inside the dataset's own archives (mm, x nose, y left) | yes; identical over 7 sessions of sub-001 and sub-023. Symmetric to 1e-8: a cap template the authors shipped, not a per-subject digitization | 60 | 5.5 mm |
| Inria_Train / Test | `ChannelsLocation.csv` shipped with the Kaggle data (idealised polar, equator 0.36) | yes (read from the shipped file); idealised 10-20 spacing | 56 | 7.4 mm |
| ERP_Longitudinal | `ChannelPosition.locs` (figshare 27201003, fetched; idealised polar, equator 0.406) | yes (the authors' file) | 57 | 7.2 mm |
| SPIS | `biosemi_64_besa_sph.besa` shipped with the data (BioSemi 64, BESA spherical, read by MNE) | yes (shipped file); a manufacturer template | 64 | 6.7 mm |
| Cho2017 | `eeg.senloc` in each subject's .mat: that subject's digitized positions (cm) | yes; differ between subjects by 1.7-3.5 cm (real digitization) | 64 per subject, 52 subjects | median 8.3 mm, max 17.1; subjects 10 (17.1) and 33 (15.3) exceed 15 mm -> template |
| GraspAndLift_Train / Test | none: its polar table was copied by us from a shared template (`_gen_eegmmidb_metadata.py`), not from the dataset | **unverified**, not used | 0 (template) | - |
| Lee2019_MI / SSVEP, Schirrmeister2017, Weibo2014, Dreyer2023, SRM_RestingState, MDD_Mumtaz, Neonatal_Helsinki, UCSD_PD, BCMI_MusicEmotion, BCIC2020-3, STEW | none found (no location file or field in the download; the BIDS sets ship no electrodes.tsv) | - | 0 (template) | - |
| BNCI2014001 / 004 / 008 (finetune) | none (MOABB attaches template montages) | - | 0 (template; loads bit-identically to grid, check 3) | - |

Checks (2026-10-01, before the runs): 1. grid default bit-identical (data, coordinates, valid masks); 2. residuals above,
montage plot `output/analysis/dcoord/montages.png` (to be copied into the report); 3. BNCI2014001 / 004 / 008 data,
coordinates, valid masks and labels bit-identical to grid. Only the coordinate keys of the 9 regenerated metadata.json
changed. Every finetune job's effective config equals its grid combined run's except paths, names and the two layout
keys (checked offline for all 54).

## Runs

- Pretrain: `mesae_tiny_p50_s16_dcoord_s1..3` = `mesae_tiny_p50_s16_s1..3` plus `preprocess_params.channel_layout:
  real` and `coords: dataset` (same corpus, window fraction, seeds, epochs).
- Finetune: combined head only (stamp_power learned rank 2 + latent_signed pca rank 2, spatial_k 8, dropout 0.3),
  the 6 cells (BNCI2014001 / 004 / 008 x loso / few-shot) x finetune seeds 1-3, the same protocols as the grid
  combined runs. The head is written into every cell config and each run's `artifacts/config.json` head is checked
  against the grid run's before any number is compared (the ADR 0023 comparison failed on exactly this).
- Reference: the existing grid `mesae_tiny_p50_s16_s1..3` combined runs.
- Also reported (not judged): backbone_eval masked MSE on each backbone's own held-out windows, the loso ridge probes
  (z, stamp power), and the coordinate-lookup jitter / mirror panel.

## Checks before the runs (all must pass)

1. **Identity:** with `coords: template`, the new code reproduces the grid and ADR 0023 `real` data bit-identically
   (restored code), and the grid default is untouched.
2. **Positions:** for every dataset with its own positions, the aligned positions of its standard-named channels lie
   within 15 mm (mean) of MNE's template; a plot of each such montage over the template head.
3. **Finetune input:** BNCI2014001 / 004 / 008 load bit-identically to grid (no own coordinates).

## Decision (loso first)

Per cell, mean over the 3 pretrain seeds (each the mean of its 3 finetune seeds), SE over pretrain seeds.

- **Adopt** (`real` + dataset coordinates becomes the default for the next corpus): beats grid on >= 1 loso cell
  (diff > 2 x sqrt(SE_a^2 + SE_b^2)) and loses none.
- **Keep grid:** loses any loso cell by that margin.
- **Neutral** (no loso cell differs beyond the margin): keep grid for now (the user's 10-10 choice stands); record the
  result for when the corpus grows with datasets that ship measured positions.
- Few-shot cells are reported, not judged.

## Result (2026-10-01): keep grid

BNCI2014004 loso +0.81 (margin 0.37, win), BNCI2014001 loso -1.74 (margin 1.69, loss), BNCI2014008 loso and all
few-shot cells level; consistent over the three seed pairs. One loso loss -> keep grid by the rule above. Report:
docs/reports/2026-10-01-dataset-coordinates.md.


## Follow-up (a): extra channels or own coordinates? (written 2026-10-01, before the runs)

The dcoord change bundles two things: the real layout's extra non-grid channels and the datasets' own positions. The
archived ADR 0023 backbones `mesae_tiny_p50_s16_real_s1..3` (real layout, template positions; same corpus and seeds)
isolate the first. They were never finetuned with the combined head (their ADR 0023 runs used protocol heads), so:
combined head, BNCI2014001 loso and BNCI2014004 loso, finetune seeds 1-3, effective configs checked against grid.

Reading (same margin rule, vs grid):
- real-template also loses 001 loso -> the extra channels cause it;
- real-template level with grid on 001 loso while dcoord loses -> the own coordinates cause it;
- in between (real-template within the margin of both grid and dcoord) -> unresolved, report the three means.
004 loso is read the same way for its gain.
