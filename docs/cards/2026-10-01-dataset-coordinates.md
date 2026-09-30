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
|---|---|---|---|
| (filled in Step 1) | | | | |

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
