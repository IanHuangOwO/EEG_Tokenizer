# Card: real channel coordinates (channel_layout real), correctness checks

Written 2026-09-30, before the code ran. ADR 0023.

- **Change:** `preprocess_params.channel_layout: real` keeps every EEG channel with its real coordinates (<= 64), or
  reduces a > 64-channel montage to the 64 canonical sites with spherical-spline interpolation for missing sites.
- **Claim:** the new path changes nothing for datasets whose channels are all canonical names, and interpolates
  missing sites at least as well as the nearest-neighbour baseline the eval already uses.

## Checks (all pre-set)

1. **Identity:** for BNCI2014001 / 004 / 008 (only canonical names) the `real` layout gives bit-identical data,
   coordinates and valid masks to `grid`, and one finetune stamp/z cache built through it is bit-identical.
2. **Gains:** per pretraining dataset, channels kept under `grid` vs `real` (a table; no pass/fail): which datasets
   gain channels, which are interpolated.
3. **Interpolation:** on Schirrmeister2017 (128 ch), for canonical sites it has: drop the site, interpolate it from
   the rest, compare with the real signal. Pass: spline relative error below inverse-distance-weighted (k = 4) error
   on average over sites.
4. **Pipeline:** an existing tiny backbone's `backbone_eval` run with a `real`-layout config completes and its
   masked MSEs are reported next to the `grid` ones (the corpus differs where datasets gained channels; not judged).

## Results (2026-09-30)

1. **Identity: pass.** BNCI2014001 / 004 / 008, all subjects: data, coordinates, valid masks and labels bit-identical
   to `grid`; a BNCI2014008 finetune feature cache (stamp codes and z) through `real` is bit-identical too. Needed one
   fix first: named channels are read in canonical-slot order (the per-trial z-score sums channels in read order, so
   metadata order gave float-rounding differences on BNCI2014008).
2. **Gains:** 11 pretrain datasets gain 2-8 channels (BCIC2020-3 +6, BETA_3s/4s +4, GraspAndLift +4, Lee2019_MI /
   SSVEP +8 (62 kept, all), Liu2022EldBETA / Wang2016 / Weibo2014 +2); Schirrmeister2017 keeps 126 channels and is
   reduced to 64 (> Nc). Found and fixed: 10-05 names (FFC1h, TPP9h, AFp3h, ...) had no position (MNE standard_1020
   only) -> standard_1005 fallback. Still without a position: CB1 / CB2 (Neuroscan cerebellar sites), dropped.
3. **Interpolation: fail as registered** (spherical spline mean relative error 0.348 vs IDW 0.311, 64 sites x 3
   subjects of Schirrmeister2017; spline better on 57% of sites). The recording reference Cz (flat, which the
   pipeline already treats as padding) and a noisy F7 dominate; excluding flat channels (the pipeline's rule): spline
   0.271 / median 0.101 vs IDW 0.213 / 0.091. **Post-test change (user, 2026-09-30):** the > 64 fill uses IDW
   (k = 4, 1/d^2) instead of splines.
4. **Pipeline: pass (completes).** `mesae_tiny_p50_s16_s2` (trained on `grid`) evaluated on `real`-layout held-out
   windows: token runs 0.4121 (grid 0.4074), random channel 0.3288 (0.3266), channel cluster 0.3525 (0.3461), time
   block 0.7788 (0.7794), motor-3 -> bci-22 0.2716 (0.2580). A grid-trained backbone sees channels it never trained on
   (+0-5%); whether a `real`-trained backbone does better is the next card (new tiny backbones on `real`).

## Follow-up: tiny backbones on `real` (2026-09-30), reverted

Combined head, mean tail balanced accuracy over 3 finetune seeds, per backbone seed:

| cell | grid s1 | real s1 | grid s2 | real s2 | real - grid |
|---|---|---|---|---|---|
| 001 loso | 50.5 | 39.7 | 49.3 | 40.0 | -10.0 |
| 004 loso | 77.0 | 74.9 | 77.4 | 73.5 | -3.0 |
| 008 loso | 69.3 | 69.2 | 69.2 | - | - |
| 001 few-shot | 41.3 | 45.7 | 40.7 | 45.0 | +4.3 |
| 004 few-shot | 75.4 | 75.1 | 75.6 | 76.3 | +0.2 |
| 008 few-shot | 58.9 | 58.9 | 60.2 | 59.3 | -0.5 |

001 loso is lower for all 9 held-out subjects on both seeds, and the head's train balanced accuracy drops too (~46% vs
~69%): the features, not generalisation. Loses two loso cells: fails the pre-set rule. Positions were MNE template
positions under both layouts (no metadata has measured ones), so this tested the extra channels + Schirrmeister2017
interpolation, not geometry. Reverted by the user; seed 3 pretrained but its finetunes stopped. ADR 0023 Outcome.
