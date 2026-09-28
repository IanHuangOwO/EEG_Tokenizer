# Overnight pipeline: finest-only skips, tiny to small corpus, trial windows

- **Date:** 2026-09-28
- **Question:** (1) do finest-only skips beat graded skip drop-path? (2) does the winner's recipe improve
  on 4x the pretraining data? (3) do EEG-FM-Compass post-event trial windows change downstream accuracy?
- **Runs:**
  - `mesae_tiny_notrial_s1` (graded drop-path, the tiny reference; live) vs `mesae_tiny_finestskip_s1`
    (archived); spread = graded vs `mesae_tiny_rankfix_s1`
  - `mesae_small_graded_s1` (same recipe, `window_fraction` 0.2; live)
- **Protocol:** unattended pipeline `scripts/run.sh`, every step gated on the previous exit code.
  Backbone metrics on the tiny corpus's held-out windows (`scripts/eval_tiny_windows.json`), identical
  masks. Downstream: pre-stamp probe (`latent_signed`, pca, spatial_k 8) and stamp head (spatial_k 8),
  loso and few-shot on BNCI2014004 / 001 / 008, finetune seeds 1-3, tail balanced accuracy, paired
  Wilcoxon over subjects.
- **Verdict:** graded drop-path stays (finest-only fails the time-block test); the small corpus improves
  11 / 14 backbone metrics and 10 / 12 downstream cells; Compass windows help the MI few-shot stamp head
  (+5.2 on BNCI2014004, +2.5 on BNCI2014001) and change nothing else beyond noise.
- **Cards / ADRs:** [finest-skip](../cards/2026-09-28-finest-skip.md),
  [tiny-to-small](../cards/2026-09-28-tiny-to-small.md); ADR 0020.

## Results

### 1. Finest-only skips vs graded drop-path (tiny)

Pre-set rule: finest wins only if all three pass (thresholds = max(3 x spread, 5%)).

| Test | Change vs graded | Threshold | Pass |
|---|---|---|---|
| channel-mask band error | -27.8% | < -7.6% | yes |
| unmasked band error | -93.4% | < -26.8% | yes |
| time-block masked MSE | +13.9% | < +3.1% | no |

| Run | masked MSE token / channel / cluster / time | seam | ridge probe 004 / 001 / 008 |
|---|---|---|---|
| finest-only | 0.303 / 0.251 / 0.276 / 0.888 | 0.014 | 65.8 / 34.1 / 57.9 |
| graded (notrial) | 0.405 / 0.318 / 0.339 / 0.780 | 0.066 | 69.4 / 38.7 / 69.8 |
| graded (skipdrop_graded) | 0.397 / 0.316 / 0.336 / 0.776 | 0.110 | 71.3 / 37.3 / 70.1 |
| rankfix | 0.412 / 0.317 / 0.338 / 0.784 | 0.108 | 69.8 / 37.8 / 69.8 |
| no drop-path (archived baseline) | 0.341 / 0.252 / 0.277 / 0.858 | 0.083 | - |

### 2. Tiny to small corpus

Pass: better than tiny by > max(3 x spread, 2%).

| Metric | Tiny | Small | Change | Spread | Pass |
|---|---|---|---|---|---|
| masked MSE token_runs | 0.4052 | 0.3291 | -18.8% | 3.7% | yes |
| masked MSE random_channel | 0.3180 | 0.2961 | -6.9% | 0.2% | yes |
| masked MSE channel_cluster | 0.3395 | 0.3143 | -7.4% | 0.5% | yes |
| masked MSE time_block | 0.7795 | 0.6993 | -10.3% | 1.0% | yes |
| masked MSE motor3_to_bci22 | 0.2561 | 0.2397 | -6.4% | 1.1% | yes |
| random_channel MSE / IDW | 1.141 | 1.062 | -6.9% | 0.2% | yes |
| motor3_to_bci22 MSE / IDW | 1.134 | 1.061 | -6.4% | 1.1% | yes |
| band error, channel masks | 0.3666 | 0.3411 | -6.9% | 0.9% | yes |
| band error, time block | 0.7688 | 0.7464 | -2.9% | 0.4% | yes |
| band error, unmasked | 0.1293 | 0.1303 | +0.8% | 7.0% | no |
| seam disagreement | 0.0660 | 0.0517 | -21.7% | 2.1% | yes |
| ridge probe BNCI2014004 | 69.42 | 71.92 | +3.6% | 2.1% | no |
| ridge probe BNCI2014001 | 38.73 | 41.67 | +7.6% | 1.3% | yes |
| ridge probe BNCI2014008 | 69.81 | 69.55 | -0.4% | 0.4% | no |

Downstream, 3 finetune seeds (mean, sd over seeds):

| Cell | Head | Tiny | Small | Diff | Subjects better | p |
|---|---|---|---|---|---|---|
| BNCI2014004 loso | probe | 71.2 (0.6) | 75.5 (0.2) | +4.2 | 8/9 | 0.008 |
| BNCI2014004 loso | stamp | 76.8 (0.4) | 76.0 (0.4) | -0.8 | 3/9 | 0.516 |
| BNCI2014004 fewshot | probe | 62.6 (0.6) | 65.9 (1.0) | +3.3 | 6/9 | 0.094 |
| BNCI2014004 fewshot | stamp | 72.2 (0.1) | 71.2 (0.7) | -1.0 | 4/9 | 0.426 |
| BNCI2014001 loso | probe | 46.2 (0.1) | 49.6 (0.1) | +3.4 | 8/9 | 0.012 |
| BNCI2014001 loso | stamp | 40.4 (0.7) | 43.8 (1.1) | +3.4 | 8/9 | 0.008 |
| BNCI2014001 fewshot | probe | 30.0 (1.2) | 34.5 (0.6) | +4.5 | 9/9 | 0.004 |
| BNCI2014001 fewshot | stamp | 42.5 (0.4) | 45.1 (0.9) | +2.6 | 6/9 | 0.164 |
| BNCI2014008 loso | probe | 69.0 (0.1) | 69.6 (0.1) | +0.6 | 5/8 | 0.250 |
| BNCI2014008 loso | stamp | 70.2 (0.2) | 70.8 (0.2) | +0.7 | 7/8 | 0.016 |
| BNCI2014008 fewshot | probe | 61.8 (0.4) | 62.5 (0.3) | +0.7 | 4/8 | 0.383 |
| BNCI2014008 fewshot | stamp | 58.9 (0.5) | 59.9 (0.7) | +1.0 | 6/8 | 0.383 |

### 3. Trial windows: ours vs EEG-FM-Compass (small backbone, 3 finetune seeds)

| Cell | Head | Old windows | Compass windows | Diff | Subjects better | p |
|---|---|---|---|---|---|---|
| BNCI2014004 loso | probe | 75.5 | 74.9 | -0.6 | 3/9 | 0.359 |
| BNCI2014004 loso | stamp | 76.0 | 75.5 | -0.5 | 5/9 | 1.000 |
| BNCI2014004 fewshot | probe | 65.9 | 68.7 | +2.8 | 4/9 | 0.570 |
| BNCI2014004 fewshot | stamp | 71.2 | 76.4 | +5.2 | 7/9 | 0.020 |
| BNCI2014001 loso | probe | 49.6 | 49.5 | -0.2 | 3/9 | 0.910 |
| BNCI2014001 loso | stamp | 43.8 | 43.0 | -0.8 | 3/9 | 0.359 |
| BNCI2014001 fewshot | probe | 34.5 | 35.4 | +0.9 | 5/9 | 0.312 |
| BNCI2014001 fewshot | stamp | 45.1 | 47.7 | +2.5 | 8/9 | 0.039 |
| BNCI2014008 loso | probe | 69.6 | 69.1 | -0.5 | 2/8 | 0.312 |
| BNCI2014008 loso | stamp | 70.8 | 70.4 | -0.4 | 1/8 | 0.023 |
| BNCI2014008 fewshot | probe | 62.5 | 60.7 | -1.8 | 2/8 | 0.195 |
| BNCI2014008 fewshot | stamp | 59.9 | 60.5 | +0.6 | 6/8 | 0.547 |

## Notes

- One pretrain seed per backbone; the spread is one near-identical pair (graded vs rankfix).
- Probe maps (old windows) show the MI probe reading 0.1-0.4 s after the cue on parieto-occipital
  channels: a possible cue-evoked shortcut, not yet checked.
- The stamp-head results used routed-era code paths that have since been removed (ADR 0022); the
  heads used static stamps only, so the numbers stand.

## Files

`2026-09-28-overnight/`: `probe_maps_<dataset>_<split>.png` (probe decision weights, old windows) and
`scripts/` -- the pipeline as it ran (`run.sh`, `decide.py`, `ft_plan.py`, `summarize_small.py`,
`windows.sh`, `compare_windows.py`, `eval_tiny_windows.json`) with its logs (`overnight.log` step log, `windows.log`, `analysis_mesae_small_graded_s1.log`). Paths
inside the scripts point at the former `output/queue/overnight/`.
