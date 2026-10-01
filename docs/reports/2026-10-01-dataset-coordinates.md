# Real layout with each dataset's own electrode coordinates (tiny, 3 seeds)

_Naming (2026-10-01): `channel_layout: real` is now `native` and `coords: dataset` is `recorded` (old values still
accepted); run names (`*_real_s*`, `*_dcoord_s*`) are unchanged. Text below uses the names of when it was written._

- **Date:** 2026-10-01
- **Question:** does pretraining on each dataset's own recorded electrode positions (plus its non-grid channels, the
  real layout) give a backbone that finetunes better than the 64-site 10-10 grid with MNE template positions?
- **Runs:** `mesae_tiny_p50_s16_dcoord_s1..3` (channel_layout real, coords dataset) vs `mesae_tiny_p50_s16_s1..3`
  (grid). Same corpus (tiny, 23 datasets), window fraction, pretrain seeds, epochs; nothing else differs.
- **Protocol:** combined head (stamp_power learned rank 2 + latent_signed pca rank 2, spatial_k 8, dropout 0.3), 6
  cells (BNCI2014001 / 004 / 008 x Compass loso / few-shot) x finetune seeds 1-3; metric tail balanced accuracy (and
  kappa). Per pretrain seed the mean of its 3 finetune seeds; a cell differs when |diff| > 2 x sqrt(SE_a^2 + SE_b^2)
  over pretrain seeds. Loso decides (card rule); few-shot reported.
- **Verdict:** keep grid over dcoord. Dataset coordinates win BNCI2014004 loso (+0.8) and lose BNCI2014001 loso (-1.7,
  just past the margin), both consistent over all three seed pairs; the other four cells are level. The pre-set rule
  (any loso loss -> keep grid) decides. Follow-up (a): the 001 loss comes from the own positions; the real layout
  with template positions wins 004 loso (+1.3) and is level on the other five cells: meets the adopt rule.
- **Cards / ADRs:** [card](../cards/2026-10-01-dataset-coordinates.md), [ADR 0023](../adr/0023-real-channel-coordinates.md),
  [coordinate-lookup card](../cards/2026-10-01-coordinate-lookup.md)

## Results

Tail balanced accuracy (%), mean +- SE over pretrain seeds (per-seed values in brackets):

| cell | grid | dcoord | diff | margin | verdict |
|---|---|---|---|---|---|
| BNCI2014001 loso | 49.5 +- 0.5 [50.5 49.3 48.9] | 47.8 +- 0.7 [49.2 47.4 46.9] | -1.74 | 1.69 | loss |
| BNCI2014004 loso | 77.1 +- 0.1 [77.0 77.4 77.1] | 78.0 +- 0.2 [78.0 78.2 77.7] | +0.81 | 0.37 | win |
| BNCI2014008 loso | 69.3 +- 0.0 [69.3 69.2 69.3] | 69.4 +- 0.2 [69.8 69.0 69.4] | +0.2 | 0.5 | level |
| BNCI2014001 few-shot | 40.9 +- 0.2 | 41.0 +- 0.2 | +0.1 | 0.6 | level |
| BNCI2014004 few-shot | 75.5 +- 0.1 | 75.4 +- 0.7 | -0.1 | 1.4 | level |
| BNCI2014008 few-shot | 59.5 +- 0.4 | 60.0 +- 0.2 | +0.5 | 0.8 | level |

Paired per seed (dcoord - grid): 001 loso -1.3 / -1.9 / -2.0; 004 loso +1.0 / +0.8 / +0.6. Kappa gives the same
picture (001 loso -2.3, 004 loso +1.6, 008 loso +0.6 just past its 0.5 margin).

Backbone metrics (reported, not judged; mean of 3 seeds):

| metric | grid | dcoord |
|---|---|---|
| loso ridge probe on z, 004 / 001 / 008 | 69.4 / 38.4 / 69.4 | 71.4 / 39.2 / 69.5 |
| masked MSE token_runs / random_channel / channel_cluster / time_block / motor3_to_bci22 | 0.403 / 0.323 / 0.343 / 0.779 / 0.258 | 0.410 / 0.318 / 0.348 / 0.778 / 0.261 |
| coordinate-embedding similarity vs closeness (Spearman) | 0.43-0.45 | 0.46 |
| ridge probe drop under 5 mm jitter / mirror (001, 004) | <= 0.3 / <= 0.3 (s1, s2) | <= 0.4 / <= 0.5 |

Masked MSE is measured on each backbone's own layout windows (dcoord windows carry the extra channels), so it is not a
like-for-like comparison. The frozen z probe is level or better for dcoord on every dataset, yet the finetuned combined
head loses 001 loso: the 001 loss is in what the SGD head extracts, not a visible loss in linear readability of z.

Coordinate sources (Step 1 of the card): BETA, Wang2016, Liu2022EldBETA, Inria, ERP_Longitudinal, SPIS (dataset or
manufacturer tables) and Cho2017 (per-subject digitized); alignment residuals 5.5-8 mm, two Cho2017 subjects above
15 mm fell back to the template; 11 datasets and the finetune sets have no source and kept the template.
`montages.png` shows every source after alignment over the template.

## Follow-up (a): extra channels vs own positions

The archived ADR 0023 backbones (real layout, template positions, same corpus and seeds) finetuned with the combined
head on the two MI loso cells:

| cell | grid | real, template positions | dcoord |
|---|---|---|---|
| BNCI2014001 loso | 49.5 +- 0.5 | 49.4 +- 0.3 | 47.8 +- 0.7 |
| BNCI2014004 loso | 77.1 +- 0.1 | **78.5 +- 0.2** | 78.0 +- 0.2 |

The 001 loss comes from the datasets' own positions (real-template is level with grid, dcoord is below real-template
by more than the margin); the extra channels alone are neutral on 001 and win 004 by +1.3 (kappa +2.7). Real layout
with template positions on all six cells (tail balanced accuracy, vs grid): 001 loso 49.4 (-0.17, level), 004 loso
78.5 (+1.34, margin 0.39, win), 008 loso 69.5 (+0.22, level), few-shot 40.8 / 76.5 / 59.7 (all level). One loso win,
no loss: it meets the adopt rule (`real_template_results.json`).

## Notes

- Only Cho2017 has truly measured, per-subject positions; the other six sources are cap or idealised templates the
  authors shipped. The change is therefore mostly "different templates per dataset" plus the extra non-grid channels,
  and the two effects are not separated here.
- The finetune sets (BNCI2014001 / 004 / 008) have no own coordinates and load bit-identically to grid; every finetune
  job's effective config equals its grid counterpart's apart from paths and the layout keys (checked offline for all
  54 and after the runs: 0 head mismatches). The earlier ADR 0023 comparison failed exactly here.
- The 001 loso loss is at the edge of the margin (1.74 vs 1.69) but has the same sign on all three seed pairs.
- `channel_layout: real` and `coords: dataset` stay in the code as options; grid with template coordinates stays the
  default (bit-identical to before).

## Files

- `results.json`: per cell and metric (tail, kappa_tail) the per-pretrain-seed values, diff, margin and verdict.
- `backbone_metrics.json`: masked MSE per mask and ridge probe per dataset, per seed.
- `montages.png`: aligned own positions vs the MNE template for every dataset with a source.
- `dcoord.plan`: the queue (pretrain commands and the 54 finetune commands with their --set head).
- Raw: `output/mesae_tiny_p50_s16_dcoord_s{1,2,3}/` (pretrain/analysis, finetune/combined/*/artifacts).
