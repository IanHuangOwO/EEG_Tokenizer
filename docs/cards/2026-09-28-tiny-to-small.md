# Mechanism card: tiny -> small corpus (4x pretraining data)

Written 2026-09-28, before the small run (ADR 0020). Evaluated automatically by
`output/queue/overnight/summarize_small.py`; report in `output/reports/overnight/small_card.md`.

- **Change:** the tiny winner's exact recipe (finest-only skips or graded drop-path, per the card
  above) trained on the small corpus (`window_fraction` 0.2: 4x the windows of every subject, nested
  over the tiny corpus; same subject split), one pretrain, seed 1: `mesae_small_<tag>_s1`.
- **Hypothesis:** more pretraining data makes the backbone better at what it is trained for -- filling
  masked content (in time and across channels), keeping visible detail -- and makes z more linearly
  usable, beyond run-to-run noise.

## Backbone metrics (primary)

Both models scored on the **tiny corpus's** held-out windows and masks (the small run is evaluated with
`window_fraction` 0.05, `output/queue/overnight/eval_tiny_windows.json`), so the comparison is paired.

| Metric | Better is | Tool |
|---|---|---|
| masked MSE: token runs, random channel, channel cluster, time block, motor3 -> bci22 | lower | `backbone_eval` |
| channel imputation vs inverse-distance interpolation (random channel, motor3 -> bci22): model / idw | lower | `backbone_eval` |
| relative band error: channel masks, time block, unmasked | lower | `backbone_eval` |
| seam disagreement | lower | `backbone_eval` |
| ridge probe, pre-stamp z, Compass loso: BNCI2014004 / 001 / 008 | higher | `ridge_probe` |

- **Spread:** per metric, graded vs `mesae_tiny_rankfix_s1` (near-identical tiny runs); no second pretrain
  seed at small scale.
- **Pass per metric:** small better than tiny by more than max(3 x spread, 2%).
- **Card passes** (scaling helps the backbone) if most metrics pass, **including** the masked MSEs and
  the ridge probe. A split result is recorded per metric; no single metric decides.

## Downstream (secondary; reported, not a gate)

- Both backbones: pre-stamp probe (`latent_signed`, pca, `spatial_k` 8) and the protocol stamp head
  with `spatial_k` 8 (instead of the protocols' 2: user, 2026-09-28), BNCI2014004 / 001 / 008,
  loso and few-shot with the Compass settings (004 session 3, 008 few-shot 5%).
- **3 finetune seeds** per cell (head init and batch order). Per cell: mean over seeds, sd over seeds
  (= head-training noise), small vs tiny paired Wilcoxon over subjects on the seed-averaged scores.
- Read as: a consistent direction across cells, beyond the seed sd, supports the backbone result.
  One pretrain seed per corpus size, so a downstream difference is not attributed to scale on its own.

## Known limits

- One pretrain per corpus size: the backbone spread comes from a different pair of tiny runs.
- The stamp-head `spatial_k` 8 is not the development-set-tuned protocol value.
