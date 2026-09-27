# 0019 — Masked-block STFT loss

Status: Rejected (2026-09-27, run `mesae_tiny_stft_s1`). The loss code was removed (a config setting
`loss.stft_weight` is rejected); its measure stays in `backbone_eval` as the masked log-spectral distance.
Date: 2026-09-27

## Context

Masked time-domain MSE sat around 0.5. For a masked rhythm whose phase cannot be inferred from
context, the MSE-optimal prediction is the phase average, which shrinks toward 0, so the loss gives
the encoder no reward for knowing a rhythm's band power. Measured on the graded model
(`backbone_eval` masked spectrum): masked predictions carry 0.10-0.53 of the true alpha/beta power,
and time blocks are worst (alpha 0.11, beta 0.10).

## Decision tested

Add a phase-blind target on the masked blocks (`MeSAE._stft_loss`): on the overlap-added trial,
multi-resolution (N = 32 / 64 / 128, Hann, hop N/4) mean `|log(|X^|+0.1) - log(|X|+0.1)|`, frames
weighted by their share of hidden samples, weight 0.2 (`loss.stft_weight`). Everything else as the
graded run (`mesae_tiny_skipdrop_graded_s1`); the per-patch mp ranking fix (`c5a0f0c`) also differs.

Pre-set rule: keep it if masked MSE is at most +3% and the pre-stamp probe gains >= 2 points on
average or the masked band power / temporal attention changes clearly, with no significant
downstream loss.

## Result

| | Graded | STFT |
|---|---|---|
| masked val MSE (ep 41-50) | 0.5277 | 0.5256 (pass) |
| unmasked val MSE | 0.128 | 0.142 (+11%) |
| masked log-spectral distance, channel masks | 0.85-0.89 | 0.58-0.62 |
| masked log-spectral distance, time blocks | 1.70 | 0.81 |
| masked alpha / beta power ratio, random channel | 0.53 / 0.38 | 0.60 / 0.48 |
| masked alpha / beta power ratio, time block | 0.11 / 0.10 | 0.14 / 0.18 |

Downstream, loso tail balanced accuracy (graded -> STFT):

| | pre-stamp probe (K 8) | stamp head |
|---|---|---|
| BNCI2014004 | 60.3 -> 58.5 (2/9, p 0.07) | 62.9 -> 60.2 (1/9, p 0.10) |
| BNCI2014001 | 47.1 -> 46.9 (4/9, p 0.82) | 44.5 -> 41.1 (1/9, p 0.027) |
| BNCI2014008 | 69.3 -> 68.7 (2/8, p 0.06) | 69.9 -> 70.2 (4/8, p 0.46) |

- The loss works as a loss: the masked log-spectral distance falls by about a third (by half on
  time blocks) and the masked power moves toward the truth, most in beta/gamma.
- The shrinkage is only partly undone: masked alpha/beta power stays at 0.14-0.60 of the truth.
- Temporal attention per block is unchanged from graded (`attention_range`).
- Downstream it does not help: no dataset gains, the BNCI2014001 stamp head drops significantly,
  and the other cells lean negative. Visible reconstruction is 11% worse.
- Stamp near-duplicates: 2 pairs >= 0.9 (graded 3), again among slow stamps.

## Consequences

Not in the recipe. Masked band power is a real, measurable gap (`backbone_eval` masked spectrum
keeps tracking it), but closing part of it this way did not reach the tasks on the tiny corpus.
Next: remove the trial term (ADR 0015, experiment 1) on the graded recipe.
